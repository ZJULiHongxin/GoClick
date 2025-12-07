# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
"""
model_name=microsoft/Florence-2-large
data_path=/data/hongxin_li/scaling_exp/UIPro_processed/LlamaFactory_data/UIPro_dedup_8MQAs_v3_SingleTurn_woWAE_florence_6484k_B_main_0D4MultiUIIntentGnd.jsonl
output_dir=/mnt/shared-storage/groups/stepone_mm/lhx/goclick_ckpts/0311_GoClick-Florence2Large_AfterCoarse-Main_woMultiUIIntentGnd

rlaunch --gpu 8 --cpu 64 --memory 800000  --positive-tags feature/gpfs=yes --group pretrain2 -- torchrun --nproc_per_node 8 --nnodes 1 --master_port 16252 florence2/finetune.py --model_name_or_path $model_name --florence_path $model_name --data_path $data_path --bf16 True --fix_vit True --output_dir $output_dir --num_train_epochs 1 --per_device_train_batch_size 48 --per_device_eval_batch_size 2 --gradient_accumulation_steps 1 --eval_strategy no --save_strategy epoch --save_total_limit 5 --learning_rate 1e-4 --weight_decay 0.1 --adam_beta2 0.95 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 2 --report_to none --run_name 0210_UIPro-Florence-Large --model_max_length 1024 --lazy_preprocess True --s3 's3://guidata-lhx'
"""

from dataclasses import dataclass, field
import json
import megfile
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import transformers
from transformers import TrainerCallback, AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers import Trainer, GPTQConfig
try:
    from transformers import deepspeed
except:
    from transformers.integrations import deepspeed
from io import BytesIO
from PIL import Image
from transformers.trainer_pt_utils import LabelSmoother
#from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from accelerate.utils import DistributedType
from colorama import Style, Fore
import re
import tqdm
from typing import Dict, Optional, Sequence, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/Florence-2-large")
    florence_path: Optional[str] = field(default=None)
    image_size: int = 448

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True
    s3: str = '' # 's3://guidata-lhx'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    remove_unused_columns: bool = False # Allow custom attributes appear to the batch.
    ddp_find_unused_parameters: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"]  ##["in_proj","out_proj","c_fc"]
    ),
    modules_to_save: List[str] = field(
        default_factory=lambda: [] # lambda: ["wte", "lm_head", "transformer.wte"]  ##["in_proj","out_proj","c_fc"]
    )
    modules_to_save: List[str] = field(
        default_factory=lambda: [] # lambda: ["wte", "lm_head", "transformer.wte"]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def clean_dict(data):
    if isinstance(data, dict):
        return {k: clean_dict(v) for k, v in data.items() if is_serializable(v)}
    elif isinstance(data, list):
        return [clean_dict(item) for item in data if is_serializable(item)]
    else:
        return data if is_serializable(data) else None

def print_trainable_params(model: torch.nn.Module):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param)

def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def clean_dict(data):
    if isinstance(data, dict):
        return {k: clean_dict(v) for k, v in data.items() if is_serializable(v)}
    elif isinstance(data, list):
        return [clean_dict(item) for item in data if is_serializable(item)]
    else:
        return data if is_serializable(data) else None

def print_trainable_params(model: torch.nn.Module):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, processor: transformers.PreTrainedTokenizer, max_len: int, s3: str = ''):
        super(LazySupervisedDataset, self).__init__()

        self.max_len = max_len
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.s3 = s3
        self.Broom = os.path.exists("/mnt/shared-storage/groups/pretrain2/")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        question, answer = self.raw_data[i]['messages'][0]['content'].replace("<image>","").strip(), self.raw_data[i]['messages'][1]['content']
        #print(f"{i} User: {question} || GPT: {answer}")
        image_file = self.raw_data[i]['images'][0]
        
        if self.s3:
            image_file = image_file.replace("test_samples/",'')
            with megfile.smart_open(os.path.join(self.s3, image_file.split('ui_data/')[-1]), 'rb') as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")
        else:
            image = Image.open(image_file if self.Broom else image_file.replace("pretrain2", "stepone_mm")).convert("RGB")

        inputs = self.processor(text=[question], images=[image], return_tensors="pt", padding=True)
        labels = self.tokenizer(text=[answer], return_tensors="pt", padding=True, return_token_type_ids=False).input_ids

        # truncate
        labels = labels[:,:self.max_len-576-inputs["input_ids"].shape[1]]
        ### Print supervised substrings to debug
        # print(self.tokenizer.decode(inputs["input_ids"][0]))
        
        # print(self.tokenizer.decode(labels[0]))
        ret = dict(
            input_ids=inputs["input_ids"][0],
            pixel_values=inputs["pixel_values"][0],
            labels=labels[0]
        )
        self.cached_data_dict[i] = ret

        if len(self.cached_data_dict) > 1024:
            self.cached_data_dict.pop(random.choice(list(self.cached_data_dict.keys())))   
        return ret

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, pixel_values, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "pixel_values", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
        pixel_values = torch.stack(pixel_values)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        return batch

def make_supervised_data_module(
        processor: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    if data_args.data_path.endswith('.json'):
        with open(data_args.data_path, "r") as f:
            train_json = json.load(f)
    else:
        train_json = []
        with open(data_args.data_path, "r") as f:
            for line in f:
                train_json.append(json.loads(line))

    print(f"Loading {len(train_json)} samples from {data_args.data_path}.")
    train_dataset = LazySupervisedDataset(train_json, processor=processor, max_len=max_len, s3=data_args.s3)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = LazySupervisedDataset(eval_json, processor=processor, max_len=max_len, s3=data_args.s3)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator = data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    #rank0_print(Fore.YELLOW + f"Image size: {config.visual['image_size']}" + Style.RESET_ALL)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        # cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        attn_implementation = "flash_attention_2"
    )
    
    # customized LoRA parameters
    target_modules = []
    target_layer_names = ["visual.conv1", "attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj", "c_attn",
                          "attn.c_proj", "w1", "w2"]
    excluded_module_names = ['post_qformer']
    lora_supported_types = [torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D]
    
    for name, module in model.named_modules():
        if all([m_name not in name for m_name in excluded_module_names]) and any(t_name in name for t_name in target_layer_names) and 'attn_pool' not in name:
            if isinstance(module, tuple(lora_supported_types)):
                target_modules.append(name)
            else:
                print(name + " not satisfy lora")
                break
                # input()
    
    lora_args.lora_target_modules = target_modules
    # else:
    #     lora_args.modules_to_save = []
    # """
    # # print the LoRA parameters
    # for name, param in model.named_parameters():
    #     if any(target in name for target in lora_args.lora_target_modules):
    #         print(name)
    # """

    if not training_args.use_lora:
        if training_args.fix_vit:
            model.vision_tower.requires_grad_(False)
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = lora_args.modules_to_save
        else:
            modules_to_save = lora_args.modules_to_save #["wte", "lm_head"]

        already_lora = model_args.model_name_or_path != model_args.florence_path

        if already_lora: # Resume from LoRA. Reference: https://discuss.huggingface.co/t/loading-peft-model-from-checkpoint-leading-into-size-missmatch/71944
            rank0_print(Fore.YELLOW + "Resume LoRA finetuning from the config in" + model_args.model_name_or_path + Style.RESET_ALL)
            lora_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        else:
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save  # This argument serves for adding new tokens. # 除了lora部分外，还有哪些层可以被训练，并且需要保存；
            )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        if already_lora:
            model = PeftModel.from_pretrained(model, 
                model_args.model_name_or_path,
                is_trainable=True # 👈 here
                ) 
        else:       
            model = get_peft_model(model, lora_config)
        
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            model._set_grad_checkpointing()

    if local_rank == 0:
        try:
            trainable_params, all_param = model.get_nb_trainable_parameters()

            param_info = f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        except:
            param_info = print_trainable_params(model)
        print(param_info)
        
        # Save the experiment configurations
        exp_config = {
            'model_args': clean_dict(vars(model_args)),
            'data_args': clean_dict(vars(data_args)),
            'training_args': clean_dict(vars(training_args)),
            "trainable_params_info": param_info,
            "num_gpus": torch.cuda.device_count()
        }
        if training_args.use_lora: exp_config['lora_args'] = clean_dict(vars(lora_args))

        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "exp_config.json"), "w") as f:
            json.dump(clean_dict(exp_config), f, indent=2)
        
        # save the Florence processor
        processor.save_pretrained(training_args.output_dir)

        if training_args.use_lora:
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                        model.named_parameters()
                    )
            print(Fore.CYAN + ", ".join(name for name in non_lora_state_dict) + Style.RESET_ALL)
            print("The above params will be saved as non-lora_trainable state dict")

    # Load data
    data_module = make_supervised_data_module(
        processor=processor, data_args=data_args, max_len=training_args.model_max_length
    )

    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            processor.save_pretrained(checkpoint_dir)
            model.module.save_pretrained(checkpoint_dir)
            
            if args.use_lora:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), lora_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                if training_args.local_rank in [0,-1]:
                    model.config.save_pretrained(checkpoint_dir)
                    model.save_pretrained(checkpoint_dir, state_dict=state_dict)

                    torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))

    # Start trainner
    trainer = Trainer(
        model=model, args=training_args, callbacks=[SaveCallback()] if training_args.use_lora else None, **data_module
    ) 
            
    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()

    print(f"Save to {training_args.output_dir}")

    if training_args.use_lora:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)

            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)

import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
if __name__ == "__main__":
    train()
