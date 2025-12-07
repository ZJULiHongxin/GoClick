# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score

import os, time, cv2
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import datasets
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from pprint import pprint
from colorama import Fore, Style
from copy import deepcopy
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.task_prompt_lib import *
from utils.eval_utils.action_matching import *
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.autogui_florence2 import AutoGUI_Florence
from utils.data_utils.misc import lower_first_letter, pred_2_point
from utils.openai_utils.misc import extract_thought_components, extract_thought_components_llamav

logging.basicConfig(level=logging.INFO)

def scroll2swipe(direction):
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'

# calculate action f1 following androidcontrol
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--andcon_dir', type=str, default='/mnt/vdb1/hongxin_li/AndroidControl_test')
parser.add_argument('--planning_result_file', type=str, default='utils/eval_utils/eval_andcon_with_funcgnd/eval_results/gpt-4o/2025-12-07-10-36-27.json')
parser.add_argument('--grounder', type=str, default=[
    'gpt-4o-mini', # provider: openai
    '/mnt/vdb1/hongxin_li/uipro_ckpt/0102_Qwen2vl-7B-490kIntentGnd/lora/checkpoint-3832', # provider: qwen2-vl
    'cckevinn/SeeClick', # provider: qwenvl
    'OS-Copilot/OS-Atlas-Base-7B', # provider: qwen2-vl
    'Samsung/TinyClick', # provider: tinyckick
    'THUDM/cogagent-chat-hf', # provider: cogagent
    '/mnt/vdb1/hongxin_li/goclick_ckpts/0314_GoClick-Florence2Large_CoreSet-v2_3814k/checkpoint-29800/' # provider: autogui_florence
    ][-1]) # /mnt/nvme0n1p1/hongxin_li/highres_autogui/checkpoints/1019_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+AutoGUI1p3M-Android68k/ , /mnt/vdb1/hongxin.li/uipro_ckpt/1030_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+UIPro18p6M/checkpoint-32656s

parser.add_argument('--provider', type=str, default=['openai', 'qwen2-vl', 'qwenvl', 'qwen2-vl', 'tinyckick', 'cogagent', 'autogui_florence'][-1])

parser.add_argument('--debug', action="store_true")
parser.add_argument('--max_prev_acts', type=int, default=6)
parser.add_argument('--device_tag', type=str, default='Android')

args = parser.parse_args()

if args.provider == 'openai':
    openai = OpenAIModel(model=args.grounder,
                        base_url=os.environ.get("OPENAI_BASE_URL", "https://xiaoai.plus/v1/"),
                        api_key=os.environ.get("OPENAI_API_KEY", ""),
                        temperature=0.0)
    postfix = openai.model.replace("/","-")

    if 'gpt' in openai.model:
        PROMPT = FUNCGND_PROMPT_GPT
    elif 'qwen2' in openai.model:
        PROMPT = FUNCGND_PROMPT_QWEN2VL
    
    USE_FUNCDESC = True
    SCALE = 1000
else:
    model_path = args.grounder.rstrip('/')
    print(f"Loading grounder from {args.grounder}")
    if 'atlas' in model_path.lower():
        vlm = QWen2VL(device='cuda', model_name=model_path)
        PROMPT = '"In this UI screenshot, what is the position of the element corresponding to the command "{}" (with bbox)?"'
        SCALE = 1000
        USE_FUNCDESC = False
    elif 'seeclick' in model_path.lower():
        vlm = QwenVL(device='cuda', model_name=model_path)
        PROMPT = 'In the UI, where should I click if I want to {} (with point)?'
        SCALE = 1
        USE_FUNCDESC = False
    elif 'tinyclick' in model_path.lower():
        vlm = TinyClick(device='cuda', model_name=model_path)
        PROMPT = 'What to do to execute the command? click on {}'
        SCALE = 1000
        USE_FUNCDESC = True
    elif 'cogagent' in model_path.lower():
        vlm = CogAgent(device='cuda', model_name=model_path)
        PROMPT = 'Generate the target element according to the UI screenshot, instruction. Please provide the answer directly (with grounding). Instruction: {}'
        SCALE = 1000
        USE_FUNCDESC = False
    elif 'florence' in model_path.lower():
        vlm = AutoGUI_Florence(device='cuda', model_name=model_path)
        SCALE = 1000
        USE_FUNCDESC = True#'intent' not in model_path.lower()
        if USE_FUNCDESC:
            PROMPT = FUNCGND_PROMPT
        else:
            PROMPT = intentgnd_prompt[0]
    else:
        USE_FUNCDESC = 'intent' not in model_path.lower()
        vlm = QWen2VL(device='cuda', model_name=model_path)
        if USE_FUNCDESC:
            PROMPT = FUNCGND_PROMPT + ' (with point)'
        else:
            PROMPT = intentgnd_prompt[0]
        SCALE = 1000

    model_path = model_path.replace("lora/","").replace("merged/","")
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    elif len(model_path.split('/')) == 2:
        postfix = model_path.replace('/', '--')
    elif 'checkpoint-' in model_path:
        postfix = '-'.join(model_path.split('/')[-2:])

print(f"Loading eval file: {args.planning_result_file} | Gnd by {postfix}")
planning_eval_results_info = json.load(open(args.planning_result_file))
meta = planning_eval_results_info["meta"]
planning_eval_results_all_repeats = planning_eval_results_info["logs"]


androidcontrol_test_raw = json.load(open(f'{args.andcon_dir}/AndroidControl-test_12685.json', 'r'))

# 筛选出HL样本
hl_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' in x['id'])
h_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' not in x['id'])
hl_h_ids = hl_ids.intersection(h_ids)


REPEAT = 3
results_all_repeats = []
metrics_all_repeats = []
for repeat, planning_eval_results in enumerate(planning_eval_results_all_repeats):
    # HL
    metrics_this_repeat = {'H': []}
    results = {'H': []}
    for mode, samples in planning_eval_results.items():
        if mode == 'HL': continue
        counts = {'total': 0, 'click': 0, 'input_text': 0, 'swipe': 0, 'long_press': 0, 'enter': 0, 'navigate_home': 0, 'navigate_back': 0, 'status': 0, 'open_app': 0, 'wait': 0}

        for step_idx, step in tqdm(enumerate(samples), total=len(samples), desc=f'Repeat {repeat+1} {mode} | {"Func gnd" if USE_FUNCDESC else "Intent Gnd"} '):
            goal = step["task"]
            counts['total'] += 1
            action_type_ref = step['action_type']

            counts[action_type_ref] += 1

            img_path = os.path.join(args.andcon_dir.replace("AndroidControl_test","").rstrip('/'), step["image"])

            image = Image.open(img_path)
            W,H=image.size
            # Used in the prompt

            raw_prompt, raw_response, new_action, response = '', None, None, None

            t1 = time.time()

            step_result = step | {"grounder_prompt": None, "grounder_response": None, "grounder_action_pred": None, "new_metrics":  {k: False for k in ['action_match', 'type_match', 'elem_acc', 'click_match', 'input_text_match', 'swipe_match', 'enter_match', 'status_match', 'navigate_home_match', 'navigate_back_match', 'open_app_match', 'wait_match', 'long_press_match', 'need_gnd']} }

            try:
                funcdesc = step.get('funcdesc','').strip(' .') # if the original response's format is wrong, this will also be recorded as a wrong format error.
                intent = step.get('summary','').replace('I ','').strip(' .')
                
                raw_prompt, raw_response = step['prompt'], step['response']
                new_action = deepcopy(step['action_pred'])

                if step['task_attr']['action_type'] in ['click', 'long_press'] and step['action_pred']['action_type'] == step['task_attr']['action_type'] and not (USE_FUNCDESC and 'None' in funcdesc):
                    prompt = PROMPT.format(funcdesc if USE_FUNCDESC else lower_first_letter(intent))
                    if args.provider == 'openai':
                        _, response, _ = openai.get_model_response(prompt, [img_path], use_img_url=True)
                    else:
                        response = vlm.get_model_response(prompt, f"file://{img_path}")
            
                    real_target = pred_2_point(response, keep_box=False, scale=SCALE)

                    step_result['grounder_prompt'] = prompt
                    step_result['grounder_response'] = response
                    new_action['target'] = real_target
                else:
                    'Not click'
            except Exception as e:
                print(e)

            try:
                step_result["grounder_response"] = response
                
                action_pred = step_result["grounder_action_pred"] = new_action
                
                # matching
                action_type_pred = action_pred['action_type']

                # special hanlding for enter
                if action_type_ref == 'enter' and action_pred['action_type'] == 'press_key' and action_pred['key'].lower() == 'enter':
                    enter_match = True
                else:
                    enter_match = False

                if action_type_ref == action_type_pred or enter_match:
                    step_result['new_metrics']['type_match'] = True

                    if action_type_ref in ['click', 'long_press']:
                        step_result['new_metrics']['need_gnd'] = True
                        target_pred = action_pred['target']
                        
                        gt_box_normalized = step['task_attr']['bbox'] # has been normalized during planning
                        
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            step_result['new_metrics']['action_match'] = step_result['new_metrics']['elem_acc'] = step_result['new_metrics'][f'{action_type_ref}_match'] = True

                    elif action_type_ref == 'input_text':
                        text_ref, text_pred = step['task_attr']['text'].lower().strip(), action_pred['text'].lower().strip()
                        
                        step_result['new_metrics']['action_match'] = step_result['new_metrics']['input_text_match'] = squad_metrics.compute_f1(text_pred, text_ref) > 0.5
                    
                    elif action_type_ref == 'swipe':
                        direction_ref, direction_pred = step['task_attr']['direction'], action_pred['direction']
                        direction_ref = scroll2swipe(direction_ref)
                        if direction_ref == direction_pred:
                            step_result['new_metrics']['action_match'] = step_result['new_metrics']['swipe_match'] = True
                    
                    elif action_type_ref == 'status':
                        status_ref, status_pred = step['task_attr']['goal_status'], action_pred['goal_status']
                        
                        if status_ref == status_pred:
                            step_result['new_metrics']['action_match'] = step_result['new_metrics']['status_match'] = True
                    elif action_type_ref == 'open_app':
                        app_name_ref, app_name_pred = step['task_attr']['app_name'], action_pred['app_name']
                        
                        if app_name_ref == app_name_pred:
                            step_result['new_metrics']['action_match'] = step_result['new_metrics']['open_app_match'] = True
                    else:
                        step_result['new_metrics']['action_match'] = step_result['new_metrics'][f'{action_type_ref}_match'] = True
                logging.info(f"Old: {step_result['metrics']['action_match']} => New: {step_result['new_metrics']['action_match']} | GT: {step['task_attr']} <=> Pred: {action_pred}")

                if step_idx % 2 == 0:
                    print(Fore.YELLOW + f"\nUser: <img>{img_path}</img> {prompt}\n" + Fore.CYAN + f"GPT: {response}" + Style.RESET_ALL)
            except Exception as e:
                print(e)
                logging.info("format wrong")
                step_result['wrong_format'] = True

            results[mode].append(step_result)

        # calculate metrics
        num_sample = counts['total']
        num_need_gnd = sum(x['new_metrics']['need_gnd'] for x in results[mode])
        
        num_action_match = sum(x['new_metrics']['action_match'] for x in results[mode])
        num_type_match = sum(x['new_metrics']['type_match'] for x in results[mode])
        num_elem_match = sum(x['new_metrics']['elem_acc'] for x in results[mode])
        final_metrics = {'step_acc': [num_action_match / num_sample, num_action_match, num_sample], 'action_type_acc': [num_type_match / num_sample, num_type_match, num_sample], 'elem_acc': [num_elem_match / num_need_gnd if num_need_gnd else 0, num_elem_match, num_need_gnd]}

        for k in counts.keys():
            if k=='total': continue
            
            cnt = counts[k]
            acc_cnt = sum(x['new_metrics'][f'{k}_match'] for x in results[mode])
            
            final_metrics[f'{k}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

        final_metrics['num_wrong_format'] = sum(1 for x in results[mode] if 'wrong_format' in x)

        pprint(final_metrics)
    
        metrics_this_repeat[mode] = final_metrics
    results_all_repeats.append(results)
    metrics_all_repeats.append(metrics_this_repeat)
# aggr
aggr_metrics = {'H': {}}

for mode in aggr_metrics.keys():
    for repeat_result in metrics_all_repeats:
        for metric_name, info in repeat_result[mode].items():
            if metric_name == 'num_wrong_format': continue
            if metric_name not in aggr_metrics[mode]: aggr_metrics[mode][metric_name] = [0,0,0]
            aggr_metrics[mode][metric_name][1] += info[1]
            aggr_metrics[mode][metric_name][2] += info[2]
    
    for metric_name in aggr_metrics[mode].keys():
        if metric_name == 'num_wrong_format': continue
        acc_cnt, cnt = aggr_metrics[mode][metric_name][1], aggr_metrics[mode][metric_name][2]
        aggr_metrics[mode][metric_name][0] = acc_cnt / cnt if cnt > 0 else 0

print("\nFinal:")
pprint(aggr_metrics)

eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results')

save_to = os.path.join(eval_result_dir, postfix)

os.makedirs(save_to, exist_ok=True)

time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_file = args.planning_result_file.replace(".json", f"_RevisedBy_{postfix}_{'FuncGnd' if USE_FUNCDESC else 'IntentGnd'}_{time_str}.json")

print(f"Finished evaluating {args.grounder} at {time_str}. Save eval results to {save_file}")
with open(save_file, "w") as f:
    meta = vars(args)
    meta['max_prev_actions'] = MAX_PREV_ACT
    
    json.dump(
        {
            "meta": meta,
            "overall_results": aggr_metrics,
            "metrics_each_repeat": metrics_all_repeats,
            "logs": results_all_repeats,
        },
        f,
        indent=2
    )