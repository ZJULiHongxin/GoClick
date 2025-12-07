# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score

import os, time, cv2
import random
import torch
import json
import traceback
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
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.task_prompt_lib import *
from utils.eval_utils.action_matching import *
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.llama3v import Llama3V
from utils.data_utils.misc import keep_unique_actions
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
parser.add_argument('--planner', type=str, default=[
    'gpt-4o', # provider: openai
    'gemini-2.0-flash-exp', # provider: openai
    'Qwen/Qwen2.5-VL-7B-Instruct', # provider: qwen2-vl
    'meta-llama/Llama-3.2-11B-Vision-Instruct', # provider: llama3
    ][0])
parser.add_argument('--provider', type=str, default=['openai', 'qwen2-vl', 'llama3'][0])

parser.add_argument('--debug', action="store_true")
parser.add_argument('--max_prev_acts', type=int, default=6)
parser.add_argument('--repeat', type=int, default=3)


args = parser.parse_args()

print(f"Loading model from {args.planner}")
if args.provider == 'openai':
    openai = OpenAIModel(model=args.planner,
                        base_url=os.environ.get("OPENAI_BASE_URL", "https://xiaoai.plus/v1/"),
                        api_key=os.environ.get("OPENAI_API_KEY", ""),
                        temperature=0.0)
    postfix = openai.model.replace("/","-")
    SCALE = 1000
else:
    if any(k in args.planner.lower() for k in ['qwen2-vl', 'qwen2.5-vl']):
        vlm = QWen2VL(device='cuda', model_name=args.planner)
        SCALE = 1000
    elif 'llama' in args.planner.lower():
        vlm = Llama3V(device='cuda', model_name=args.planner)
        SCALE = '1.00'
    postfix = vlm.model_name.replace("/","-")


androidcontrol_test_raw = json.load(open(f'{args.andcon_dir}/AndroidControl-test_12685.json', 'r'))


# 筛选出HL样本
hl_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' in x['id'])
h_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' not in x['id'])
hl_h_ids = hl_ids.intersection(h_ids)

time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

results_all_repeats = []
metrics_all_repeats = []
for repeat in range(args.repeat):
    selected_ids = random.sample(list(hl_h_ids), 500 if not args.debug else 7)
    
    hl_samples = [x for x in androidcontrol_test_raw if '-HL' in x['id'] and x['id'].split('-H')[0] in selected_ids]
    h_samples = [x for x in androidcontrol_test_raw if '-HL' not in x['id'] and x['id'].split('-H')[0] in selected_ids]
    androidcontrol_test = random.sample(androidcontrol_test_raw, 500)

    # HL
    metrics_this_repeat = {'HL': [], 'H': []}
    results = {'HL': [], 'H': []}

    # Only consider high-level instrucitons
    for mode, samples in zip(['H'], [h_samples]):
        counts = {'total': 0, 'click': 0, 'input_text': 0, 'swipe': 0, 'long_press': 0, 'enter': 0, 'navigate_home': 0, 'navigate_back': 0, 'status': 0, 'open_app': 0, 'wait': 0}

        for step_idx, step in tqdm(enumerate(samples), total=len(samples), desc=f'Repeat {repeat+1} {mode}'):
            goal = step["task"]
            counts['total'] += 1
            action_type_ref = step['action_type']

            counts[action_type_ref] += 1

            img_path = os.path.join(args.andcon_dir.replace("AndroidControl_test","").rstrip('/'), step["image"])

            image = Image.open(img_path)
            W,H=image.size
            # Used in the prompt

            _, clean_step_instructions = keep_unique_actions(step['history'])
            history = clean_step_instructions[max(0,len(clean_step_instructions)-args.max_prev_acts):]
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,step['step_id']-args.max_prev_acts+1))) if len(history) > 0 else 'None'


            prompt = ANDROIDCONTROL_PLANNING_PROMPT_COT.format(global_task=goal, history=history_str, step_instruction='', xscale=SCALE, yscale=SCALE)

            t1 = time.time()
                
            step_result = step | {"img_path": img_path, "prompt": prompt, "response": None, "action_pred": None, "metrics":  {k: False for k in ['action_match', 'type_match', 'elem_acc', 'click_match', 'input_text_match', 'swipe_match', 'enter_match', 'status_match', 'navigate_home_match', 'navigate_back_match', 'open_app_match', 'wait_match', 'long_press_match', 'need_gnd']} }

            retry = 0
            temperature = 0.0
            while retry < 3:
                try:    # several sample's img dir lead to error, just jump it
                    if args.provider == 'openai':
                        _, response, _ = openai.get_model_response(prompt, [img_path], use_img_url=True, temperature=temperature)
                    else:
                        response = vlm.get_model_response(prompt, f"file://{img_path}", temperature=temperature)

                    if args.provider == 'llama3' and '**' in response:
                        obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components_llamav(response)
                    else:
                        obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components(response)
                    
                    assert len(obs) > 0 and len(thought) > 0 and len(funcdesc) > 0 and len(action_pred_raw) > 0 and len(summary) > 0
                    break
                except Exception as e:
                    print(e)
                    retry += 1
                    temperature = 0.6

            try:
                step_result["response"] = response

                action_dict_start = action_pred_raw.rfind('{"action_type')
                if action_dict_start == -1:
                    action_dict_start = action_pred_raw.rfind("{'action_type")
                action_pred = eval(action_pred_raw[action_dict_start:])
                
                step_result["action_pred"] = action_pred
                
                # matching
                action_type_pred = action_pred['action_type']

                # special hanlding for enter
                if action_type_ref == 'enter' and action_pred['action_type'] == 'press_key' and action_pred['key'].lower() == 'enter':
                    enter_match = True
                else:
                    enter_match = False

                if action_type_ref == action_type_pred or enter_match:
                    step_result['metrics']['type_match'] = True

                    if action_type_ref in ['click', 'long_press']:
                        step_result['metrics']['need_gnd'] = True
                        target_pred = list(map(lambda p: p / SCALE, action_pred['target']))
                        
                        gt_box = step['task_attr']['bbox']
                        gt_box_normalized = list(map(lambda p:round(p, 3), [gt_box[0]/W, gt_box[1]/H, gt_box[2]/W, gt_box[3]/ H]))
                        
                        step['task_attr']['bbox'] = gt_box_normalized
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True

                    elif action_type_ref == 'input_text':
                        text_ref, text_pred = step['task_attr']['text'].lower().strip(), action_pred['text'].lower().strip()
                        
                        step_result['metrics']['action_match'] = step_result['metrics']['input_text_match'] = squad_metrics.compute_f1(text_pred, text_ref) > 0.5
                    
                    elif action_type_ref == 'swipe':
                        direction_ref, direction_pred = step['task_attr']['direction'], action_pred['direction']
                        direction_ref = scroll2swipe(direction_ref)
                        if direction_ref == direction_pred:
                            step_result['metrics']['action_match'] = step_result['metrics']['swipe_match'] = True
                    
                    elif action_type_ref == 'status':
                        status_ref, status_pred = step['task_attr']['goal_status'], action_pred['goal_status']
                        
                        if status_ref == status_pred:
                            step_result['metrics']['action_match'] = step_result['metrics']['status_match'] = True
                    elif action_type_ref == 'open_app':
                        app_name_ref, app_name_pred = step['task_attr']['app_name'], action_pred['app_name']
                        
                        if app_name_ref == app_name_pred:
                            step_result['metrics']['action_match'] = step_result['metrics']['open_app_match'] = True
                    else:
                        step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = True
                logging.info(f"{step_result['metrics']['action_match']}: GT: {step['task_attr']} <=> Pred: {action_pred}")

                if step_idx % 2 == 0:
                    print(Fore.YELLOW + f"\nUser: <img>{img_path}</img> {prompt}\n" + Fore.CYAN + f"GPT: {response}" + Style.RESET_ALL)
                
                step_result["observation"], step_result["thought"], step_result["funcdesc"], step_result["summary"] = obs, thought, funcdesc, summary
            except Exception as e:
                print(traceback.format_exc())
                logging.info("format wrong")
                step_result['wrong_format'] = True

            results[mode].append(step_result)

        # calculate metrics
        num_sample = counts['total']
        num_need_gnd = sum(x['metrics']['need_gnd'] for x in results[mode])
        
        num_action_match = sum(x['metrics']['action_match'] for x in results[mode])
        num_type_match = sum(x['metrics']['type_match'] for x in results[mode])
        num_elem_match = sum(x['metrics']['elem_acc'] for x in results[mode])
        final_metrics = {'step_acc': [num_action_match / num_sample, num_action_match, num_sample], 'action_type_acc': [num_type_match / num_sample, num_type_match, num_sample], 'elem_acc': [num_elem_match / num_need_gnd if num_need_gnd else 0, num_elem_match, num_need_gnd]}

        for k in counts.keys():
            if k=='total': continue
            
            cnt = counts[k]
            acc_cnt = sum(x['metrics'][f'{k}_match'] for x in results[mode])
            
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
os.makedirs(eval_result_dir, exist_ok=True)

save_to = os.path.join(eval_result_dir, postfix)

os.makedirs(save_to, exist_ok=True)
save_file = os.path.join(save_to, time_str + '.json')
with open(save_file, "w") as f:
    meta = vars(args)
    
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

print(f"Finished evaluating {args.planner} at {time_str}. Save eval results to {save_file}")