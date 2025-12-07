import os, time, cv2
import random
import json
from copy import deepcopy
from tqdm import tqdm
import re
import logging
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from utils.data_utils.task_prompt_lib import *
from pprint import pprint
from utils.data_utils.misc import average_iou
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.misc import keep_unique_actions, contains_chinese
from utils.openai_utils.misc import extract_thought_components, extract_thought_components_llamav

from colorama import Fore, Style

def clean_answer(text):
    text = text.lower().strip(' .?!').replace('"', '').replace("'", '').replace(',', '').replace(";", '')
    
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--guicourse_dir', type=str, default='/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse')
    parser.add_argument('--planner', type=str, default=[
        'gpt-4o', # provider: openai
        'gemini-2.0-flash-exp', # provider: openai
        'Qwen/Qwen2.5-VL-7B-Instruct', # provider: qwen2-vl
        'meta-llama/Llama-3.2-11B-Vision-Instruct', # provider: llama3
        ][0])
    parser.add_argument('--provider', type=str, default=['openai', 'qwen2-vl', 'llama3'][0])
    parser.add_argument('--device_type', type=str, default='Web', choices=['Web', 'Mobile'])
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--max_prev_acts', type=int, default=6)
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

    ROOT = "/mnt/vdb1/hongxin_li"
    guiact_imgs_dir = os.path.join(args.guicourse_dir, "GUIAct")

    eval_result_dir = os.path.join(os.path.dirname(__file__), f'eval_results/GUIAct-{args.device_type}')
    os.makedirs(eval_result_dir, exist_ok=True)

    save_to = os.path.join(eval_result_dir, postfix)

    test_file = os.path.join(args.guicourse_dir, f"{args.device_type}_test.json")
    
    guiact_test = json.load(open(test_file, 'r'))

    results = {'en': [], 'zh': []}
    metrics_each_lang = {}
    # "click": 1461,
    # "scroll": 560,
    # "status": 197,
    # "hover": 39,
    # "drag": 194,
    # "hotkey": 14

    # Web Split: {'drag', 'scroll', 'hotkey', 'hover', 'click', 'status'}

    # press_key can be used to replace enter
    actions = ['total', 'click', 'scroll', 'hover','drag', 'press_key', 'hotkey', 'status', 'swipe', 'tap', 'input_text', 'enter']
    counts = {k:{act: 0 for act in actions} for k in ['en', 'zh']}

    if args.debug:
        guiact_test = random.sample(guiact_test, 6)

    for step_idx, step in tqdm(enumerate(guiact_test), total=len(guiact_test), desc=f"{postfix} on GUIAct-{args.device_type}"):
        goal = step["task"]
        
        lang = 'zh' if contains_chinese(goal) else 'en'

        counts[lang]['total'] += 1
        action_type_ref = step['action_type']
        #if action_type_ref not in ['status']: continue
        counts[lang][action_type_ref] += 1

        img_path = os.path.join(args.guicourse_dir, step["image"].split('GUICourse/')[-1])

        try:
            image = Image.open(img_path)
        except:
            continue
        W,H=image.size
        # Used in the prompt

        # history只有一个元素
        # prompt = SIMPLE_PROMPT.format(global_task=goal, history=step['history'][0], step_instruction='')

        raw_history = step['history']
        if isinstance(raw_history, list):
            _, clean_step_instructions = keep_unique_actions(raw_history)
            history = clean_step_instructions[max(0,len(clean_step_instructions)-MAX_PREV_ACT):]
            
            history_str = ' '.join(f"Step {i}. {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,len(clean_step_instructions)-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'
        else:
            history_str = raw_history

        t1 = time.time()

        raw_prompt = TWOSTAGE_GUIACTMOBILE_PLANNING_PROMPT_COT if args.device_type == 'Mobile' else TWOSTAGE_GUIACTWEB_PLANNING_PROMPT_COT
        prompt = raw_prompt.format(global_task=goal, history=history_str, step_instruction='', xscale=SCALE, yscale=SCALE)

        metrics = {f'{act}_match': False for act in actions}
        metrics['action_match'] = metrics['type_match'] = metrics['elem_acc'] = metrics['need_gnd'] = False

        task_attr = step['task_attr']

        sample_id = step['id'].split('_')[-1]
        step_result = {
                "step_id": sample_id,
                "img_path": img_path,
                "task": goal,
                "step_instruciton": step['step_instruction'],
                "gt_action": step['step_info'],
                "prompt": None, "response": None, "original_action": task_attr,
                "action_pred": None,
                "metrics":  deepcopy(metrics)
            }
            
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

                step_result['prompt'], step_result['response'], step_result['observation'], step_result['thought'], step_result['funcdesc'], step_result['summary'], step_result['action_pred_raw'] = prompt, response, obs, thought, funcdesc, summary, action_pred_raw
                break
            except Exception as e:
                print(e)
                retry += 1
                temperature = 0.6

        try:
            action_pred = eval(action_pred_raw)
       
            step_result["action_pred"] = action_pred

            # matching
            action_type_pred = action_pred['action_type']

            if action_type_ref == action_type_pred:
                step_result['metrics']['type_match'] = True

                if action_type_ref in ['click', 'hover']:
                    step_result['metrics']['need_gnd'] = True
                    target_pred = list(map(lambda p: p / SCALE, action_pred['target']))
                    
                    if 'bbox' in task_attr:
                        gt_box_normalized = task_attr['bbox']
                    
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True
                    elif 'center' in task_attr:
                        center_normalized = task_attr['center']

                        if np.linalg.norm(np.array(center_normalized)-np.array(target_pred)) < 0.14:
                            step_result['metrics']['action_match'] = step_result['metrics']['elem_acc'] = step_result['metrics'][f'{action_type_ref}_match'] = True

                elif action_type_ref == 'input_text':                    
                    if squad_metrics.compute_f1(task_attr['text'], action_pred['text']) > 0.5 :
                        step_result['metrics']['action_match'] = step_result['metrics']['input_text_match'] = True
                
                elif action_type_ref == 'scroll':
                    step_result['metrics']['action_match'] = step_result['metrics']['scroll_match'] = task_attr['direction'] == action_pred['direction']
                
                elif action_type_ref == 'status':
                    status_ref, status_pred = task_attr['goal_status'], action_pred['goal_status']
                    
                    if status_ref == status_pred:
                        answer_ref, answer_pred = clean_answer(task_attr['answer']), clean_answer(action_pred.get('answer', ''))
                        if len(answer_pred) == 0 and len(answer_pred) == 0:
                            answer_f1 = 1.0
                        else:
                            answer_f1 = squad_metrics.compute_f1(answer_ref, answer_pred)
                        
                        step_result['metrics']['action_match'] = step_result['metrics']['status_match'] = answer_f1 > 0.5
                
                elif action_type_ref == 'drag':
                    drag_start, drag_end = list(map(lambda p: p/SCALE, action_pred['start'])), list(map(lambda p: p/SCALE, action_pred['end']))

                    # post-process the drage points to handle the case where the drag points do not form a rectangle
                    if drag_start[0] == drag_end[0]:
                        drag_start[0] -= 0.01
                        drag_end[0] += 0.01
                    elif drag_start[1] == drag_end[1]:
                        drag_start[1] -= 0.01
                        drag_end[1] += 0.01

                    gt_box = min(task_attr['from'][0], task_attr['to'][0]), min(task_attr['from'][1], task_attr['to'][1]), max(task_attr['from'][0], task_attr['to'][0]), max(task_attr['from'][1], task_attr['to'][1]) # the order of dragging start and end is not fixed
                    
                    iou = average_iou(np.array([gt_box, [drag_start[0], drag_start[1], drag_end[0], drag_end[1]]])).item()
                    
                    step_result['metrics']['action_match'] = step_result['metrics']['drag_match'] = iou > 0.5
                
                elif action_type_ref == 'hotkey':
                    keycomb_ref, keycomb_pred = task_attr['key_comb'].lower(), action_pred['key_comb'].lower()
                    
                    step_result['metrics']['action_match'] = step_result['metrics']['hotkey_match'] = keycomb_ref == keycomb_pred
                else:
                    step_result['metrics']['action_match'] = step_result['metrics'][f'{action_type_ref}_match'] = True
            
            if step_idx % 3 == 0:
                p = prompt if step_idx <= 2 else prompt[prompt.rfind("The user's task"): prompt.rfind("Your output should")]
                print(f"{Fore.CYAN}{p}{Style.RESET_ALL} => {Fore.GREEN}{step_result['response']}{Style.RESET_ALL}")

            print(f"{step_idx}: {step_result['metrics']['action_match']}: {step_result['gt_action']} <=> {action_pred}")

        except Exception as e:
            print(e)
            logging.info("format wrong")
            step_result['wrong_format'] = True


        results[lang].append(step_result)

    # calculate metrics
    for lang in results.keys():
        num_sample = counts[lang]['total']
        num_need_gnd = sum(x['metrics']['need_gnd'] for x in results[lang])
        num_action_match = sum(x['metrics']['action_match'] for x in results[lang])
        num_type_match = sum(x['metrics']['type_match'] for x in results[lang])
        num_elem_match = sum(x['metrics']['elem_acc'] for x in results[lang])
        final_metrics = {'step_acc': [(num_action_match / num_sample) if num_sample > 0 else 0., num_action_match, num_sample], 'action_type_acc': [(num_type_match / num_sample) if num_sample > 0 else 0., num_type_match, num_sample], 'elem_acc': [(num_elem_match / num_need_gnd) if num_need_gnd > 0 else 0., num_elem_match, num_need_gnd]}

        for k in counts[lang].keys():
            if k=='total': continue
            
            cnt = counts[lang][k]
            acc_cnt = sum(x['metrics'][f'{k}_match'] for x in results[lang])
            
            final_metrics[f'{k}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]

        final_metrics['num_wrong_format'] = sum(1 for x in results[lang] if 'wrong_format' in x)

        pprint(lang + ': ' + str(final_metrics))
        
        metrics_each_lang[lang] = final_metrics

# aggr metrics
aggr_metrics = {}
for lang, metrics_subset in metrics_each_lang.items():
    for metric_name, info in metrics_subset.items():
        if metric_name == 'num_wrong_format':
            aggr_metrics['num_wrong_format'] = aggr_metrics.get('num_wrong_format',0) + metrics_subset['num_wrong_format']
            continue

        if metric_name not in aggr_metrics: aggr_metrics[metric_name] = [0,0,0]
        
        aggr_metrics[metric_name][1] += metrics_subset[metric_name][1]
        aggr_metrics[metric_name][2] += metrics_subset[metric_name][2]

for metric_name in aggr_metrics.keys():
    if metric_name == 'num_wrong_format': continue
    acc_cnt, cnt = aggr_metrics[metric_name][1], aggr_metrics[metric_name][2]
    aggr_metrics[metric_name][0] = acc_cnt / cnt if cnt > 0 else 0
    
print("\nFinal:")
pprint(aggr_metrics)

os.makedirs(save_to, exist_ok=True)
save_file = os.path.join(save_to, args.device_type + '-' + datetime.now().strftime("%m-%d-%H-%M-%S")) + '.json'
with open(save_file, "w") as f:
    json.dump(
        {
            "meta": vars(args),
            "overall_results": aggr_metrics,
            "metrics_each_lang": metrics_each_lang,
            "logs": results,
        },
        f,
        indent=2
    )

print(f"Finised evaluation {postfix} on GUIAct-{args.device_type}. Save to {save_file}")
    

