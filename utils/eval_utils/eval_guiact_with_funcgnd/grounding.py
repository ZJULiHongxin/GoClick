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
from utils.data_utils.misc import average_iou, lower_first_letter, pred_2_point
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.cogagent import CogAgent
from utils.openai_utils.qwenvl import QwenVL
from utils.openai_utils.tinyclick import TinyClick
from utils.openai_utils.autogui_florence2 import AutoGUI_Florence
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.misc import keep_unique_actions, contains_chinese

from colorama import Fore, Style

def clean_answer(text):
    text = text.lower().strip(' .?!').replace('"', '').replace("'", '').replace(',', '').replace(";", '')
    
    return text


parser = argparse.ArgumentParser()
parser.add_argument('--guicourse_dir', type=str, default='/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse')
parser.add_argument('--planning_result_file', type=str, default='utils/eval_utils/eval_guiact_with_funcgnd/eval_results/GUIAct-Web/gpt-4o/Web-12-05-23-00-54.json')
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
        USE_FUNCDESC = True #'intent' not in model_path.lower()
        if USE_FUNCDESC:
            PROMPT = FUNCGND_PROMPT
        else:
            PROMPT = intentgnd_prompt[0]
        
        PROMPT += ' (Output the center coordinates of the target)'
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

# Load the evaluation results
print(f"Loading eval file: {args.planning_result_file} | Gnd by {postfix}")
planning_eval_results_info = json.load(open(args.planning_result_file))
meta = planning_eval_results_info["meta"]
planning_eval_results = planning_eval_results_info["logs"]

planning_results_dict = {}
for leng, results in planning_eval_results.items():
    for x in results:
        unique_id = f'{os.path.basename(x["img_path"])}|{x["task"]}'
        planning_results_dict[unique_id] = x

ROOT = "/mnt/vdb1/hongxin_li"
guiact_imgs_dir = os.path.join(args.guicourse_dir, "GUIAct")

DEVICE_TYPE = meta["device_type"]
eval_result_dir = os.path.join(os.path.dirname(__file__), f'eval_results/GUIAct-{DEVICE_TYPE}')
os.makedirs(eval_result_dir, exist_ok=True)

save_to = os.path.join(eval_result_dir, postfix)

test_file = {'Web': '/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/guiact-web-test_wActRef_s1000_2346.json',
                'Mobile': '/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/GUICourse_processed/guiact-smartphone-test_wActRef_s1000_2070.json'}[DEVICE_TYPE]

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
    guiact_test = random.sample(guiact_test, 8)

for step_idx, step in tqdm(enumerate(guiact_test), total=len(guiact_test), desc=f"{postfix} on {DEVICE_TYPE}"):
    goal, unique_id = step["task"], f"{os.path.basename(step['image'])}|{step['task']}"
    if unique_id not in planning_results_dict: continue

    lang = 'zh' if contains_chinese(goal) else 'en'

    counts[lang]['total'] += 1
    action_type_ref = step['action_type']
    # if action_type_ref not in ['drag']: continue
    counts[lang][action_type_ref] += 1

    img_path = os.path.join(args.guicourse_dir, step["image"].split('GUICourse/')[-1])

    try:
        image = Image.open(img_path)
    except: continue
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

    metrics = {f'{act}_match': False for act in actions}
    metrics['action_match'] = metrics['type_match'] = metrics['elem_acc'] = metrics['need_gnd'] = False

    task_attr = step['task_attr']
    step_result = deepcopy(planning_results_dict[unique_id])
    
    step_result |= {"new_metrics":  deepcopy(metrics) }

    t1 = time.time()

    # unload the grounding task to the grounder
    raw_prompt, raw_response, new_action, response = '', None, None, None
    
    if not step_result.get('wrong_format', False):
        try:
            funcdesc = step_result.get('funcdesc','').strip(' .') # if the original response's format is wrong, this will also be recorded as a wrong format error.
            intent = step_result.get('summary','').replace('I ','').strip(' .')
            
            raw_prompt, raw_response = step_result['prompt'], step_result['response']

            new_action = deepcopy(step_result['action_pred'])

            if isinstance(new_action, str): new_action = eval(new_action)

            if step_result['action_pred']['action_type'] == 'click' and step_result['gt_action']['action_type'] == 'click':
                prompt = PROMPT.format(funcdesc if USE_FUNCDESC else lower_first_letter(intent))
                if args.provider == 'openai':
                    _, response = openai.get_model_response(prompt, [img_path], use_img_url=True)
                else:
                    response = vlm.get_model_response(prompt, f"file://{img_path}")
        
                real_target = pred_2_point(response, keep_box=False, scale=SCALE)

                new_action['target'] = real_target
            else:
                'Not click'
        except Exception as e:
            print(e)


        try:
            action_pred = new_action
        
            step_result["grounder_action_pred"] = action_pred

            # matching
            action_type_pred = action_pred['action_type']

            if action_type_ref == action_type_pred:
                step_result['new_metrics']['type_match'] = True

                if action_type_ref in ['click', 'hover']:
                    step_result['new_metrics']['need_gnd'] = True
                    target_pred = action_pred['target']
                    
                    if 'bbox' in task_attr:
                        gt_box_normalized = task_attr['bbox']
                    
                        if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                            step_result['new_metrics']['action_match'] = step_result['new_metrics']['elem_acc'] = step_result['new_metrics'][f'{action_type_ref}_match'] = True
                    elif 'center' in task_attr:
                        center_normalized = task_attr['center']

                        if np.linalg.norm(np.array(center_normalized)-np.array(target_pred)) < 0.14:
                            step_result['new_metrics']['action_match'] = step_result['new_metrics']['elem_acc'] = step_result['new_metrics'][f'{action_type_ref}_match'] = True

                elif action_type_ref == 'input_text':                    
                    if squad_metrics.compute_f1(task_attr['text'], action_pred['text']) > 0.5 :
                        step_result['new_metrics']['action_match'] = step_result['new_metrics']['input_text_match'] = True
                
                elif action_type_ref == 'scroll':
                    step_result['new_metrics']['action_match'] = step_result['new_metrics']['scroll_match'] = task_attr['direction'] == action_pred['direction']
                
                elif action_type_ref == 'status':
                    status_ref, status_pred = task_attr['goal_status'], action_pred['goal_status']
                    
                    if status_ref == status_pred:
                        answer_ref, answer_pred = clean_answer(task_attr['answer']), clean_answer(action_pred.get('answer', ''))
                        if len(answer_pred) == 0 and len(answer_pred) == 0:
                            answer_f1 = 1.0
                        else:
                            answer_f1 = squad_metrics.compute_f1(answer_ref, answer_pred)
                        
                        step_result['new_metrics']['action_match'] = step_result['new_metrics']['status_match'] = answer_f1 > 0.5
                
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
                    
                    step_result['new_metrics']['action_match'] = step_result['new_metrics']['drag_match'] = iou > 0.5
                
                elif action_type_ref == 'hotkey':
                    keycomb_ref, keycomb_pred = task_attr['key_comb'], action_pred['key_comb']
                    
                    step_result['new_metrics']['action_match'] = step_result['new_metrics']['hotkey_match'] = keycomb_ref == keycomb_pred
                else:
                    step_result['new_metrics']['action_match'] = step_result['new_metrics'][f'{action_type_ref}_match'] = True
            
            if step_idx % 3 == 0:
                p = prompt if step_idx <= 2 else prompt[prompt.rfind("The user's task"): prompt.rfind("Your output should")]
                print(f"{Fore.CYAN}{p}{Style.RESET_ALL} => {Fore.GREEN}{step_result['response']}{Style.RESET_ALL}")

            print(f"{step_idx}: Old: {step_result['metrics']['action_match']} => New: {step_result['new_metrics']['action_match']}: {step_result['gt_action']} <=> {action_pred}")
        except Exception as e:
            print(e)
            logging.info("format wrong")
            step_result['wrong_format'] = True
    else:
        print('wrong format')

    results[lang].append(step_result)

# calculate metrics
for lang in results.keys():
    num_sample = counts[lang]['total']
    num_need_gnd = sum(x['new_metrics']['need_gnd'] for x in results[lang])
    num_action_match = sum(x['new_metrics']['action_match'] for x in results[lang])
    num_type_match = sum(x['new_metrics']['type_match'] for x in results[lang])
    num_elem_match = sum(x['new_metrics']['elem_acc'] for x in results[lang])
    final_metrics = {'step_acc': [(num_action_match / num_sample) if num_sample > 0 else 0., num_action_match, num_sample], 'action_type_acc': [(num_type_match / num_sample) if num_sample > 0 else 0., num_type_match, num_sample], 'elem_acc': [(num_elem_match / num_need_gnd) if num_need_gnd > 0 else 0., num_elem_match, num_need_gnd]}

    for k in counts[lang].keys():
        if k=='total': continue
        
        cnt = counts[lang][k]
        acc_cnt = sum(x['new_metrics'][f'{k}_match'] for x in results[lang])
        
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
save_to = args.planning_result_file.replace(".json", f"_RevisedBy_{postfix}_{'FuncGnd' if USE_FUNCDESC else 'IntentGnd' }.json")


with open(save_to, "w") as f:
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

print(f"Finised evaluation {postfix} on GUIAct-{DEVICE_TYPE}. Save to {save_to}")


