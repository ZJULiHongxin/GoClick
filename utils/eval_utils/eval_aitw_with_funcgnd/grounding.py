# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score

import os, time, cv2
import random
import torch
import json
from tqdm import tqdm

import datasets
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
from utils.eval_utils.action_matching import *
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import pred_2_point, lower_first_letter

try:
    from utils.openai_utils.qwen2vl import QWen2VL
except Exception as e:
    print(e)
    
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.cogagent import CogAgent
from utils.openai_utils.qwenvl import QwenVL
from utils.openai_utils.tinyclick import TinyClick
from utils.openai_utils.autogui_florence2 import AutoGUI_Florence

from copy import deepcopy

logging.basicConfig(level=logging.INFO)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--planning_result_file', type=str, default='utils/eval_utils/eval_aitw_with_funcgnd/eval_results/gemini-2.5-pro/2025-12-03-20-13-07.json')
parser.add_argument('--grounder', type=str, default=[
    'gpt-4o-mini', # provider: openai
    '/mnt/vdb1/hongxin_li/uipro_ckpt/0102_Qwen2vl-7B-490kIntentGnd/lora/checkpoint-3832', # provider: qwen2-vl
    'cckevinn/SeeClick', # provider: qwenvl
    'OS-Copilot/OS-Atlas-Base-7B', # provider: qwen2-vl
    'Samsung/TinyClick', # provider: tinyckick
    'THUDM/cogagent-chat-hf', # provider: cogagent
    '/mnt/vdb1/hongxin_li/uipro_ckpt/0223_IJCV-UIPro-Florence-Large+AgentTaskIntentGnd_4295k/checkpoint-33556', # provider: autogui_florence
    '/mnt/vdb1/hongxin_li/goclick_ckpts/0314_GoClick-Florence2Large_CoreSet-v2_3814k/checkpoint-29800/' # provider: autogui_florence
    ][-1]) # /mnt/nvme0n1p1/hongxin_li/highres_autogui/checkpoints/1019_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+AutoGUI1p3M-Android68k/ , /mnt/vdb1/hongxin.li/uipro_ckpt/1030_SliME-Gemma-1p1-2B_5MBoxQAs+SimpUI5p3M+UIPro18p6M/checkpoint-32656s

parser.add_argument('--provider', type=str, default=['openai', 'qwen2-vl', 'qwenvl', 'qwen2-vl', 'tinyckick', 'cogagent', 'autogui_florence'][-1])
parser.add_argument('--imgs_dir', type=str, default='/mnt/vdb1/hongxin_li/AITW/aitw_images/')
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()


aitw_imgs_dir = args.imgs_dir

local_data_file = "utils/eval_utils/AITW_test/data/test-00000-of-00001.parquet"
if os.path.exists(local_data_file):
    temp = pd.read_parquet(local_data_file)
    aitw_test = temp.to_dict(orient='records')
else:
    aitw_test = datasets.load_dataset("HongxinLi/AITW_test", split='test')

# aitw_test = json.load(open(os.path.join(os.path.dirname(aitw_imgs_dir),'aitw_data_test.json'), 'r'))
aitw_test_each_app = {}
for x in aitw_test:
    app = x['image'].split('/')[0]
    aitw_test_each_app.setdefault(app, []).append(x)

score_average = 0
time_record = [0,0]
tasks_result = {}
tasks_logs = {}


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

print(f"Loading eval file: {args.planning_result_file} | Gnd by {postfix}")
planning_eval_results_info = json.load(open(args.planning_result_file))
meta = planning_eval_results_info["meta"]
planning_eval_results = planning_eval_results_info["logs"]

for task, steps in aitw_test_each_app.items():
    tasks_logs[task] = []
    print("Task: " + task)

    corr_action = 0
    corr_type = 0
    num_text = 0
    corr_text = 0
    num_scroll = 0
    corr_scroll = 0
    num_click = 0
    corr_click = 0
    num_both_click = 0
    corr_both_click = 0
    num_wrong_format = 0
    num = 0

    planning_result_this_task = {}
    for x in planning_eval_results[task]:
        planning_result_this_task[x['step_id']] = x

    for step_i, step in tqdm(enumerate(steps), total=len(steps), desc=f"{task} | {postfix} | use_funcgnd: {USE_FUNCDESC}"):
        if args.debug and step_i >=2: break# % 100 >= 1: continue
        img_filename = step["image"]
        img_path = os.path.join(aitw_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not found")
            continue

        goal = step["step"]["goal"]

        action_ref = action_2_format(step["step"])
        step_id = f"{step['step']['ep_id']}-{step['step']['step']}"
        temp = planning_result_this_task[step_id]

        # Format step history
        raw_history = step['history']
        if f'Step {MAX_PREV_ACT+1}.' in raw_history:
            steps = []      
            this_step_idx = 0
            while True:
                next_step_idx = raw_history.find('Step ', this_step_idx+1)

                end = False
                if next_step_idx == -1:
                    end = True
                    next_step_idx = raw_history.find('\n', this_step_idx+1)

                this_step = raw_history[raw_history.find('. ', this_step_idx)+2:next_step_idx].strip(' .')
                steps.append(this_step)

                if end: break
                this_step_idx = next_step_idx

            history = ' '.join(f'Step {i}. {step}.' for i, step in enumerate(steps[-MAX_PREV_ACT:], start=1))
        else:
            history = step['history']

        prompt_lst = []
        response_lst = []
        action_pred_lst = []

        t1 = time.time()

        # unload the grounding task to the grounder
        raw_prompt, raw_response, new_action, response = '', None, None, None

        try:
            funcdesc = planning_result_this_task[step_id].get('funcdesc','').strip(' .') # if the original response's format is wrong, this will also be recorded as a wrong format error.
            intent = planning_result_this_task[step_id].get('summary','').replace('I ','').strip(' .')

            raw_prompt, raw_response, raw_status = planning_result_this_task[step_id]['prompt'], planning_result_this_task[step_id]['response'], planning_result_this_task[step_id]['status']

            new_action = deepcopy(planning_result_this_task[step_id]['action_pred'])
            if isinstance(new_action, str): new_action = eval(new_action)

            if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]) and new_action['action_type'] == 'click' and not (USE_FUNCDESC and 'None' in funcdesc):
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

        if raw_response and step_i % 5 == 0:
            key_content = raw_prompt[raw_prompt.rfind("The user's task is:"):raw_prompt.rfind("\nYour output should")]
            print(Fore.CYAN + f"User: {key_content}\n" + Fore.YELLOW + f"Original Response by {meta['planner']}: {raw_response}\n" + f"Gnd by {postfix}: {response}\n" + Style.RESET_ALL)

        time_record[0] += time.time() - t1; time_record[1] += 1
        num += 1
        
        try:
            action_pred = pred_2_format(new_action, scale=1)
            
            annot_position = np.array(
                [step["step"]["annot_position"][i:i + 4] for i in range(0, len(step["step"]["annot_position"]), 4)]) # [y, x, h, w, ...]
            
            if False:
                img = cv2.imread(img_path)
                H,W =img.shape[:2]
                for anno in annot_position:
                    y,x,h,w = anno
                    x1,y1,x2,y2 = x*W,y*H,(x+w)*W,(y+h)*H
                    x1,y1,x2,y2 = list(map(round, [x1,y1,x2,y2]))
                    cv2.rectangle(img, (x1,y1),(x2,y2), color=(0,0,255),thickness=2)
                cv2.imwrite("test.png", img)

            # If the action requires a target, we conduct auxhiliary checking to avoid False Negative by:
            # 1) chekcing if the predicted target falls in the same textbox if the interaction target is a textbox.
            # 2) chekcing if the predicted target falls in the same link region if the interaction target is a website in a searching result list.
            refexp = funcdesc + intent
            correct_if_in_the_same_row = any([k in refexp for k in ['search', 'website']]) and all([k not in refexp for k in ['initiate']])

            check_match = check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                action_pred["action_type"], action_ref["touch_point"],
                                                                action_ref["lift_point"], action_ref["action_type"],
                                                                annot_position, correct_if_in_the_same_row=correct_if_in_the_same_row)
            

            # step accuracy
            if check_match == True:
                corr_action += 1
                match_label = 1
                #logging.info("Step: " + str(j) + " right")
                temp["status"] = "correct"
            else:
                match_label = 0
                #logging.info("Step: " + str(j) + " wrong")
                temp["status"] = "wrong"
            # type accuracy
            if action_pred["action_type"] == action_ref["action_type"]:
                corr_type += 1
                temp["status"] += ",action_correct"
            else: temp["status"] += ",action_wrong"
            
            # text accuracy
            if action_ref["action_type"] == 3:
                num_text += 1
                if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                        action_pred["typed_text"] in action_ref["typed_text"]) or (
                        action_ref["typed_text"] in action_pred["typed_text"]):
                    corr_text += 1
                    temp["status"] += ",typedText_correct"
                else: temp["status"] += ",typedText_wrong"

            if action_ref["action_type"] == 4:
                # click accuracy
                if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                    num_click += 1
                    if match_label:
                        corr_click += 1
                # scroll accuracy
                else:
                    num_scroll += 1
                    if match_label:
                        corr_scroll += 1
                if (action_pred["action_type"] == 4) and is_tap_action(action_ref["touch_point"], action_ref["lift_point"]) and is_tap_action(
                        action_pred["touch_point"], action_pred["lift_point"]):
                    num_both_click += 1
                    if match_label:
                        corr_both_click += 1

            temp["grounder_action_pred"] = new_action
        except:
            num_wrong_format += 1
            # logging.info("Step: " + str(j) + " wrong format")
            temp["new_status"] = "wrong format"

        tasks_logs[task].append(temp)

    logging.info("Avg Time: " + str(time_record[0] / time_record[1]))

    action_acc = f"{100 * corr_action / num if num else 0:.2f}% / {corr_action} / {num}"
    action_type_acc = f"{100 * corr_type / num if num else 0:.2f}% / {corr_type} / {num}"
    
    text_acc = f"{100 * corr_text / num_text if num_text else 0:.2f}% / {corr_text} / {num_text}"
    click_acc = f"{100 * corr_click / num_click if num_click else 0:.2f}% / {corr_click} / {num_click}"
    scroll_acc = f"{100 * corr_scroll / num_scroll if num_scroll else 0:.2f}% / {corr_scroll} / {num_scroll}"
    dual_click_acc = f"{100 * corr_both_click / num_both_click if num_both_click else 0:.2f}% / {corr_both_click} / {num_both_click}"
    
    tasks_result[task] = {"action_acc": action_acc, "action_type_acc": action_type_acc, "text_acc": text_acc, "click_acc": click_acc, "scroll_acc": scroll_acc, "dual_click_acc": dual_click_acc, "num_wrong_format": num_wrong_format}
    
    score_average += corr_action / num if num else 0

    print(f"Action Acc: {action_acc}")
    print(f"Type Acc: {action_type_acc}")
    print(f"Text Acc: {text_acc}")
    print(f"Click Acc: {click_acc}")
    print(f"Scroll Acc: {scroll_acc}")
    print(f"dual_click_acc: {dual_click_acc}")
    print(f"Num wrong format: {num_wrong_format}")

tasks_result['avg'] = score_average / len(tasks_result)
logging.info("Average score: " + str(score_average / len(tasks_result)))

save_to = args.planning_result_file.replace(".json", f"_RevisedBy_{postfix}_{'FuncGnd' if USE_FUNCDESC else 'IntentGnd' }.json")

time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

print(f"Finished evaluating {args.grounder} at {time_str}. Save eval results to {save_to}")

meta = vars(args)
meta['use_funcdesc'] = USE_FUNCDESC
with open(save_to, "w") as f:
    json.dump({"meta": meta, "eval_result": tasks_result, "time": time_str, "logs": tasks_logs}, f, indent =2)