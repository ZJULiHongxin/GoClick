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
from datetime import datetime
from colorama import Fore, Style
from utils.eval_utils.action_matching import *
from utils.data_utils.task_prompt_lib import *
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.llama3v import Llama3V
from utils.openai_utils.misc import extract_thought_components, extract_thought_components_UITARS
from utils.data_utils.misc import make_ui_tars_messages,remove_redundant_spaces, keep_unique_actions, parse_UITARS_action


logging.basicConfig(level=logging.INFO)


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--planner', type=str, default=['gpt-4o-2024-11-20', 'gemini-2.0-flash-exp', 'gemini-2.5-pro', 'Qwen/Qwen2.5-VL-7B-Instruct', 'meta-llama/Llama-3.2-11B-Vision-Instruct', 'bytedance-research/UI-TARS-72B-DPO'][2])

parser.add_argument('--provider', type=str, default=['openai', 'qwen2-vl', 'llama3'][0])
parser.add_argument('--imgs_dir', type=str, default='/mnt/nvme0n1p1/hongxin_li/UI_training_data/AITW/aitw_images/')
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

aitw_imgs_dir = args.imgs_dir

aitw_test = datasets.load_dataset("HongxinLi/AITW_test", split='test')

aitw_test_each_app = {}
for x in aitw_test:
    app = x['image'].split('/')[0]
    aitw_test_each_app.setdefault(app, []).append(x)

score_average = 0
time_record = [0,0]
tasks_result = {}
tasks_logs = {}

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

eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results', postfix)
os.makedirs(eval_result_dir, exist_ok=True)

time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


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

    for step_i, step in tqdm(enumerate(steps), total=len(steps), desc=f"{postfix} | {args.planner} | {task}"):
        if args.debug and step_i >=2: break# % 100 >= 1: continue
        img_filename = step["image"]
        step_idx = int(img_filename.split('_')[-1][:-4])
        img_path = os.path.join(aitw_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not found")
            continue

        goal = step["step"]["goal"]

        action_ref = action_2_format(step["step"])
        temp = {
            "step_id": f"{step['step']['ep_id']}-{step['step']['step']}",
            "step_info": step,
            "split": task,
            "action_ref": action_ref,
            "prompt": None,
            "response": None,
            }

        # Format step history
        raw_history = step['history']
        if isinstance(raw_history, list):
            _, clean_step_instructions = keep_unique_actions(raw_history)
            history = clean_step_instructions[max(0,len(clean_step_instructions)-MAX_PREV_ACT):]

            END_PUNC, LINE_SPLIT = ':' if 'atlas' in postfix.lower() else '.', '\n' if 'atlas' in postfix.lower() else ' '
            history_str = LINE_SPLIT.join(f"Step {i}{END_PUNC} {instruc.strip(' .')}." for i, instruc in enumerate(history, start=max(1,step_idx-MAX_PREV_ACT+1))) if len(history) > 0 else 'None'
        elif f'Step {args.max_prev_acts+1}.' in raw_history:
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
            
            history_str = ' '.join(f'Step {i}. {step}.' for i, step in enumerate(steps[-args.max_prev_acts:], start=max(1,len(steps)-args.max_prev_acts+1)))
        # elif 'atlas' in postfix.lower():
        #     history_str = 'None' if step_i == 0 else '\n'.join(f'Step {i}: {step.strip(" .")}.' for i, step in enumerate(steps[:step_i], start=1))
        else:
            history_str = step['history']

        prompt_lst = []
        response_lst = []
        action_pred_lst = []
        
        t1 = time.time()
        num += 1

        prompt = AITW_PLANNING_PROMPT_FUNCDESC.format(global_task=goal, history=history_str, xscale=SCALE, yscale=SCALE)

        retry = 0
        temperature = 0.0
        while retry < 3:
            try:    # several sample's img dir lead to error, just jump it
                if args.provider == 'openai':
                    _, response, _ = openai.get_model_response(prompt, [img_path], use_img_url=True, temperature=temperature)
                else:
                    response = vlm.get_model_response(prompt, f"file://{img_path}", temperature=temperature)

                temp["prompt"] = prompt
                temp["response"] = response

                if args.provider == 'llama3' and '**' in response:
                    obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components_llamav(response)
                else:
                    obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components(response)

                assert len(obs) > 0 and len(thought) > 0 and len(funcdesc) > 0 and len(action_pred_raw) > 0 and len(summary) > 0

                temp["action_pred"], temp["observation"], temp["thought"], temp["funcdesc"], temp["summary"] = action_pred_raw, obs, thought, funcdesc, summary

                break
            except Exception as e:
                print(e)
                retry += 1
                temperature = 0.6
        
        if step_i % 1 == 0:
            key_content = prompt[prompt.rfind("The user's task is:"):prompt.rfind("\nYour output should")]
            print(Fore.CYAN + f"User: {key_content}\n" + Fore.YELLOW + f"GPT: {response}\n" + Style.RESET_ALL)
        time_record[0] += time.time() - t1; time_record[1] += 1
        
        try:
            # parse actions
            # observation = re.findall(r"Observation:\s*(.*?)$", response, re.MULTILINE | re.DOTALL)[0].strip()
            # think = re.findall(r"Thought:\s*(.*?)$", response, re.MULTILINE | re.DOTALL)[0].strip()
            # funcdesc = re.findall(r"Target's Functionality:\s*(.*?)$", response, re.MULTILINE | re.DOTALL)[0].strip(' *-_')
            action_pred_raw = eval(action_pred_raw)
            action_pred = pred_2_format(action_pred_raw, scale=SCALE)
            
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
            check_match = check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                action_pred["action_type"], action_ref["touch_point"],
                                                                action_ref["lift_point"], action_ref["action_type"],
                                                                annot_position)
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

        except:
            num_wrong_format += 1
            # logging.info("Step: " + str(j) + " wrong format")
            temp["status"] = "wrong format"

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

    save_to = os.path.join(eval_result_dir, time_str + '.json')

    print(f"Finished evaluating {args.planner} at {time_str} on {task}. Save eval results to {save_to}")
    with open(save_to, "w") as f:
        json.dump({"meta": vars(args), "eval_result": tasks_result, "time": time_str, "logs": tasks_logs}, f, indent = 2)