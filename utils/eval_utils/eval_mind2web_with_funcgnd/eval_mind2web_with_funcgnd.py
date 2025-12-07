# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score
import os, traceback
import random
import torch
import json
from tqdm import tqdm
import ast
import argparse
from PIL import Image
import numpy as np
from datetime import datetime
from utils.eval_utils.eval_utils import mind2web_action2step
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import keep_unique_actions

import transformers.data.metrics.squad_metrics as squad_metrics
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.aguvis import AGUVIS
from utils.data_utils.misc import make_ui_tars_messages,remove_redundant_spaces, keep_unique_actions, parse_UITARS_action, JSON2UITARS_action, parse_AGUVIS_action
from utils.openai_utils.misc import extract_thought_components, extract_thought_components_UITARS, extract_thought_components_aguvis, extract_thought_components_llamav

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

# calculate action f1 following mind2web
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

# convert action to prediction format
def action2step(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            click_point = [f"{item:.2f}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:  # for scroll action, we assign an action_type_id for each scroll
            if step_data["action_type_text"] == 'scroll down':
                action_type_new = 0
            elif step_data["action_type_text"] == 'scroll up':
                action_type_new = 1
            elif step_data["action_type_text"] == 'scroll left':
                action_type_new = 8
            elif step_data["action_type_text"] == 'scroll right':
                action_type_new = 9
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = action_type
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = action_type
        action = "{{\"action_type\": {}}}".format(action_type_new)

    return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mind2web_dir', type=str, default='/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web')
    parser.add_argument('--planner', type=str, default=[
        'gpt-4o', # provider: openai
        'gemini-2.0-flash-exp', # provider: openai
        'Qwen/Qwen2.5-VL-7B-Instruct', # provider: qwen2-vl
        'meta-llama/Llama-3.2-11B-Vision-Instruct', # provider: llama3
        'bytedance-research/UI-TARS-7B-DPO', # vllm
        'xlangai/Aguvis-7B-720P'][0]) # vllm
    parser.add_argument('--provider', type=str, default=['openai', 'qwen2-vl', 'llama3'][0])
    parser.add_argument('--scale', type=int, default=1000)

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--max_prev_acts', type=int, default=9)
    parser.add_argument('--conv_mode', type=bool, default=False)
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
        elif 'aguvis' in args.planner.lower():
            vlm = AGUVIS(device='cuda', model_name=args.planner)
            SCALE = 1
        elif 'llama' in args.planner.lower():
            vlm = Llama3V(device='cuda', model_name=args.planner)
            SCALE = '1.00'
            
        postfix = vlm.model_name.replace("/","-")

    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results')
    save_to = os.path.join(eval_result_dir, postfix)
    os.makedirs(save_to, exist_ok=True)

    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_file = os.path.join(save_to, time_str + '.json')

    mind2web_imgs_dir = os.path.join(args.mind2web_dir, "mind2web_images")

    results_all_tasks = {'website': [], 'task': [], 'domain': []}
    metrics_all_tasks = {'website': [], 'task': [], 'domain': []}

    for task in ['website', 'task', 'domain']:
        mind2web_test = json.load(open(f'{args.mind2web_dir}/mind2web_data_test_' + task + '.json', 'r'))

        results = []
        for ep_idx, episode in tqdm(enumerate(mind2web_test), total=len(mind2web_test), desc=f"Evaluating {task}"):
            if args.debug and ep_idx > 1:
                break

            goal = episode["confirmed_task"]
            annot_id = episode["annotation_id"]
            previous_actions = []
            results_actions = []

            prev_actions, prev_obs, past_action_strs = [], [], []
            for action_repr in episode['action_reprs']:
                elem, act = action_repr.split('->')
                act = act.strip()
                elem = elem.replace('  ',' ').strip()
                if 'TYPE:' in act:
                    split_id = act.find(':')
                    act, text = act[:split_id], act[split_id+1:]
                    text = text.strip(' \n\\').replace('"', '\\"').replace('\n', '\\n')
                    prev_act_str = f"type \"{text}\" into the {elem}"
                elif act == 'ENTER':
                    prev_act_str = f"press enter on {elem}"
                elif act == 'CLICK':
                    prev_act_str = f"click on {elem}"
                elif act == 'HOVER':
                    prev_act_str = f"hover over {elem}"
                elif 'SELECT:' in act:
                    split_id = act.find(':')
                    act, value = act[:split_id], act[split_id+1:]
                    value = value.strip()
                    prev_act_str = f"select {value} in the {elem}"
                else:
                    raise Exception(f"unknown action: {act}")
                prev_actions.append(prev_act_str)
                    
            for step_i, step in enumerate(episode["actions"]):
                if args.debug and step_i > 2:
                    break

                if "bbox" not in step:
                    print("action not found")
                    continue

                filename = annot_id + '-' + step["action_uid"] + '.jpg'
                img_path = os.path.join(mind2web_imgs_dir, filename)
                if not os.path.exists(img_path):
                    print("img not found")
                    continue
                image = Image.open(img_path)

                prev_obs.append(img_path)
                past_action_strs.append(prev_actions[step_i])

                # Used in the prompt
                action_step, bbox_ref = mind2web_action2step(step, image.size, scale=args.scale, return_bbox=True)

                try:
                    action_step_ref = ast.literal_eval(action_step)
                except:
                    continue

                # if action_step_ref['action_type'] not in ['input_text']:
                #     continue
                # make history str
                _, clean_prev_step_instructions = keep_unique_actions(prev_actions[:step_i])
                retained_history = clean_prev_step_instructions[-MAX_PREV_ACT:]
                history_str = ' '.join(f"Step {i}. {remove_redundant_spaces(instruc.replace('  ',' ').strip(' .'))}." for i, instruc in enumerate(retained_history, start=max(1,len(clean_prev_step_instructions) - MAX_PREV_ACT+1))) if len(retained_history) > 0 else 'None'

                if 'tars' in args.planner.lower():
                    prompt = MIND2WEB_PLANNING_PROMPT_UI_TARS.format(global_task=goal)
                else:
                    prompt = MIND2WEB_PLANNING_PROMPT_COT.format(global_task=goal, history=history_str)

                retry = 0
                temperature = 0.0
                while retry < 3:
                    try:    # several sample's img dir lead to error, just jump it
                        if args.provider == 'openai':
                            if 'tars' in args.planner.lower():
                                messages = make_ui_tars_messages(
                                    initial_instruc=prompt,
                                    past_obs=prev_obs,
                                    past_action_strs=past_action_strs,
                                    past_actions=[JSON2UITARS_action(x['GT_action']) for x in results_actions],
                                    max_history=5, # UI-TARS only supports max 5 history
                                    conv_mode=args.conv_mode
                                ) 
                                _, response, _ = openai.get_model_response_with_prepared_messages(messages, temperature=temperature)
                            else:
                                _, response, _ = openai.get_model_response(prompt, [img_path], use_img_url=True, temperature=temperature)
                        elif 'aguvis' in args.planner.lower():
                            prompt, response = vlm.get_model_response(instruction=goal, image=img_path, previous_actions=clean_prev_step_instructions, platform='web')
                        else:
                            response = vlm.get_model_response(prompt, f"file://{img_path}", temperature=temperature)

                        if args.provider == 'llama3' and '**' in response:
                            obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components_llamav(response)
                        elif 'tars' in args.planner.lower():
                            obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components_UITARS(response)
                            action_pred_raw = parse_UITARS_action(action_pred_raw, platform='mind2web')
                        elif 'aguvis' in args.planner.lower():
                            obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components_aguvis(response)
                            action_pred_raw = parse_AGUVIS_action(action_pred_raw, platform='mind2web')
                        else:
                            obs, thought, funcdesc, action_pred_raw, summary = extract_thought_components(response)
                        
                        assert len(funcdesc) > 0 and len(summary) > 0
                        break
                    except Exception as e:
                        print(f"invalid response: {response}")
                        traceback.print_exc()
                        retry += 1
                        temperature = 0.6

                step_result = {"img_path": os.path.basename(img_path), "task": goal, "prompt": prompt, "response": response,
                            "GT_action": action_step, "GT_box": bbox_ref,
                            "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}

                try:
                    # action_pred = ast.literal_eval(response)
                    # 典型乱输出例子：# '{"action_type": "SELECT, CLICK", "target": (39,51), "value": "Things To Do"}'
                    # '{"action_type": 7, "target": (80,63), "value": "Desktop Computer"}'
                    # '{"action_type": CLICK, "target": (10,93}, CLICK)'
                    # '{"action_type": CLICK, "target": (11,80}}'
                    action_pred = ast.literal_eval(action_pred_raw)
                    
                    if action_pred["action_type"] in ['click', 'hover']: action_pred["action_type"] = 'click'

                    if action_pred["action_type"] == action_step_ref["action_type"] or action_pred["action_type"] == action_step_ref.get("ori_act", "").lower():
                        step_result["Op_match"] = True

                    click_point = action_pred.get("target", (-1.0,-1.0))

                    if action_pred["action_type"] == 'enter':
                        step_result["Ele_match"] = step_result["Op_match"]
                        step_result["Op_F1"][0] = 1.0
                    else:
                        if (bbox_ref[0] <= click_point[0] / SCALE <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] / SCALE <= bbox_ref[3]):
                            step_result["Ele_match"] = True
                        elif click_point == (-1, -1):
                            step_result["Ele_match"] = True

                        # 按照mind2web的方式，把action转换成一个字符串，即如果是TYPE需要考虑字符间的F1
                        pred_str = str(action_pred["action_type"])
                        if action_pred["action_type"] in [3, "input_text"] or action_pred["action_type"] in [2, "select"]:
                            pred_str += ' '
                            pred_str += action_pred.get("text",  action_pred.get("value")).lower()
                        ref_str = str(action_step_ref["action_type"])
                        if action_step_ref["action_type"] in [3, "input_text"] or action_step_ref["action_type"] in [2, "select"]:
                            ref_str += ' '
                            ref_str += action_step_ref["value"].lower()

                        op_f1 = squad_metrics.compute_f1(pred_str, ref_str)
                        step_result["Op_F1"][0] = op_f1

                        # For UI-tars
                        if op_f1 > 0.9 and step_result["Ele_match"] and 'Thought:' in response and 'Action:' in response:
                            past_action_strs[-1] = response.split('Action:')[0].strip()

                    print(f"Op: {step_result['Op_match']} | Elem: {step_result['Ele_match']} |  Elem: {step_result['Op_F1']} | GT:{action_step_ref} <=> {action_pred}")
                
                    step_result["observation"], step_result["thought"], step_result["funcdesc"], step_result["summary"], step_result["action_pred"] = obs, thought, funcdesc, summary, action_pred
                except Exception as e :
                    traceback.print_exc()
                    print("format wrong")
                    step_result["status"] = "wrong format"

                action_step_ref['box'] = bbox_ref

                results_actions.append(step_result)

            results_all_tasks[task].append(results_actions)

        # calculate metrics
        num_step = 0
        num_episode = 0
        num_op = 0
        num_ele = 0
        op_f1 = {"click": [], "select": [], "input_text": []}
        macro_ele_acc = {}
        macro_step_acc = {}
        macro_action_f1 = {}
        num_step_success = 0
        num_episode_success = 0
        for i, item in enumerate(results_all_tasks[task]):
            macro_ele_acc[i] = []
            macro_step_acc[i] = []
            macro_action_f1[i] = []
            num_episode += 1
            episode_success = True
            for step_result in item:
                num_step += 1

                if step_result["Op_match"]:
                    num_op += 1

                if step_result["Ele_match"]:
                    num_ele += 1
                    macro_ele_acc[i].append(1)
                else:
                    macro_ele_acc[i].append(0)

                if step_result["Op_F1"][1] in op_f1:
                    op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
                macro_action_f1[i].append(step_result["Op_F1"][0])

                if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                    num_step_success += 1
                    macro_step_acc[i].append(1)
                else:
                    macro_step_acc[i].append(0)
                    episode_success = False

            if episode_success:
                num_episode_success += 1

        marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values() if len(x) > 0])

        eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/mind2web', postfix)
        os.makedirs(eval_result_dir, exist_ok=True)

        print("Operation F1: " + str(marco_op_f1))
        print("Element Acc: " + str(num_ele / num_step))
        print("Step Success: " + str(num_step_success / num_step))
        print("Episode Success: " + str(num_episode_success / num_episode))
        print("Operation F1 cate: " + str([np.mean(x) if len(x) else 0.0 for x in op_f1.values()]))

        macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values() if len(x) > 0]).item()
        macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values() if len(x) > 0]).item()
        macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values() if len(x) > 0]).item()
        print("Macro Ele Acc: " + str(macro_ele_acc))
        print("Macro Op F1: " + str(macro_action_f1))
        print("Macro Step SR: " + str(macro_step_acc))

        print(f"{macro_ele_acc*100:2f} | {macro_action_f1*100:.2f} | {macro_step_acc*100:.2f}")

        metrics_all_tasks[task] = {
            "Operation F1": marco_op_f1,
            "Element Acc": num_ele / num_step,
            "Step Success": num_step_success / num_step,
            "Episode Success": num_episode_success / num_episode,
            "Operation F1 cate": str([np.mean(x).item() if len(x) else 0.0 for x in op_f1.values()]),
            'Macro Ele Acc': macro_ele_acc,
            'Macro Op F1': macro_action_f1,
            'Macro Step SR': macro_step_acc
        }

        with open(save_file, "w") as f:
            json.dump(
                {
                    "meta": vars(args),
                    "overall_results": metrics_all_tasks,
                    "log": results_all_tasks,
                },
                f,
                indent=2
            )

        print(f"Finised evaluation {args.planner} on Mind2Web. Save to {save_file}")