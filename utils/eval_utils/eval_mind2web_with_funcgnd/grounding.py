# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score
import os, time
import random
import torch
import json
from tqdm import tqdm
import ast
import argparse
from PIL import Image
import numpy as np
from copy import deepcopy
from datetime import datetime
from utils.eval_utils.eval_utils import mind2web_action2step
from utils.data_utils.task_prompt_lib import *
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.autogui_florence2 import AutoGUI_Florence

from utils.data_utils.misc import pred_2_point, TextProcessor
from utils.openai_utils.misc import extract_thought_components


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
    parser.add_argument('--planning_result_file', type=str, default='utils/eval_utils/eval_mind2web_with_funcgnd/eval_results/gpt-4o/2025-12-05-19-40-18.json')
    parser.add_argument('--grounder', type=str, default=[
        'gpt-4o-mini', # provider: openai
        '/mnt/vdb1/hongxin_li/uipro_ckpt/1222_Qwen2vl-7B-FuncGnd490k/lora/checkpoint-7664/', # provider: qwen2-vl
        'cckevinn/SeeClick', # provider: qwenvl
        'OS-Copilot/OS-Atlas-Base-7B', # provider: qwen2-vl
        'Samsung/TinyClick', # provider: tinyckick
        'THUDM/cogagent-chat-hf', # provider: cogagent
        '/mnt/vdb1/hongxin_li/uipro_ckpt/0223_IJCV-UIPro-Florence-Large+AgentTaskIntentGnd_4295k/checkpoint-33556' # provider: autogui_florence
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
        
        USE_FUNCDESC = True
        SCALE = 1000
    else:
        model_path = args.grounder.rstrip('/')
        print(f"Loading grounder from {args.grounder}")
        if 'florence' in model_path.lower():
            vlm = AutoGUI_Florence(device='cuda', model_name=model_path)
            SCALE = 1000
            USE_FUNCDESC = False#'intent' not in model_path.lower()
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
    planning_eval_results_all_tasks = planning_eval_results_info["log"]


    mind2web_imgs_dir = os.path.join(args.mind2web_dir, "mind2web_images")

    results_all_tasks = {'website': [], 'task': [], 'domain': []}
    metrics_all_tasks = {'website': [], 'task': [], 'domain': []}

    # text processor
    tproc = TextProcessor()

    for task, results in planning_eval_results_all_tasks.items():
        mind2web_test = json.load(open(f'{args.mind2web_dir}/mind2web_data_test_' + task + '.json', 'r'))

        for ep_idx, (episode, episode_result) in tqdm(enumerate(zip(mind2web_test, results)), total=len(mind2web_test), desc=f"Evaluating {task} | {postfix} | {'UseFuncGnd' if USE_FUNCDESC else 'UseIntentGnd'}"):
            if args.debug and ep_idx > 2:
                break

            goal = episode["confirmed_task"]
            annot_id = episode["annotation_id"]
            previous_actions = []
            results_actions = []
                    
            for step_i, (step, step_result) in enumerate(zip(episode["actions"], episode_result)):
                if args.debug and step_i > 3:
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

                # Used in the prompt
                action_step, bbox_ref = mind2web_action2step(step, image.size, scale=SCALE, return_bbox=True)

                try:
                    action_step_ref = ast.literal_eval(action_step)
                except:
                    continue

                try:
                    funcdesc = step_result.get('funcdesc','').strip(' .') # if the original response's format is wrong, this will also be recorded as a wrong format error.
                    intent = step_result.get('summary','').replace('I ','').strip(' .')

                    if 'action_pred' not in step_result:
                        if args.provider == 'llama3' and '**' in step_result['response']:
                            _, _, _, action_pred_raw, _ = extract_thought_components_llamav(step_result['response'])
                        elif 'tars' in args.planning_result_file.lower():
                            _, _, _, action_pred_raw, _ = extract_thought_components_UITARS(step_result['response'])
                            action_pred_raw = parse_UITARS_action(action_pred_raw, platform='mind2web')
                        else:
                            _, _, _, action_pred_raw, _ = extract_thought_components(step_result['response'])

                        step_result['action_pred'] = eval(action_pred_raw)

                    new_action = deepcopy(step_result['action_pred'])

                    if len(intent.split(' ')) > 6 and intent.lower().startswith('click'):
                        prompt = PROMPT.format(funcdesc if USE_FUNCDESC else tproc.extract_main_clause(intent))

                        if args.provider == 'openai':
                            _, response, _ = openai.get_model_response(prompt, [img_path], use_img_url=True)
                        else:
                            response = vlm.get_model_response(prompt, f"file://{img_path}")
                
                        real_target = pred_2_point(response, keep_box=False, scale=1)

                        step_result['grounder_prompt'] = prompt
                        step_result['grounder_response'] = response
                        new_action['target'] = real_target
                except Exception as e:
                    print(e)

                try:
                    step_result["new_Op_match"] = False
                    step_result["new_Ele_match"] = False
                    step_result["new_Op_F1"] = [0, action_step_ref["action_type"]]
                    action_pred = step_result["grounder_action_pred"] = new_action
                    # action_pred = ast.literal_eval(response)
                    # 典型乱输出例子：# '{"action_type": "SELECT, CLICK", "target": (39,51), "value": "Things To Do"}'
                    # '{"action_type": 7, "target": (80,63), "value": "Desktop Computer"}'
                    # '{"action_type": CLICK, "target": (10,93}, CLICK)'
                    # '{"action_type": CLICK, "target": (11,80}}'
                    
                    if action_pred["action_type"] in ['click', 'hover']: action_pred["action_type"] = 'click'

                    if action_pred["action_type"] == action_step_ref["action_type"] or action_pred["action_type"] == action_step_ref.get("ori_act", "").lower():
                        step_result["new_Op_match"] = True

                    click_point = action_pred.get("target", (-1.0,-1.0))

                    if action_pred["action_type"] == 'enter':
                        step_result["new_Ele_match"] = step_result["new_Op_match"]
                        step_result["new_Op_F1"][0] = 1.0
                    else:
                        if (bbox_ref[0] <= click_point[0] / SCALE <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] / SCALE <= bbox_ref[3]):
                            step_result["new_Ele_match"] = True

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
                        step_result["new_Op_F1"][0] = op_f1
                    
                    print(f"Old Op_F1: {step_result['Op_F1'][0]} / Old Ele_match: {step_result['Ele_match']} | New Op_F1: {step_result['new_Op_F1'][0]} / New Ele_match: {step_result['new_Ele_match']} || GT:{action_step_ref} <=> {action_pred}")
                except Exception as e :
                    print(e)
                    print("format wrong")
                    step_result["status"] = "wrong format"

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

                if step_result["new_Op_match"]:
                    num_op += 1

                if step_result["new_Ele_match"]:
                    num_ele += 1
                    macro_ele_acc[i].append(1)
                else:
                    macro_ele_acc[i].append(0)

                if step_result["new_Op_F1"][1] in op_f1:
                    op_f1[step_result["new_Op_F1"][1]].append(step_result["new_Op_F1"][0])
                macro_action_f1[i].append(step_result["new_Op_F1"][0])

                if step_result["new_Op_F1"][0] == 1.0 and step_result["new_Ele_match"]:
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

        elem_acc = num_ele / num_step if num_step > 0 else 0
        step_success = num_step_success / num_step if num_step > 0 else 0
        episode_success = num_episode_success / num_episode if num_episode > 0 else 0
        print("Element Acc: " + str(elem_acc))
        print("Step Success: " + str(step_success))
        print("Episode Success: " + str(episode_success))
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
            "Element Acc": elem_acc,
            "Step Success": step_success,
            "Episode Success": episode_success,
            "Operation F1 cate": str([np.mean(x).item() if len(x) else 0.0 for x in op_f1.values()]),
            'Macro Ele Acc': macro_ele_acc,
            'Macro Op F1': macro_action_f1,
            'Macro Step SR': macro_step_acc
        }

    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_file = args.planning_result_file.replace(".json", f"_RevisedBy_{postfix}{'_FuncGnd' if USE_FUNCDESC else '_IntentGnd'}_{time_str}.json")

    print(f"Finished evaluating {args.grounder} at {time_str}. Save eval results to {save_file}")

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