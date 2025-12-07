import os, json, cv2, numpy as np
import datasets
from tqdm import tqdm
from copy import deepcopy
from utils.eval_utils.action_matching import *

aitw_imgs_dir = '/data0/jingran/workspace/UI_training_data/AITW/aitw_images'
result_file = "utils/eval_utils/eval_aitw_with_funcgnd/eval_results/Qwen-Qwen2-VL-72B-Instruct/2024-12-20-08-49-48_RevisedBy_merged_checkpoint-7664.json"
planning_eval_results_info = json.load(open(result_file))

meta = planning_eval_results_info["meta"]
planning_eval_results = planning_eval_results_info["logs"]

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

SCALE = 1000

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

    for step_i, step in tqdm(enumerate(steps), total=len(steps), desc=f"{task}"):
        img_filename = step["image"]
        img_path = os.path.join(aitw_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not found")
            continue

        goal = step["step"]["goal"]

        action_ref = action_2_format(step["step"])
        step_id = f"{step['step']['ep_id']}-{step['step']['step']}"
        temp = planning_result_this_task[step_id]
        
        funcdesc = planning_result_this_task[step_id]['funcdesc'] # if the original response's format is wrong, this will also be recorded as a wrong format error.
        intent = planning_result_this_task[step_id].get('summary','')
        raw_prompt, raw_response, raw_status = planning_result_this_task[step_id]['prompt'], planning_result_this_task[step_id]['response'], planning_result_this_task[step_id]['status']
        
        old_action = planning_result_this_task[step_id]['action_pred']
        new_action = deepcopy(old_action)

        num += 1
        try:
            if "wrong format" in temp['status']:
                raise Exception("wrong format")

            if 'grounder_action_pred' in temp:
                action_pred = pred_2_format(temp['grounder_action_pred'], scale=1)
            else:
                action_pred = pred_2_format(temp['action_pred'], scale=1000)
            
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
                
                if action_ref['action_type'] in [4, 'click'] and action_ref['touch_point'] == action_ref['lift_point']:
                    H,W = img.shape[:2]
                    cv2.circle(img, [round(action_ref['touch_point'][1]*W), round(action_ref['touch_point'][0]*H)], 6, (0,255,0), 2)
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

            # type accuracy
            if action_pred["action_type"] == action_ref["action_type"]:
                corr_type += 1
                temp["status"] += "action_correct"
                
                # step accuracy
                if check_match == True:
                    corr_action += 1
                    match_label = 1
                    #print("Step: " + str(j) + " right")
                    temp["status"] = "correct," + temp["status"]
                else:
                    match_label = 0
                    #print("Step: " + str(j) + " wrong")
                    temp["status"] = "wrong," + temp["status"]

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
            else: temp["status"] += "action_wrong"     

            temp["grounder_action_pred"] = new_action
        except:
            num_wrong_format += 1
            # print("Step: " + str(j) + " wrong format")
            temp["status"] = "wrong format"

        tasks_logs[task].append(temp)

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

print("Average score: " + str(score_average / len(tasks_result)))

tasks_result['avg'] = score_average / len(tasks_result)

save_to = result_file.replace(".json", f"_Recalc.json")

with open(save_to, "w") as f:
    json.dump({"meta": meta, "eval_result": tasks_result, "logs": tasks_logs}, f, indent =2)

print(f"Finish processing {result_file}")