import os, json, random, cv2
import numpy as np
from colorama import Fore, Style

SCALE = 1000

main_file = "utils/eval_utils/eval_aitw_with_funcgnd/eval_results/Qwen-Qwen2-VL-72B-Instruct/2024-12-20-08-49-48.json"

gnd_file = "utils/eval_utils/eval_aitw_with_funcgnd/eval_results/Qwen-Qwen2-VL-72B-Instruct/2024-12-20-08-49-48_RevisedBy_merged_checkpoint-3832.json"

results = json.load(open(gnd_file))['logs']

img_dir = "/data0/jingran/workspace/UI_training_data/AITW/aitw_images"
for task, samples in results.items():
    for sample in samples:
        img_file = os.path.join(img_dir, sample['step_info']['image'])
        img = cv2.imread(img_file)
        H, W = img.shape[:2]

        if not (sample['status'].startswith('correct') and not sample['new_status'].startswith('correct')): continue

        if 'scroll' not in sample['step_info']['step']['action_type_text'] and 'target' in sample['action_pred']:
            old_action, new_action = sample['action_pred'], sample['grounder_action_pred']
            
            old_target, new_target = old_action['target'], new_action['target']
            
            old_target = [round(old_target[0] / SCALE * W), round(old_target[1] / SCALE * H)]
            new_target = [round(new_target[0] * W), round(new_target[1] * H)]
            
            cv2.circle(img, old_target, radius=8, color=(0,160,0), thickness=2)
            cv2.circle(img, new_target, radius=8, color=(0,255,0), thickness=2)
            
            # GT
            action_ref = sample['action_ref']
            y, x = action_ref['touch_point']
            gt_target = [round(x * W), round(y * H)]
            
            cv2.circle(img, gt_target, radius=8, color=(0,0,255), thickness=2)
            cv2.circle(img, gt_target, radius=1, color=(0,0,255), thickness=-1)
            
            cv2.imwrite("test.png", img)
            
            print(Fore.GREEN + f"Task: {sample['step_info']['step']['goal']} | Step instruc: {sample['step_info']['step']['action_addition']}\nFuncDesc: {sample['funcdesc']}\n" + Fore.CYAN + f"GT: {action_ref['action_type']} {gt_target}\n" + Fore.YELLOW + f"Old: {old_target}\tNew: {new_target}" + Style.RESET_ALL)
            
            1+1