import re, torch, cv2
from PIL import Image

ACTION_MAPPING = {
    "CLICK": 4,
    "SELECT": 2,
    "TYPE": 3
}

def map_box_to_point(box: list):
    x1, y1, x2, y2 = box
    return [(x1+x2)/2, (y1+y2)/2]

letter_idx_pattern_1 = re.compile(r'(?<![\w])([A-Z])\.')
letter_idx_pattern_2 = re.compile(r'(?<![\w])([A-Z])\:')

def extract_letter_index(text, num_choices):
    # exp1: E. (13,89)
    text = text.strip()
    
    if "None" in text:
        letter_index = -1
    elif len(text) == 1:
        letter_index = ord(text)
    else:
        letter_1 = letter_idx_pattern_1.search(text)
        letter_2 = letter_idx_pattern_2.search(text)
        
        letter_1_index = ord(letter_1.group(1)) if letter_1 is not None else -1
        letter_2_index = ord(letter_2.group(1)) if letter_2 is not None else -1
            
        if 0 < letter_1_index - 65 < num_choices:
            letter_index = letter_1_index
        elif 0 < letter_2_index - 65 < num_choices:
            letter_index = letter_2_index
        else:
            letter_index = None
    
    return letter_index

def mind2web_action2step(action, image_size, scale=1, use_action_id=True, return_bbox=False):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    
    if scale == 1:
        click_point = [round(item, 3) for item in click_point]
        click_point = [f"{item:.2f}" for item in click_point]
    else:
        click_point = [f"{int(scale * item):d}" for item in click_point]

    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"], action["bbox"]["y"] + action["bbox"]["height"]]

        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = '{{"action_type": {}, "target": {}, "ori_act": "{}"}}'.format('"click"', click_point, action_type)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = '{{"action_type": {}, "target": {}, "value": "{}"}}'.format('"select"', click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = '{{"action_type": {}, "target": {}, "value": "{}"}}'.format('"input_text"', click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step

# convert action to prediction format
def aitw_action2step(step_data, scale=1):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            
            if scale == 1:
                click_point = [f"{item:.2f}" for item in click_point]
            else:
                click_point = [f"{int(item*scale):d}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"target\": {}}}".format(action_type_new, click_point)
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


import base64, openai, re
from colorama import Fore, Style

HOMEPAGES = {
    'kuaishou': "com.smile.gifmaker/com.yxcorp.gifshow.HomeActivity",
    'douyin': "com.ss.android.ugc.aweme/.splash.SplashActivity"
}

def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL, end='')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class OpenAIModel:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 1024):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        self.p_tokens = self.cmpl_tokens = 0


    def get_model_response(self, prompt: str, images) -> (bool, str):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": content
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=30
        )
        
        usage = response.usage
        self.p_tokens += usage.prompt_tokens
        self.cmpl_tokens += usage.completion_tokens
        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            print_with_color(f"Request cost is "
                             f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                             "yellow")
        else:
            return False, response.error["message"]
        return True, response.choices[0].message.content