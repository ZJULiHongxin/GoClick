# 🎯 GoClick: Two-Stage UI Grounding with Function Description

<div align="center">

**A powerful two-stage framework for precise UI element grounding using function descriptions**

[![Paper](https://img.shields.io/badge/Paper-IJCV-blue)](IJCV_GoClick.pdf)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Benchmarks](#-benchmarks)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

**GoClick** is a novel two-stage framework for UI element grounding that separates the planning and grounding tasks. Instead of directly predicting click coordinates, GoClick first generates a function description of the target element, then uses this description to precisely locate the element in the UI screenshot.

### Why Two-Stage?

- **Better Generalization**: Function descriptions are more robust across different UI layouts
- **Improved Accuracy**: Separating planning and grounding allows each stage to specialize
- **Interpretability**: Function descriptions provide clear reasoning for element selection

---

## ✨ Key Features

- 🎯 **Two-Stage Architecture**: Planning → Grounding pipeline
- 🧠 **Function Description**: Generates semantic descriptions of target UI elements
- 🔧 **Florence-2 Based**: Built on Microsoft's Florence-2-large vision-language model
- 📊 **Multi-Benchmark Support**: Evaluated on AITW, AndroidControl, GUIAct, and Mind2Web
- 🚀 **Easy Training**: Simple training script with HuggingFace Transformers
- 📈 **Comprehensive Evaluation**: Complete evaluation pipeline for all supported benchmarks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Planning                        │
│  Input: UI Screenshot + Task Goal + History                │
│  Output: Action Type + Function Description + Intent       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Stage 2: Grounding                         │
│  Input: UI Screenshot + Function Description               │
│  Output: Target Element Coordinates                        │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1: Planning
The planner analyzes the UI screenshot and task context to:
- Determine the **action type** (click, scroll, input_text, etc.)
- Generate a **function description** of what the target element should do
- Extract the **intent** summarizing the action

### Stage 2: Grounding
The grounder uses the function description to:
- Locate the precise UI element matching the description
- Return the **target coordinates** for the action

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/ZJULiHongxin/GoClick
cd GoClick
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Note: Flash Attention should be installed separately to match the PyTorch and CUDA version.
```

---

## 🚀 Quick Start

### Using Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


def postprocess(text: str, image_size: tuple[int]):
    """Function that decodes model's generation into action json.

    Args:
        text: single generated sample
        image_size: corresponding image size
    """
    point_pattern = r"<loc_(\d+)>,<loc_(\d+)>"

    try:
        location = re.findall(point_pattern, text)[0]
        if len(location) > 0:
            point = [int(loc) for loc in location]

    except Exception:
        point = (0, 0)

    return point

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("HongxinLi/GoClick-Large")
processor = AutoProcessor.from_pretrained("HongxinLi/GoClick-Large")

# Load UI screenshot
image = Image.open("ui_screenshot.png")

# Stage 1: Planning

# Functionality Grounding (For AutoGUI FuncPred Benchmark)
planning_prompt = f"Locate the element according to its detailed functionality description. {goal_info} (Output the center coordinates of the target)"

# Intent Grounding (For RefExp, MOTIF, and VisualWebBench Action Grounding)
planning_prompt = f"I want to {goal_info}. Please locate the target element I should interact with. (Output the center coordinates of the target)"

# Description Grounding (For ScreenSpot/v2 and VisualWebBench Element Grounding))
planning_prompt = f"Where is the {goal_info} element? (Output the center coordinates of the target)"


inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
    do_resize=True,
).to(model.device, dtype=model.dtype)

outputs = model.generate(
            **inputs,
            do_sample= False,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )

text_output = processor.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
text_output = postprocess(text_output, img_size)

```

---

## 🎓 Training

### Data Preparation

The training data is available on HuggingFace at [HongxinLi/GoClick_sft_data](https://huggingface.co/datasets/HongxinLi/GoClick_sft_data). Download via `hf download HongxinLi/GoClick_sft_data  --repo-type dataset --local-dir path/to/GoClick_sft_data`, unzip, and organize it as follows:

```
root/
├── GoClick_sft_data/
│   ├── GoClick_images
│   └── GoClick_CoreSet-v2_3814k_florence.jsonl
```

Update the `data_path=GoClick_CoreSet-v2_3814k_florence.jsonl` in `florence2/sft.sh`:

```bash
# Download from HuggingFace (update with actual dataset name)
# data_path=YOUR_HUGGINGFACE_DATASET_PATH
```

### Training Configuration

Edit `florence2/sft.sh` to configure your training:

```bash
model_name=microsoft/Florence-2-large
data_path=YOUR_DATA_PATH  # Update with HuggingFace dataset path
output_dir=YOUR_OUTPUT_DIR

torchrun --nproc_per_node 8 --nnodes 1 --master_port 16252 \
    florence2/finetune.py \
    --model_name_or_path $model_name \
    --florence_path $model_name \
    --data_path $data_path \
    --bf16 True \
    --fix_vit True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 2 \
    --report_to none \
    --run_name GoClick-Training \
    --model_max_length 1024 \
    --lazy_preprocess True
```

### Start Training

```bash
bash florence2/sft.sh
```

### Training Options

Key training arguments:

- `--model_name_or_path`: Base model (choices: `microsoft/Florence-2-large` and `microsoft/Florence-2-base`)
- `--data_path`: Path to training data (JSONL format)
- `--output_dir`: Output directory for checkpoints
- `--bf16`: Use bfloat16 precision
- `--fix_vit`: Freeze vision encoder (recommended)
- `--use_lora`: Enable LoRA fine-tuning (optional)
- `--model_max_length`: Maximum sequence length

---

## 📊 GUI Grounding Evaluation
We recommend using [AutoGUI evaluation kit](https://autogui-project.github.io/) to perform GUI element grounding evaluation on multiple GPUs.


## 📊 Agent Task Evaluation

GoClick provides comprehensive evaluation scripts for multiple benchmarks. The evaluation follows a two-stage process:

1. **Planning Stage**: Run the planning script to generate action predictions with function descriptions
2. **Grounding Stage**: Run the grounding script to refine predictions using the function descriptions

### AITW Benchmark

Firstly, download the AITW screenshot images from [SeeClick AITW Data](https://box.nju.edu.cn/f/96ba5115bae24eaaa44e/), unzip it, and organize it as follows:

```
root/
├── AITW/
│   ├── aitw_images
│   ├── aitw_data_test.json
│   ├── aitw_data_train.json
│   └── aitw_data_val.json
```

#### Stage 1: Planning


```bash
python utils/eval_utils/eval_aitw_with_funcgnd/eval_aitw_with_funcgnd.py \
    --planner gpt-4o \
    --provider openai \
    --imgs_dir /path/to/AITW/aitw_images/ \
    --debug  # Remove for full evaluation
```

#### Stage 2: Grounding
```bash
python utils/eval_utils/eval_aitw_with_funcgnd/grounding.py \
    --planning_result_file utils/eval_utils/eval_aitw_with_funcgnd/eval_results/gpt-4o/TIMESTAMP.json \
    --grounder /path/to/your/grounder/model \
    --provider autogui_florence \
    --imgs_dir /path/to/AITW/aitw_images/
```

### AndroidControl Benchmark

First download and unzip the [GoClick AndroidControl Test Data](https://huggingface.co/datasets/HongxinLi/AndroidControl_test) via 
```
hf download HongxinLi/AndroidControl_test  --repo-type dataset --local-dir path/to/AndroidControl_test
```


#### Stage 1: Planning
```bash
python utils/eval_utils/eval_andcon_with_funcgnd/eval_androidcontrol_with_funcgnd.py \
    --andcon_dir path/to/AndroidControl_test \
    --planner gpt-4o \
    --provider openai \
    --debug False \
    --max_prev_acts 6
```

#### Stage 2: Grounding
```bash
python utils/eval_utils/eval_andcon_with_funcgnd/grounding.py \
    --andcon_dir path/to/AndroidControl_test \
    --planning_result_file utils/eval_utils/eval_andcon_with_funcgnd/eval_results/gpt-4o/TIMESTAMP.json \
    --grounder /path/to/your/grounder/model \
    --provider autogui_florence \
    --max_prev_acts 6
```

### GUIAct Benchmark

First download the GUIAct data from [HongxinLi/GUIAct](https://huggingface.co/datasets/HongxinLi/GUIAct), unzip it, and organize it as follows:

```
root/
├── GUICourse/
│   └── GUIAct/
│   │    └── imgs
│   ├── Web_test.json
│   └── Mobile_test.json
```

#### Stage 1: Planning
```bash
python utils/eval_utils/eval_guiact_with_funcgnd/eval_guiact_with_funcgnd.py \
    --guicourse_dir root/GUICourse \
    --planner gpt-4o \
    --provider openai \
    --device_type Mobile \ # or Web
    --debug False \
    --max_prev_acts 6
```

#### Stage 2: Grounding
```bash
python utils/eval_utils/eval_guiact_with_funcgnd/grounding.py \
    --guicourse_dir root/GUICourse \
    --planning_result_file utils/eval_utils/eval_guiact_with_funcgnd/eval_results/GUIAct-Mobile/gpt-4o/TIMESTAMP.json \
    --grounder /path/to/your/grounder/model \
    --provider autogui_florence
```

### Mind2Web Benchmark

Firstly, download the Mind2Web screenshot images from [SeeClick Mind2Web images](https://box.nju.edu.cn/f/33e203d170ab48b0b922/) and test set JSON files from [SeeClick Mind2Web annotations](https://box.nju.edu.cn/f/e30b861fa7604668821b/), unzip them, and organize them as follows:

```
root/
├── Mind2Web/
│   ├── mind2web_images/
│   ├── mind2web_data_test_domain.json
│   ├── mind2web_data_test_task.json
│   └── mind2web_data_test_website.json
```

#### Stage 1: Planning
```bash
python utils/eval_utils/eval_mind2web_with_funcgnd/eval_mind2web.py \
    --mind2web_dir root/Mind2Web/mind2web_images/ \
    --planner gpt-4o \
    --provider openai \
    --scale 1000 \
    --debug False \
    --max_prev_acts 9
```

#### Stage 2: Grounding
```bash
python utils/eval_utils/eval_mind2web_with_funcgnd/grounding.py \
    --mind2web_dir root/Mind2Web/mind2web_images/ \
    --planning_result_file utils/eval_utils/eval_mind2web_with_funcgnd/eval_results/gpt-4o/TIMESTAMP.json \
    --grounder /path/to/your/grounder/model \
    --provider autogui_florence
```

### Evaluation Options

Common arguments for evaluation scripts:

- `--planner`: Model for planning stage (e.g., `gpt-4o`, `Qwen/Qwen2.5-VL-7B-Instruct`)
- `--provider`: API provider (`openai`, `qwen2-vl`, `llama3`)
- `--grounder`: Path to grounding model checkpoint
- `--imgs_dir`: Directory containing benchmark images
- `--debug`: Run on a small subset for testing
- `--max_prev_acts`: Maximum number of previous actions in history

---

## 📈 Benchmarks

GoClick is evaluated on four major UI grounding benchmarks:

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| **AITW** | Android In The Wild | Action Accuracy, Element Accuracy, Click Accuracy |
| **AndroidControl** | Android UI Control | Step Accuracy, Element Accuracy, Action Type Accuracy |
| **GUIAct** | GUI Action Dataset | Step Accuracy, Element Accuracy (Web & Mobile) |
| **Mind2Web** | Web Navigation | Operation F1, Element Accuracy, Step Success Rate |

### Benchmark Data

- **AITW**: Available via `datasets.load_dataset("HongxinLi/AITW_test", split='test')`
- **AndroidControl**: Download from [official repository](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
- **GUIAct**: Available in processed format (see evaluation scripts for paths)
- **Mind2Web**: Download from [official repository](https://github.com/OSU-NLP-Group/Mind2Web)

---

## 🎯 Results

*(Update with your actual results from the paper)*

### Performance Highlights

- **AITW**: [Your results]
- **AndroidControl**: [Your results]
- **GUIAct**: [Your results]
- **Mind2Web**: [Your results]

### Key Findings

- Function descriptions significantly improve grounding accuracy
- Two-stage approach outperforms end-to-end methods
- Better generalization across different UI layouts

---

## 📝 Citation

If you use GoClick in your research, please cite our paper:

```bibtex
@article{goclick2024,
  title={GoClick: Two-Stage UI Grounding with Function Description},
  author={Your Name and Collaborators},
  journal={International Journal of Computer Vision},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on [Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- Uses [HuggingFace Transformers](https://github.com/huggingface/transformers)
- Evaluation benchmarks: AITW, AndroidControl, GUIAct, Mind2Web

---

<!-- ## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

--- -->

<div align="center">

**⭐ If you find GoClick useful, please star this repository! ⭐**

</div>
