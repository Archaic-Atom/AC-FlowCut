# AC-FlowCut:Set-Level Attention-Output Repair for Training-Free Visual Token Pruning

AC-FlowCut is a training-free visual-token pruning patch for LLaVA-style vision-language models. It keeps FlowCut's progressive information-flow trajectory and high-confidence prefix, then repairs only the final boundary. The repair evaluates a small candidate band with a multi-query, attention-output reconstruction proxy. It does not train parameters, merge hidden states, transport discarded states, or change the requested final patch budget.

## Supported Runtime

The code has been validated in the following environment:

- **Python**: 3.10.x (recommended 3.10.19)
- **PyTorch**: 2.1.2 + **torchvision**: 0.16.2
- **Transformers**: 4.37.2
- **Flash-Attention**: 2.x (`flash-attn --no-build-isolation`)
- **Base Framework**: [LLaVA](https://github.com/haotian-liu/LLaVA) source tree
- **Supported Models**: 
  - LLaVA-1.5-7B (square CLIP inputs)
  - LLaVA-NeXT-7B (any-resolution spatial merge path)

> **Token Budget Note:**  
> The class token is always retained and is not counted in `target_num`. For a square image with `target_num=64`, the visual span contains 64 patch tokens. For any-resolution inputs, spatial merge and unpadding may leave fewer than the requested number of valid patches; the runtime enforces a positive global span no larger than the target.

## Installation

### 1. Create Conda Environment

```bash
conda create -n ac-flowcut python=3.10 -y
conda activate ac-flowcut
pip install --upgrade pip
```

### 2. Install LLaVA & Dependencies

AC-FlowCut operates on top of LLaVA. Install LLaVA and Flash-Attention from the official repository:

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install flash-attn --no-build-isolation
```

### 3. Install AC-FlowCut

Clone this repository and install it in editable mode:

```bash
git clone https://github.com/Archaic-Atom/AC-FlowCut.git  
cd AC-FlowCut
pip install -r requirements.txt
pip install -e .
```

File organization as follow::
```
├── LLaVA-main
    ├── ac-flowcut
    ├── llava
    ├── playground
    ├── script
```

### 4. Offline Environment Setup (Optional)

If running in an isolated environment with pre-cached models and CLIP checkpoints:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Quick Start

Load a LLaVA model with the standard LLaVA loader, then apply AC-FlowCut before calling `generate`:

```python
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from ac-flowcut import ac_flowcut

model_path = "/path/to/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
)

# Keep 64 visual patch tokens. Model weights remain frozen.
model = ac_flowcut(model, target_num=64)
```

The original FlowCut selector is also available as a matched baseline:

```python
from ac-flowcut import flowcut

model = flowcut(model, target_num=64)
```

For custom configurations (e.g. tuning core protection or query scopes), use `apply_token_reduction`:

```python
from ac-flowcut import apply_token_reduction

model = apply_token_reduction(
    model,
    {
        "method": "ac_flowcut",
        "target_num": 64,
        "attention_coreset_protect_core": True,
        "attention_coreset_query_scope": "all",
    },
)
```

To remove the runtime patch and restore original inference behavior:

```python
from ac-flowcut import disable_token_reduction

model = disable_token_reduction(model)
```

## Package Layout

```text
AC-FlowCut/
├── ac-flowcut/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── registry.py
│   ├── clip_encoder.py
│   ├── modeling_clip_flowcut.py
│   ├── modeling_llama_flowcut.py
│   ├── llava_arch.py
│   ├── llava_llama_flowcut.py
│   ├── utils_flowcut.py
│   ├── trace.py
│   └── strategies/
│       ├── __init__.py
│       ├── base.py
│       ├── flowcut_strategy.py
│       └── none.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## License

This project is released under the Apache License 2.0. See `LICENSE` for the full text. LLaVA, Transformers, CLIP, and corresponding model checkpoints retain their original licenses.

## Acknowledgment

- This work is built upon [LLaVA](https://llava-vl.github.io/), [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), and [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA). We thank them for their excellent open-source contributions.
- We also thank [FlowCut](https://github.com/TungChintao/FlowCut), [FastV](https://github.com/pkunlp-icler/FastV), [SparseVLM](https://github.com/Gumpest/SparseVLMs), [VisionZip](https://github.com/dvlab-research/VisionZip), and others for their valuable research insights.
```