# AdaLoRA-QAT: Efficient Medical Image Segmentation

**Adaptive Low-Rank Adaptation with Quantization-Aware Training for Medical Image Segmentation**

*Prantik Deb · Srimanth Dhondy · N. Ramakrishna · Anu Kapoor · Raju S. Bapi · Tapabrata Chakraborti*

IIIT Hyderabad · NIMS Hyderabad · The Alan Turing Institute · UCL

**IEEE ISBI 2025 (Accepted)**

[Project Page](https://prantik-pdeb.github.io/adaloraqat.github.io/)

---

## Overview

AdaLoRA-QAT achieves **95.6% Dice score** on chest X-ray lung segmentation with:
- **2.24× smaller model**
- **16.6× fewer trainable parameters**
- **No accuracy loss** compared to full fine-tuning

Perfect for deploying medical AI on resource-constrained devices.

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/prantik-pdeb/adalora-qat.git
cd adalora-qat

# Create environment
conda create -n adaloraqat python=3.10 -y
conda activate adaloraqat

# Install dependencies
pip install torch torchvision transformers pillow numpy scipy monai brevitas tqdm
```

### Prepare Data

Organize your dataset:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Train
```bash
# Edit data_dir in sam_Ada_LoRA_QAT_two_stage.py
python sam_Ada_LoRA_QAT_two_stage.py
```

## Results

**AdaLoRA-QAT** 

**Key Metrics:**
- Dice: 95.59% ± 0.04%
- NSD: 94.31% ± 0.05%
- HD95: 8.21 ± 1.23 px

---

## How It Works

**Two-Stage Training:**

1. **Stage 1 (FP32)**: Hybrid training with AdaLoRA on encoder + full decoder fine-tuning
2. **Stage 2 (INT8)**: Full model quantization with QAT

**Architecture:**
- Vision Encoder: INT8 quantization + AdaLoRA (FP32) on attention
- Mask Decoder: INT8 quantization
- Prompt Encoder: INT8 quantization

---

## Usage

### Inference
```python
import torch
from transformers import SamModel, SamProcessor
from train_adalora_qat import AdaLoRA_Sam, apply_full_quantization_to_sam

# Load model
sam_base = SamModel.from_pretrained('facebook/sam-vit-base')
apply_full_quantization_to_sam(sam_base, bit_width=8, skip_qkv=True)
model = AdaLoRA_Sam(sam_base, max_rank=48, target_rank=32, alpha=32.0)

# Load weights
checkpoint = torch.load('checkpoints_stage2_int8_full/best_model_stage2_int8.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
image = Image.open('test.png')
inputs = processor(image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
outputs = model(**inputs)
mask = (torch.sigmoid(outputs.pred_masks) > 0.5).cpu().numpy()
```

---

## Configuration

Key parameters in `sam_Ada_LoRA_QAT_two_stage.py`:
```python
config = {
    'data_dir': '/path/to/data',  # Your dataset path
    'stage1': {
        'batch_size': 16,         # Adjust for GPU memory
        'num_epochs': 25,
        'learning_rate': 5e-5,
    },
    'stage2': {
        'num_epochs': 10,
        'learning_rate': 5e-7,
    },
}
```

**GPU Memory Issues?** Reduce `batch_size` to 8 and set `accumulation_steps: 2`.

---

## Citation
```bibtex
@inproceedings{deb2025adaloraqat,
  title={AdaLoRA-QAT: Adaptive Low-Rank Adaptation with Quantization-Aware Training for Medical Image Segmentation},
  author={Deb, Prantik and Dhondy, Srimanth and Ramakrishna, N. and Kapoor, Anu and Bapi, Raju S. and Chakraborti, Tapabrata},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2025}
}
```

---

## License

MIT License - see LICENSE file

---


---

## Acknowledgments
- Meta AI for [SAM](https://github.com/facebookresearch/segment-anything)
- Xilinx for [Brevitas](https://github.com/Xilinx/brevitas)
- MONAI for [medical imaging tools](https://monai.io/)
