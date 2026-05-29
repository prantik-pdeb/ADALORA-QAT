# AdaLoRA-QAT: Efficient Medical Image Segmentation

**Adaptive Low-Rank Adaptation with Quantization-Aware Training for Medical Image Segmentation**

*Prantik Deb · Srimanth Dhondy · N. Ramakrishna · Anu Kapoor · Raju S. Bapi · Tapabrata Chakraborti*

IIIT Hyderabad · NIMS Hyderabad · The Alan Turing Institute · UCL

**IEEE ISBI 2026 (Accepted: Oral Presentation)**

[Project Page](https://prantik-pdeb.github.io/adaloraqat.github.io/)

[Hugging Face model page](https://huggingface.co/srimanth-d/ADALORA-QAT)

---

## Overview

AdaLoRA-QAT achieves **95.6% Dice score** on chest X-ray lung segmentation with:
- **2.24× smaller model** 
- **16.6× fewer trainable parameters** 

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

python training/sam_Ada_LoRA_QAT_two_stage.py
```

Training completes in ~4 hours on a single GPU.

---

## Results

| Metric | Score |
|--------|-------|
| **Dice** | 95.59% ± 0.04% |
| **IoU** | 91.58% ± 0.07% |
| **NSD** | 94.31% ± 0.05% |

**Model Efficiency:**
- Model Size: 2.24× compression
- Trainable Parameters: 16.6× reduction

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
python -u inference/inference.py \
--image_path sample_data/images/C19RD_COVID-29.png \
--checkpoint_path "best_model_stage2_int8.pth" \
--bbox 0 0 511 511 --save_mask --visualize \
--output_mask_path ./inf_res.png \
--save_overlay ./overlay
```
The above will run the inference script with the sample image present in the folder [sample_data](sample_data). The model weights could be downloaded from hugging face (https://huggingface.co/srimanth-d/ADALORA-QAT/resolve/main/best_model_stage2_int8.pth)

---

## Configuration

Key parameters in [`sam_Ada_LoRA_QAT_two_stage.py`](training/sam_Ada_LoRA_QAT_two_stage.py):
```python
config = {
    'data_dir': '/path/to/data',  
    'stage1': {
        'batch_size': 16,        
        'num_epochs': 25,
        'learning_rate': 5e-5,
    },
    'stage2': {
        'num_epochs': 10,
        'learning_rate': 5e-7,
    },
}

```
The above configuration is contains key hyperparameters that could be changed(if you want to **"experiment"**). If you are an advanced user and want to experiment more, you might refer the python file [sam_Ada_LoRA_QAT_two_stage.py](training/sam_Ada_LoRA_QAT_two_stage.py) and make changes to the **"configuration dict"**(present at the end of the code.).

---

## Citation

When using this model, please cite: Deb, Prantik, et al. "ADALORA-QAT: Adaptive Low Rank and Quantization Aware Segmentation." 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI). IEEE, 2026.

```
@inproceedings{deb2026adalora,
  title={ADALORA-QAT: Adaptive Low Rank and Quantization Aware Segmentation},
  author={Deb, Prantik and Dhondy, Srimanth and Ramakrishna, N and Kapoor, Anu and Bapi, Raju S and Chakraborti, Tapabrata},
  booktitle={2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2026},
  organization={IEEE}
}
```

---

## License

MIT License - see [LICENSE.md](./LICENSE.md) file

---

## Acknowledgments

- Meta AI for [SAM](https://github.com/facebookresearch/segment-anything)
- Xilinx for [Brevitas](https://github.com/Xilinx/brevitas)
- MONAI for [medical imaging tools](https://monai.io/)
