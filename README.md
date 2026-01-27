# AdaLoRA-QAT  
**Adaptive Low-Rank and Quantization-Aware Segmentation**

**Prantik Deb · Srimanth Dhondy · N. Ramakrishna · Anu Kapoor · Raju S. Bapi · Tapabrata Chakraborti**  
IIIT Hyderabad · NIMS Hyderabad · The Alan Turing Institute · UCL  

**IEEE ISBI 2026 (Accepted)**  

[🫁 Project Page](https://prantik-pdeb.github.io/adaloraqat.github.io/) · [📄 Paper] · [💾 Code] · [📦 Pretrained Models]

---


::contentReference[oaicite:0]{index=0}


## Abstract
We present **AdaLoRA-QAT**, a two-stage fine-tuning framework that integrates **adaptive low-rank parameterization** with **quantization-aware training** for efficient and reliable medical image segmentation.  
On chest X-ray lung segmentation, AdaLoRA-QAT achieves **95.6% Dice**, matching full-precision fine-tuning while reducing **trainable parameters by 16.6×** and achieving **2.24× model compression**, enabling deployment on resource-constrained clinical hardware without compromising anatomical fidelity.

---

## Highlights
- Two-stage **PEFT + QAT** framework for foundation models  
- Mixed-precision INT8 quantization with FP32 attention and AdaLoRA parameters  
- Clinically robust: no statistically significant degradation vs. full-precision SAM  

---

## Results
| Method | Dice (%) | Param Reduction |
|------|---------|----------------|
| SAM Decoder Fine-Tuning | 95.55 | 23.6× |
| **AdaLoRA-QAT (Ours)** | **95.59** | **16.6×** |

Wilcoxon signed-rank test confirms **no significant difference** from the full-precision baseline.

---

## Getting Started
```bash
conda create -n adaloraqat python=3.10
conda activate adaloraqat
pip install -r requirements.txt
