# 🚀 Flow-QSM: Bridging Learned Priors and Physical Models for Quantitative Susceptibility Mapping

## 🔓 Overview

**Flow-QSM** is a physics-guided conditional **flow-matching framework** for fast, accurate, and robust Quantitative Susceptibility Mapping (QSM).

It bridges **learned generative priors** with **MRI physical models** to solve the ill-posed dipole inversion problem.

---

## ✨ Key Features

* 🧠 **Flow Matching Prior**
  Learns a continuous generative prior over susceptibility maps

* 🧲 **Physics-Guided Sampling**
  Incorporates dipole forward model during reverse inference

* 🧩 **Customized Architecture**

  * Patch-wise positional encoding (global consistency)
  * Dual-branch skip-backbone modulation (detail preservation)

* ⚡ **Efficient Inference**
  Fast reconstruction with high fidelity

---

## 📁 Project Structure

```bash
Flow-QSM/
│── train.py                  # Training entry
│── inference.py              # Inference entry
│── pipeline.py               # Sampling pipeline
│
├── modules/                   # Network architectures
├── schedulers/              # Flow matching scheduler
├── config/                     # model config 
├── scripts/                  # Training / inference scripts
│
├── output_dir/               # Checkpoints & logs
└── utils/                    # Utility functions
```

---

## ⚙️ Environment Setup

### 🔧 Requirements

* Python >= 3.10
* CUDA >= 11.8
* PyTorch == 2.3.1
* DeepSpeed == 0.16.4
* HuggingFace Accelerate

---

### 📦 Installation

```bash
conda create -n flowqsm python=3.10
conda activate flowqsm

# Install PyTorch (adjust CUDA version if needed)
pip install torch==2.3.1

# Core dependencies
pip install deepspeed==0.16.4
pip install accelerate

# Other dependencies
pip install -r requirements.txt
```

---

## 🚀 Training


### 🔵 Multi-GPU (Recommended)

```bash
./scripts/train.sh
```

---



## 🔍 Inference


### 🔵 Multi-GPU Inference

```bash
.scripts/inference.sh
```

---



## 📊 Outputs

* Reconstructed QSM maps (`.nii.gz`)
* Intermediate sampling results (optional)
* Logs & evaluation metrics

---


## 📦 Pretrained Weights

You can download the pretrained model weights from Baidu Netdisk:

- **File:** `mp_rank_00_model_states.pt`
- **Link:** https://pan.baidu.com/s/1SKrezLBM3WQue9-mKNtiQw
- **Extraction Code:** `u7ua`

**Installation:**
After downloading, place the weight file into your checkpoint directory:
```bash
./checkpoints/mp_rank_00_model_states.pt
```


## 📬 Contact

If you have any questions or would like to collaborate, feel free to contact:

📧 qinhaoming@stu.xmu.edu.cn