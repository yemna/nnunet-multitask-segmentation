# Multi-task nnU-Net for Pancreas Cancer Segmentation and Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2-green.svg)](https://github.com/MIC-DKFZ/nnUNet)

A comprehensive multi-task deep learning framework for simultaneous pancreas cancer segmentation and subtype classification in 3D CT scans, built upon the nnU-Net v2 framework.

## 🎯 Project Overview

This project extends the nnU-Net v2 framework with a custom multi-scale classification head to perform:
- **Segmentation**: Normal pancreas (label 1) and pancreas lesions (label 2)
- **Classification**: Lesion subtype classification into 3 classes (0, 1, 2)

## 📋 Requirements

### Environment Setup
```bash
# Create conda environment
conda create -n nnunet_new python=3.11
conda activate nnunet_new

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install nnU-Net v2
pip install nnunetv2

# Additional dependencies
pip install nibabel numpy scipy scikit-learn matplotlib seaborn pandas
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU (tested on local GPU setup)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ for dataset and models

## 🚀 Quick Start

The entire development workflow was executed interactively within a Jupyter Notebook (`nnunet_with_classification.ipynb`). This approach was chosen due to persistent Colab limitations and provides better debugging capabilities.

### 1. Environment Variables Setup

```python
import os

# Set up nnU-Net directories (adjust paths as needed)
BASE = "D:/nnunet_with_classification/data"  # Your base directory
os.environ["nnUNet_raw"] = f"{BASE}/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = f"{BASE}/nnUNet_preprocessed"
os.environ["nnUNet_results"] = f"{BASE}/nnUNet_results"
```

### 2. Data Preparation

The dataset should be organized as:

```
data/
├── train/
│   ├── subtype0/
│   ├── subtype1/
│   └── subtype2/
├── validation/
│   ├── subtype0/
│   ├── subtype1/
│   └── subtype2/
└── test/
```

**Interactive Workflow (Recommended):**

Execute cells 3-5 in `nnunet_with_classification.ipynb` for:
- Automated dataset ingestion with subtype detection
- nnU-Net v2 format conversion (dataset.json, splits_final.json)
- Classification label mapping (classification_labels.csv)
- Label format fixes (float → integer conversion)

### 3. Training

**Interactive Training:**

Execute the training cells (6-7) in the notebook:

```python
# Preprocessing
!nnUNetv2_plan_and_preprocess -d 777 -c 3d_fullres --verify_dataset_integrity -pl nnUNetPlannerResEncM

# Training with custom trainer
!nnUNetv2_train 777 3d_fullres 0 -tr nnUNetTrainerWithClassification -p nnUNetResEncUNetMPlans --c
```

### 4. Inference

**Interactive Inference:**

Execute inference cells (8-12) in the notebook for both segmentation and classification:

```python
# Custom inference script with classification
!python scripts/inference.py \
  --model_dir "path/to/model" \
  --input_dir "path/to/validation" \
  --output_dir "path/to/predictions" \
  --fold 0 \
  --checkpoint checkpoint_best.pth \
  --num_classes 3 \
  --no-tta
```

## 📁 Repository Structure

```
nnunet-multitask-segmentation/
├── nnUNetTrainerWithClassification.py      # 🔥 Enhanced multi-task trainer
├── nnunet_with_classification.ipynb        # 📓 Main development workflow notebook
├── scripts/
│   ├── inference.py                        # 🎯 Custom inference with classification
└── README.md                               # 📖 This file
```

## 📊 Performance Results

### Dataset Distribution

| Split | Subtype 0 | Subtype 1 | Subtype 2 | Total |
|-------|-----------|-----------|-----------|-------|
| Train | 62 (24.6%) | 106 (42.1%) | 84 (33.3%) | 252 |
| Validation | 9 (25.0%) | 15 (41.7%) | 12 (33.3%) | 36 |

### Achieved Results

| Metric | Target (PhD) | Achieved | Status |
|--------|--------------|----------|--------|
| **Whole Pancreas DSC** | ≥0.91 | 0.8607 | 
