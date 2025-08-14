# nnunet-multitask-segmentation
Multi-task deep learning for pancreas cancer segmentation and classification using nnU-Net v2

```markdown
# Multi-task nnU-Net for Pancreas Cancer Segmentation and Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2-green.svg)](https://github.com/MIC-DKFZ/nnUNet)

A comprehensive multi-task deep learning framework for simultaneous pancreas cancer segmentation and subtype classification in 3D CT scans, built upon the nnU-Net v2 framework.

## 🎯 Project Overview

This project extends the nnU-Net v2 framework with a custom multi-scale classification head to perform:
- **Segmentation**: Normal pancreas (label 1) and pancreas lesions (label 2)
- **Classification**: Lesion subtype classification into 3 classes (0, 1, 2)

### Key Features
- 🔥 **Multi-scale classification head** with attention mechanism
- 🎯 **Class imbalance handling** through weighted loss functions with progressive training
- 🚀 **Early stopping** for efficient training (100 epochs patience)
- 🔧 **Comprehensive debugging tools** with ClassificationDebugger for training optimization
- 📊 **Advanced metrics tracking** for both tasks
- ⚡ **Inference speed optimization** (10%+ improvement over baseline)
- 🎓 **Progressive classification training** with warmup epochs
- 🔍 **Real-time training analysis** and class prediction monitoring

## 🗃️ Architecture

```

Input CT → Shared Encoder → ┌─ Segmentation Decoder → Segmentation Masks
└─ Multi-Scale Classification Head → Subtype Prediction
↑
(Last 3 encoder stages with attention)

````

The implementation features:
- **Shared nnU-Net ResEncM encoder** for feature extraction
- **Original nnU-Net decoder** for segmentation
- **Custom MultiScaleClassificationHead** with attention-based feature fusion from last 3 encoder stages
- **Progressive training strategy** with classification warmup and boosting

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
pip install nibabel numpy scipy scikit-learn matplotlib seaborn
````

### Hardware Requirements

  - **GPU**: CUDA-compatible GPU (tested on local GPU setup)
  - **Memory**: 16GB+ RAM recommended
  - **Storage**: 10GB+ for dataset and models

## 🚀 Quick Start

The entire development workflow, including data preparation, training, and inference, was executed interactively within a Jupyter Notebook. The standalone `.py` scripts are provided for convenience or for users who prefer a command-line approach.

### 1\. Environment Variables Setup

```python
import os

# Set up nnU-Net directories (adjust paths as needed)
BASE = "D:/nnunet_with_classification/data"  # Your base directory
os.environ["nnUNet_raw"] = f"{BASE}/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = f"{BASE}/nnUNet_preprocessed"
os.environ["nnUNet_results"] = f"{BASE}/nnUNet_results"
```

### 2\. Data Preparation

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

**Recommended Approach: Using Jupyter Notebook**

```python
# The complete data preparation workflow is demonstrated in the
# nnunet_with_classification.ipynb notebook. Cells 3-5 handle:
# - Automated dataset ingestion with subtype detection
# - nnU-Net v2 format conversion (dataset.json, splits_final.json)
# - Classification label mapping (classification_labels.csv)
# - Label format fixes (float → integer conversion)
```

**Alternative: Using Standalone Script**

```bash
python data_preparation.py --input_dir /path/to/your/data
```

### 3\. Training

**Recommended Approach: Using Jupyter Notebook**

```bash
# Execute training directly from the Jupyter Notebook cells after
# planning and preprocessing. This mirrors the CLI commands:
nnUNetv2_plan_and_preprocess -d 777 -c 3d_fullres --verify_dataset_integrity -pl nnUNetPlannerResEncM
nnUNetv2_train 777 3d_fullres 0 -tr NNUNet_tuned -p nnUNetResEncUNetMPlans
```

### 4\. Inference

**Recommended Approach: Using Jupyter Notebook**
Inference was performed interactively within the notebook. The commands below reflect the CLI calls used for segmentation and the custom script used for classification.

  - **Segmentation Inference (CLI)**: The standard nnU-Net v2 CLI command was used for segmentation. A standalone `.py` script is provided for convenience.

    ```bash
    nnUNetv2_predict -i /path/to/test -o /path/to/results -d 777 -c 3d_fullres
    ```

  - **Classification Inference (Custom Script)**: Since `nnUnetv2` lacks a native CLI for classification, a custom script was created and run within the notebook cells. This script is also provided as a standalone file.

    ```bash
    python inference_classification.py --input_dir /path/to/test --output_file subtype_results.csv
    ```

-----

## 📁 Repository Structure

```
nnunet-multitask-segmentation/
├── NNUNet_tuned.py                           # 🔥 Enhanced multi-task trainer
├── default_nnunetv2_inference_optimize.py   # 🚀 Inference optimization script
├── nnunet_with_classification.ipynb         # 📓 Main development workflow notebook
├── data_preparation.py                      # 📁 Standalone data formatting script
├── train.py                                 # 🚀 Standalone training script wrapper
├── inference_segmentation.py                # 🎯 Standalone segmentation inference script
├── inference_classification.py              # 🎯 Standalone classification inference
├── evaluation.py                            # 📊 Standalone validation metrics computation
├── requirements.txt                         # 📦 Python dependencies
└── README.md                                # 📖 This file
```

-----

## 🔧 Implementation Details

### Custom Trainer (NNUNet\_tuned.py)

The core implementation extends `nnUNetTrainer` with:

  - **MultiScaleClassificationHead**: Processes features from last 3 encoder scales with attention mechanism
  - **Progressive training**: Classification warmup (15 epochs) with gradual boost scaling (1.0 → 3.0)
  - **Advanced class balancing**: Weighted loss functions based on combined train+validation distribution
  - **Early stopping**: 100 epochs patience with 1e-3 minimum delta
  - **ClassificationDebugger**: Real-time analysis of class predictions and logit distributions
  - **Enhanced optimization**: 4x learning rate for classifier, gradient clipping (value 12)

### Key Technical Features

#### Multi-Scale Feature Fusion

```python
# Captures features from last 3 encoder stages
self.num_scales = 3
# Attention-weighted feature aggregation with spatial reduction
attention_weights = self.attention(concat_features)
weighted_features = (stacked_features * attention_weights).sum(dim=1)
```

#### Progressive Classification Training

```python
# Classification warmup with gradual boost
if self.current_epoch < self.cls_warmup_epochs:  # 15 epochs
    progress = self.current_epoch / self.cls_warmup_epochs
    cls_boost = 1.0 + 2.0 * progress  # Scales from 1.0 to 3.0
else:
    cls_boost = 3.0  # Full boost after warmup
    
total_loss = seg_loss + (self.cls_loss_weight * cls_boost) * cls_loss
```

#### Advanced Class Imbalance Handling

```python
# Inverse frequency weighting based on combined dataset distribution
# Class 0: 71 samples, Class 1: 121 samples, Class 2: 96 samples (total: 288)
class_weights = torch.tensor([
    total_samples / (2.5 * 71),  # 1.63 for class 0
    total_samples / (4 * 121),   # 0.60 for class 1
    total_samples / (2.8 * 96)   # 1.07 for class 2
])
```

#### Real-time Debugging and Analysis

```python
# ClassificationDebugger tracks:
# - Per-class prediction distributions
# - Logit imbalances and bias detection
# - Never-predicted classes identification
# - Per-epoch accuracy analysis
```

#### Training Efficiency Optimizations

  - **Early stopping**: 100 epochs patience (vs no stopping in baseline)
  - **Max epochs**: 500 (vs 1000 default)
  - **Gradient clipping**: Value 12 for stability in multi-task training
  - **Enhanced learning rates**: 4x base rate for classification head
  - **Progressive training**: Warmup + boosting strategy

## 📊 Performance Results

### Dataset Distribution

| Split | Subtype 0 | Subtype 1 | Subtype 2 | Total |
|-------|-----------|-----------|-----------|-------|
| Train | 62 (24.6%) | 106 (42.1%) | 84 (33.3%) | 252 |
| Validation | 9 (25.0%) | 15 (41.7%) | 12 (33.3%) | 36 |
| **Combined** | **71 (24.7%)** | **121 (42.0%)** | **96 (33.3%)** | **288** |

*Note: Trainer uses combined statistics for class weight calculation*

### Target Performance

| Level | Whole Pancreas DSC | Lesion DSC | Macro F1 |
|-------|-------------------|------------|----------|
| Undergraduate | ≥0.85 | ≥0.27 | ≥0.6 |
| Master/PhD | ≥0.91 | ≥0.31 | ≥0.7 |

### Achieved Results

| Metric | Achieved Score | Target (PhD) | Status |
|--------|----------------|--------------|---------|
| **Whole Pancreas DSC** | **0.9183** | ≥0.91 | ✅ **Above expectations** |
| **Lesion DSC** | **0.6443** | ≥0.31 | ✅ **Above expectations** |
| **Classification Macro F1** | **0.1961** | ≥0.70 |  **Expectations not met** |
| **Inference Speed Improvement** | `10%+` | ≥10% | ✅ **Above expectations** |

**Training Configuration:**

  - **Max Epochs**: 500 with early stopping (100 epochs patience)
  - **Training Strategy**: Progressive multi-scale classification with 15-epoch warmup
  - **Classification Loss Weight**: 2.5 with progressive boosting (1.0 → 3.0)
  - **Optimizer**: Enhanced with 4x learning rate for classification head
  - **Gradient Clipping**: Value 12 for stability in multi-task training

**Training Insights:**

  - **Segmentation Excellence**: Multi-scale encoder features significantly improve segmentation performance
  - **Classification Challenge**: Despite advanced debugging and progressive training, model exhibits majority class bias
  - **Technical Success**: Framework provides comprehensive debugging tools revealing training dynamics
  - **Architecture Strength**: Attention-based multi-scale fusion effectively captures hierarchical features

**ClassificationDebugger Findings:**

  - Real-time monitoring revealed persistent class imbalance despite weighted losses
  - Logit distribution analysis showed bias toward majority class (Class 1)
  - Never-predicted class detection helped identify training issues early

## ⚡ Inference Speed Optimization

For Master/PhD candidates, implemented optimization strategies achieving a **10%+ speed improvement** over the baseline model.

### Key Optimization Strategies:

  - **FP16 Inference**: Utilized `torch.cuda.amp.autocast()` to reduce memory bandwidth and accelerate computations.
  - **Optimized Post-processing**: Implemented vectorized operations for faster post-processing steps.
  - **Multi-scale attention optimization** to reduce computational overhead.
  - **Efficient Feature Extraction**: Optimized the use of encoder hooks for faster feature capture during inference.

### Example Optimized Inference Code

```python
# Example optimized inference
with torch.cuda.amp.autocast():
    self._enc_features = []  # Clear previous features
    seg_logits = self.network(data.half())
    cls_logits = self._classification_forward()
```

## 🛠️ Development Setup

### Local Development (Used Due to Colab Limitations)

This implementation was developed locally due to persistent Colab issues:

  - **Colab T4 GPU Issues**: Continuous timeouts and connection failures during long training sessions
  - **Session Interruptions**: Colab kept running out of time and failing to reconnect to T4 GPU
  - **Training Continuity**: Local setup ensures uninterrupted training for the full 500 epochs with early stopping
  - **Persistent Storage**: Models and debugging logs remain accessible between sessions
  - **Stable GPU Access**: Consistent GPU availability without reconnection issues

### Notebook Workflow (Development Approach)

The included `nnunet_with_classification.ipynb` was used for the complete development workflow and demonstrates:

**Environment Setup (Cells 1-2):**

  - PyTorch and nnU-Net v2 environment verification
  - CUDA availability checking
  - nnU-Net directory structure creation

**Data Preparation (Cells 3-5):**

  - Automated dataset organization from source folders
  - `dataset.json` creation with nnU-Net v2 format
  - `splits_final.json` generation preserving train/validation split
  - `classification_labels.csv` creation with subtype mapping
  - Label format fixes (floating point → integer conversion)

**Training and Preprocessing (Cells 6-7):**

  - nnU-Net preprocessing pipeline execution
  - Custom trainer (NNUNet\_tuned) training with multi-scale classification

**Inference and Evaluation (Cells 8-12):**

  - Segmentation inference on validation and test data
  - Multi-scale classification inference with FP16/FP32 compatibility fixes
  - Comprehensive evaluation with DSC and F1 score calculations
  - Real-time results analysis and performance reporting

**Key Implementation Details:**

  - **Local GPU Setup**: Used due to persistent Colab T4 timeouts and connection failures
  - **Notebook-Based Development**: All core functionality implemented in cells for interactive debugging
  - **Multi-scale Feature Hooks**: Real-time encoder feature capture during inference
  - **FP16 Compatibility**: Automatic dtype conversion for mixed-precision compatibility

**Note**: The development was conducted entirely within Jupyter notebook cells due to Colab limitations. Standalone scripts are provided as alternatives for users who prefer command-line execution.

-----

## 🛠 Troubleshooting

### Common Issues

#### 1\. Label Format Issues

```python
# Fix non-integer labels in masks
arr = np.rint(arr)           # round to nearest integer
arr = np.clip(arr, 0, 2)     # enforce label set {0,1,2}
arr = arr.astype(np.uint8)   # ensure proper dtype
```

#### 2\. Environment Variables

Ensure nnU-Net paths are set before importing:

```python
os.environ["nnUNet_raw"] = "/path/to/raw"
os.environ["nnUNet_preprocessed"] = "/path/to/preprocessed"
os.environ["nnUNet_results"] = "/path/to/results"
```

#### 3\. CUDA Memory Issues

  - Reduce batch size in plans file
  - Use gradient checkpointing
  - Enable FP16 training with autocast

#### 4\. Classification Training Issues

```python
# Monitor ClassificationDebugger output for:
# - Never predicted classes
# - Large logit imbalances
# - Class distribution mismatches
# Adjust cls_loss_weight and warmup_epochs accordingly
```

#### 5\. Feature Hook Issues

```python
# Ensure hooks are properly set up
self._setup_feature_hooks()
# Clear features before each forward pass
self._enc_features = []
```

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@misc{nnunet_multitask_2025,
  title={Multi-task nnU-Net for Pancreas Cancer Segmentation and Classification},
  author={[Your Name]},
  year={2025},
  url={[https://github.com/yourusername/nnunet-multitask-segmentation](https://github.com/yourusername/nnunet-multitask-segmentation)}
}
```

## 📄 License

This project is built upon nnU-Net v2. Please refer to the original nnU-Net license for usage terms.

## 🤝 Contributing

This repository represents a technical assessment implementation. For questions or improvements, please open an issue.

## 🔗 References

  - [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z)
  - [Large-scale pancreatic cancer detection via non-contrast CT and deep learning](https://doi.org/10.1038/s41591-023-02640-w)
  - [Metrics reloaded: recommendations for image analysis validation](https://doi.org/10.1038/s41592-023-02151-z)

-----

**Technical Assessment**: Deep Learning for Medical Imaging
**Framework**: nnU-Net v2 with Multi-Scale Classification Extension
**Task**: Multi-task Pancreas Cancer Segmentation and Classification

```
```
