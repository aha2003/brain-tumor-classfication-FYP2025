# Brain Tumor Classification using Deep Learning Approaches

An empirical comparative study of deep learning architectures for multi-class brain tumor classification from MRI scans. This repository contains the implementation pipelines developed for a BSc (Hons) Computer Science (Artificial Intelligence) dissertation at Heriot-Watt University.

## Project Overview
Brain tumor analysis from MRI scans is a critical task in medical imaging, but manual interpretation is often time-consuming and subject to inter-observer variability. This project evaluates multiple deep learning architectures to analyze trade-offs between convolution-based feature extraction and transformer-based global attention mechanisms on a fixed MRI dataset.

### Dataset Specifications
*   **Source:** Brain Tumor Classification (MRI) Dataset (Bhuvaji et al.)
*   **Total Samples:** 3,264 T1-weighted contrast-enhanced MRI slices
*   **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor
*   **Input Resolution:** 224 × 224 pixels (standardized preprocessing)
*   **Data Splits:** 76% Training (2,485 images), 14% Validation (450 images), 10% Testing (329 images) initialized via a constant random seed (123).

---

## Models Evaluated

*   **Custom Baseline CNN**
    Lightweight 3-layer convolutional network used as a performance baseline.
*   **VGG-16**
    Deep convolutional network with high parameter capacity for hierarchical feature extraction.
*   **ResNet-50**
    Residual network with skip connections to improve gradient flow and enable deeper training stability.
*   **EfficientNet-B0**
    Compact architecture using compound scaling and MBConv blocks for parameter-efficient learning.
*   **DeiT (Vision Transformer)**
    Transformer-based encoder using multi-head self-attention to model global spatial relationships.

---

## Hyperparameters & Reproducibility
To ensure an objective architectural evaluation, global training hyperparameters were standardized across all candidate models:

*   **Framework Environment:** Hybrid deep learning stack using TensorFlow/Keras for CNN configurations and PyTorch (via the `timm` library) for the DeiT Transformer pipeline.
*   **Hardware Compute:** Trained on GPU runtimes utilizing cloud-provisioned NVIDIA T4 accelerators.
*   **Optimization Engine:** Adam Optimizer.
*   **Loss Function:** Categorical Cross-Entropy.
*   **Batch Sizing:** Standardized to 16 for CNN architectures; throttled to 4–8 for DeiT-ViT variants due to multi-head self-attention memory limits.
*   **Regularization & Controls:** Global Dropout rate of 0.5. Regularization controlled via Early Stopping targeting validation loss optimization across a 5-epoch patience threshold.
*   **Cross-Validation:** Performance profiles validated via 5-fold cross-validation ($k=5$) mechanisms.

### Two-Phase Transfer Learning Blueprint
Pre-trained models utilized ImageNet initial weights adapted via a strict two-stage process:
1.  **Phase 1 (Feature Extraction):** Pre-trained backbone fully frozen; custom classifier heads trained at an initial learning rate of $1.0 \times 10^{-4}$.
2.  **Phase 2 (Fine-Tuning):** Selected deep layers unfrozen (e.g., Block 4/5 for VGG-16; top transformer layers for DeiT) and Retrained at a decoupled learning rate of $1.0 \times 10^{-5}$ to counter catastrophic forgetting.

---

## Experimental Results

### Performance Summary

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Generalization Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeiT (Transformer)** | **77.7%** | **0.841** | **0.771** | **0.739** | **4.0%** |
| **VGG-16** | 75.9% | 0.832 | 0.751 | 0.728 | 4.0% |
| **ResNet-50** | 73.6% | 0.819 | 0.729 | 0.711 | 3.0% |
| **Custom CNN** | 67.5% | 0.771 | 0.655 | 0.627 | 11.0% |
| **EfficientNet-B0** | 66.0% | 0.702 | 0.664 | 0.637 | 4.0% |

---

## Key Findings

*   Transformer-based models achieved the highest overall classification accuracy in this setup, suggesting strong performance in capturing global spatial relationships in MRI scans.
*   Glioma classification remained consistently challenging across all architectures, indicating limitations of pure classification models for infiltrative tumor structures.
*   Transfer learning significantly improved stability and reduced overfitting compared to training from scratch.
*   Lightweight architectures (e.g., EfficientNet-B0) showed competitive efficiency but lower overall predictive performance in this dataset configuration.

---

## Repository Structure
The project is organized into separate Jupyter notebooks for each model:

```text
├── Baseline_CNN_MRI.ipynb
├── Deit_ViT.ipynb
├── EfficientNet.ipynb
├── ResNet_50.ipynb
└── VGG_16.ipynb
```

### Optional Colab Execution (DeiT)
For transformer-based training you may encounter an error loading the file from github, a Colab runtime is available:
[Execute Live on Google Colab](https://colab.research.google.com/drive/1_PUsrdvCOH-39jdhzfa8brdw1KxnaiHz?usp=sharing)

---

## Installation

### Requirements
```bash
pip install tensorflow torch torchvision timm scikit-learn numpy pandas matplotlib seaborn
```

### Quick Start
```bash
git clone https://github.com/aha2003/brain-tumor-classfication-FYP2025.git
```
Open any notebook in JupyterLab, Jupyter Notebook, or VSCode and run sequentially.

---

## 📝 Reference
```text
Aboushady, A. (2026). Brain Tumor Classification using Deep Learning Approaches.
School of Mathematical and Computer Sciences, Heriot-Watt University.
```
