# 🚗 State Farm Distracted Driver Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-success)

## 📌 Project Overview
This repository contains a robust Deep Learning solution for the [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) Kaggle challenge. The goal is to accurately classify 10 distinct behaviors of drivers based on dashboard camera images, helping to prevent accidents caused by distracted driving.

## 🎯 Key Methodologies & Contributions

To build a highly accurate and generalizable model, this project implements several advanced techniques:

* **Hybrid Architecture:** Developed a hybrid **CNN-Transformer** model utilizing a **ConvNeXt** backbone combined with **Self-Attention** and **Attention Pooling** mechanisms to capture both local textures and global context effectively.
* **Overfitting Prevention:** Applied powerful data augmentation techniques, specifically **Mixup**, to improve model robustness against unseen data.
* **Zero Data Leakage Strategy:** Implemented a rigorous **Driver ID-based train-test split**. Instead of randomly splitting images (which leads to severe data leakage due to highly correlated frames of the same driver), the dataset is split based on unique subject IDs to ensure true generalization.

## 📊 Dataset Classes
The model is trained to classify the following 10 distracted driver behaviors:
- `c0`: Safe driving
- `c1`: Texting (right hand)
- `c2`: Talking on the phone (right hand)
- `c3`: Texting (left hand)
- `c4`: Talking on the phone (left hand)
- `c5`: Operating the radio
- `c6`: Drinking
- `c7`: Reaching behind
- `c8`: Hair and makeup
- `c9`: Talking to a passenger

## 🚀 Results & Performance
The hybrid model successfully mitigates overfitting and achieves state-of-the-art level performance on the validation set:
* **Accuracy:** `97.00%`
* **F1-Score:** `0.963`

## 📁 Repository Structure
* `state-farm-distracted-driver-detection (6).ipynb`: Main Jupyter Notebook containing the full pipeline (Data Preprocessing, Augmentation, Model Definition, Training Loop, and Evaluation).
* `coordinate.csv`: Auxiliary spatial/coordinate data used during preprocessing.

## 🛠️ How to Use
1. Clone this repository:
   ```bash
   git clone [https://github.com/LuuVanDuan/State-Farm-Distracted-Driver-Detection.git](https://github.com/LuuVanDuan/State-Farm-Distracted-Driver-Detection.git)
