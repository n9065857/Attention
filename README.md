
 *Title
 
  Improving robustness in X-ray image classification through attention mechanisms in convolutional neural networks
  
 *Description
 
This project implements a deep learning framework in MATLAB 2024a that integrates attention mechanisms into convolutional neural networks (CNNs) for the classification of musculoskeletal radiographs (MURA dataset) and frontal-view chest X-rays. The model addresses noise, interpretability, and generalisation challenges by incorporating hierarchical feature fusion, robust feature selection (SIFT), and ensemble learning techniques (KNN, Decision Tree, and SVM).

* Dataset Information
  
1. MURA Dataset
Source: Stanford ML Group

Link: https://stanfordmlgroup.github.io/competitions/mura/

Reference: Rajpurkar et al., 2017 (https://arxiv.org/abs/1712.06957)

Description: 40,561 grayscale X-ray images from 14,863 studies, annotated as normal or abnormal.

2. Chest X-ray Pneumonia Dataset
Source: Kaggle

Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Description: Frontal chest radiographs labelled with pneumonia, tuberculosis, COVID-19, or normal conditions.

 * Code Information
 * 
Written in MATLAB 2024a

* Implements:

CNN backbones: Xception and InceptionResNetV2

Attention block modules

Feature fusion and SIFT-based feature selection

Majority voting ensemble using KNN, Decision Tree, and SVM

Explainability tools: Grad-CAM, Occlusion Sensitivity, and t-SNE

* Usage Instructions
1. Preparation
Download and extract the MURA and Chest X-ray datasets.

Follow the dataset folder structures outlined in the manuscript or use the provided script, organize_dataset.

2. Running the Code
Run main_training_script.m to start training with self-supervised pretraining and attention-enabled CNNs.

Use feature_fusion_and_selection.m for feature extraction and SIFT-based selection.

Run train_classifiers.m for training KNN, Tree, and SVM classifiers.

Use evaluate_ensemble.m to run majority voting and generate performance metrics.

Visualise with visualize_attention_maps.m and plot_tsne.m.

3. Output
Accuracy, precision, recall, F1, and Kappa for each task.

Confusion matrices and attention-based heatmaps.

 * Requirements
 * 
MATLAB 2024a ,GPU

Toolboxes:

Deep Learning Toolbox

Statistics and Machine Learning Toolbox

Image Processing Toolbox

 * Methodology
   
Pretraining: Self-supervised learning using in-domain grayscale X-rays.

Feature extraction: Custom attention modules with CNNs.

Fusion: Feature concatenation across models.

Selection: SIFT-based keypoint selection.

Ensemble: Majority vote over KNN, SVM, and Tree classifiers.

Evaluation: Accuracy, F1-score, Cohen’s Kappa, t-SNE clustering, and Grad-CAM for visual interpretability.

Citations
Rajpurkar et al. (2017). MURA Dataset: https://arxiv.org/abs/1712.06957

Chest X-ray Kaggle Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

 * License & Contribution Guidelines
 * 
This code is made available for academic and non-commercial use only. For licensing and contribution, please contact the corresponding author. If you use this code, please cite the corresponding publication
