# Feature Matching with Transformers and Camera Localization  

## Overview  
This project explores the use of **LoFTR (Detector-Free Local Feature Mapping with Transformers)** for feature matching across multiple images. Additionally, a lightweight **Neural Network** is implemented for **3D reconstruction** and **camera localization**, leveraging LoFTR-based matches as ground truth. The approach significantly improves accuracy over traditional methods like SIFT.  

## Features  
- **Feature Matching with Transformers:** Utilizes LoFTR for accurate keypoint correspondences without traditional feature detectors.  
- **Neural Network for 3D Reconstruction:** A lightweight model estimates 3D structure and camera poses.  
- **Improved Localization Accuracy:** Achieves **90% loss reduction** compared to SIFT-based matches when used with the same neural network.  

## Technologies Used  
- **Deep Learning:** Transformers, LoFTR, Neural Networks  
- **Computer Vision:** Feature Matching, 3D Reconstruction  
- **Camera Localization:** Pose Estimation, Reconstruction Error Optimization  
