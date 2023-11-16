# Super-Res-MRI-Project
## Abstract

In this project, we aim to enhance the quality and clarity of MRI brain scans using state-of-the-art super-resolution techniques. We propose two distinct approaches:

1. **Modified SRDiff Model with Diffusion:** We leverage a modified super-resolution model with a diffusion component, tailored for enhancing MRI images. This approach combines the "one-to-many" mapping capabilities of the SRDiff model with innovative regularization techniques from MR Image Denoising.

**Paper Reference:**  
- **Title**: SRDiff: Single image super-resolution with diffusion probabilistic models
  - **Authors**: Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, Yueting Chen
  - **Published in**: Neurocomputing, Volume 479, Pages 47-59
  - **DOI**: [10.1016/j.neucom.2022.01.029](https://doi.org/10.1016/j.neucom.2022.01.029)

3. **Modified SRCNN Model:** We introduce a deep convolutional neural network (CNN) with dense blocks for feature extraction and residual connections for high-frequency detail learning. This approach draws inspiration from "Brain MRI SR Using 3D Deep Densely Connected NN" and "Residual Dense Network for Image Super-Resolution."

Our study utilizes the BRaTS Brain Tumor Segmentation dataset from 2018 to 2020, with a specific focus on T1CE scans. We will evaluate the models using established metrics such as PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), LPIPS (Learned Perceptual Image Patch Similarity), LR_PSNR (Low-Resolution PSNR), and pixel standard deviation. These metrics will assess the improvements in MRI scan quality, emphasizing image clarity and detail enhancement. Our goal is to develop more effective and precise imaging methods, enhancing the reliability of diagnostic processes.

## Environment Installation
Using Anaconda to manage virtual environment
```bash
conda create --name SRM_env python=3.11
conda activate SRM_env
pip install -r requirements.txt
```
## Dataset and Preparation
BRaTS Brain Tumor Segmentation

- The BRaTS (Brain Tumor Segmentation) dataset is a widely recognized dataset in the medical imaging field, particularly for brain tumor analysis.
- It includes multi-institutional pre-operative MRI scans from 2018 to 2020, providing a diverse range of images.
- Focus on T1-weighted contrast-enhanced (T1CE) scans, known for their high detail and contrast, ideal for super-resolution tasks.
