# Super-Res-MRI-Project
## Abstract

In this project, we aim to enhance the quality and clarity of MRI brain scans using state-of-the-art super-resolution techniques. We propose two distinct approaches:

1. **Modified SRDiff Model with Diffusion:** We leverage a modified super-resolution model with a diffusion component, tailored for enhancing MRI images. This approach combines the "one-to-many" mapping capabilities of the SRDiff model with innovative regularization techniques from MR Image Denoising.

2. **Modified SRCNN Model:** We introduce a deep convolutional neural network (CNN) with dense blocks for feature extraction and residual connections for high-frequency detail learning. This approach draws inspiration from "Brain MRI SR Using 3D Deep Densely Connected NN" and "Residual Dense Network for Image Super-Resolution."

Our study utilizes the BRaTS Brain Tumor Segmentation dataset from 2018 to 2020, with a specific focus on T1CE scans. We will evaluate the models using established metrics such as PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), LPIPS (Learned Perceptual Image Patch Similarity), LR_PSNR (Low-Resolution PSNR), and pixel standard deviation. These metrics will assess the improvements in MRI scan quality, emphasizing image clarity and detail enhancement. Our goal is to develop more effective and precise imaging methods, enhancing the reliability of diagnostic processes.
