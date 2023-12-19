# Super-Res-MRI-Project
## Abstract

In this project, we aim to enhance the quality and clarity of MRI brain scans using state-of-the-art super-resolution techniques. We propose two distinct approaches:

1. **Modified SRDiff Model with Diffusion:** We leverage a modified super-resolution model with a diffusion component, tailored for enhancing MRI images. This approach combines the "one-to-many" mapping capabilities of the SRDiff model with innovative regularization techniques from MR Image Denoising.

**Paper Reference:**  
- **Title**: SRDiff: Single image super-resolution with diffusion probabilistic models
  - **Authors**: Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, Yueting Chen
  - **Published in**: Neurocomputing, Volume 479, Pages 47-59
  - **DOI**: [10.1016/j.neucom.2022.01.029](https://doi.org/10.1016/j.neucom.2022.01.029)

2. **Modified SRCNN Model:** We introduce a deep convolutional neural network (CNN) with dense blocks for feature extraction and residual connections for high-frequency detail learning. This approach draws inspiration from "Brain MRI SR Using 3D Deep Densely Connected NN" and "Residual Dense Network for Image Super-Resolution."

**Paper Reference:**  
- **Title**: Brain MRI super resolution using 3D deep densely connected neural networks
  - **Authors**: Yuhua Chen; Yibin Xie; Zhengwei Zhou; Feng Shi; Anthony G. Christodoulou; Debiao Li
  - **Published in**: 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)
  - **DOI**: 10.1109/ISBI.2018.8363679 (https://ieeexplore.ieee.org/abstract/document/8363679)
 
- **Title**: Residual Dense Network for Image Super-Resolution
  - **Authors**: Yulun Zhang; Yapeng Tian; Yu Kong; Bineng Zhong; Yun Fu
  - **Published in**: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
  - **DOI**: 10.1109/CVPR.2018.00262 (https://ieeexplore.ieee.org/document/8578360/authors#authors)

Our study utilizes the BRaTS Brain Tumor Segmentation dataset from 2018 to 2020, with a specific focus on T1CE scans. We will evaluate the models using established metrics such as PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), LPIPS (Learned Perceptual Image Patch Similarity), LR_PSNR (Low-Resolution PSNR), and pixel standard deviation. These metrics will assess the improvements in MRI scan quality, emphasizing image clarity and detail enhancement. Our goal is to develop more effective and precise imaging methods, enhancing the reliability of diagnostic processes.

Dataset can be found here: https://drive.google.com/drive/folders/104mZkiHqriF2tR5tVn4aLOA094xbWS-4?usp=share_link

**Links to Colab**

RDN-UNet Approach: https://colab.research.google.com/drive/1o5t5duzKJUf1aJvEidCj67aGGz8-ym4t?usp=sharing

SRCNN Approach: https://colab.research.google.com/drive/1-DrOp5qtKGkD1A8r6vcki46Oh3Ue2Vjk#scrollTo=F146M3JBTHp_&uniqifier=1
