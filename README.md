[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fseungho-snu%2Ffxsr&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# FxSR
# Flexible Image Super-Resolution using Conditional Objective

Seung Ho Park, Young Soo Moon, and Nam Ik Cho

## Abstract
Recent studies have significantly enhanced the performance of single-image super-resolution (SR) using convolutional neural networks (CNNs). While there can be many high-resolution (HR) solutions for a given input, most existing CNN-based methods do not explore alternative solutions during the inference. A typical approach to obtaining alternative SR results is to train multiple SR models with different loss weightings and exploit the combination of these models. Instead of using multiple models, we present a more efficient method to train a single adjustable SR model on various combinations of losses by taking advantage of multi-task learning. Specifically, we optimize an SR model with a conditional objective during training, where the objective is a weighted sum of multiple perceptual losses at different feature levels. The weights vary according to given conditions, and the set of weights is defined as a style controller. Also, we present an architecture appropriate for this training scheme, which is the Residual-in-Residual Dense Block equipped with spatial feature transformation layers. At the inference phase, our trained model can generate locally different outputs conditioned on the style control map. Extensive experiments show that the proposed SR model produces various desirable reconstructions without artifacts and yields comparable quantitative performance to state-of-the-art SR methods.
<br><br>

# Usage:

## Environments
- Pytorch 1.10.0
- CUDA 11.5 & cuDNN 11.4
- Python 3.8


## Quick usage on your data:
you can choose any number [0, 1] for t.

    python test.py -opt options/test/test_FxSR_4x.yml -t 0.8
    
## Test models
Download the pretrained FxSR-PD 4x model from OneDrive <a href="https://1drv.ms/u/s!AqwRM35EFiRZgtQw4TFXp54wqcv4FQ?e=S3OzUL">Link</a> 

Download the pretrained FxSR-PD 8x model from OneDrive <a href="https://1drv.ms/u/s!AqwRM35EFiRZgtQu2ME4sOnJpQc2nA?e=Oe8E2J">Link</a> 

Download the pretrained FxSR-DS 4x model from OneDrive <a href="https://1drv.ms/u/s!AqwRM35EFiRZgtJRwIcAThRZt8R8ig?e=0iIRZo">Link</a> 

Download the pretrained FxSR-DS 8x model from OneDrive <a href="https://1drv.ms/u/s!AqwRM35EFiRZgtJPp3h34ypLN3EMeg?e=Gx3RPc">Link</a> 


<!--    
## Related Work

### Distortion Oriented Single Image Super-Resolution

#### [RRDB (ECCV 2018)] ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks <a href="https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf">Link</a> 

### Perception Oriented Single Image Super-Resolution

#### [SRGAN (CVPR 2017)] Photo-realistic Single Image Super-Resolution Using a Generative Adversarial Network <a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html">Link</a> 

#### [ESRGAN (ECCV 2018)] ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks <a href="https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf">Link</a> 

#### [SRFlow (ECCV 2020)] Recovering Realistic Texture in Image Super-Resolution by Deep Spatial Feature Transform <a href="http://de.arxiv.org/pdf/2006.14200?gitT">Link</a>
<br><br>
-->

## Brief Description of Our Proposed Method

### <u>TARGETED PERCEPTUAL LOSS</u>

<p align="center"><img src="figure/fig_VGGs.png" width="700"></p>

The effect of choosing different layers when estimating perceptual losses on different regions, e.g., on edge and texture regions, where the losses correspond to MSE, ReLU 2-2 (VGG22), and ReLU 4-4 (VGG44) of the VGG-19 network.


### <u>PROPOSED SR WITH FLEXIBLE STYLE</u>

<p align="center"><img src="figure/eqn_01.PNG" width="200"></p>

<p align="center"><img src="figure/eqn_02.PNG" width="500"></p>

### <u>PROPOSED NETWORK ARCHITECTURE</u>

<p align="center"><img src="figure/fig_architecture.png" width="700"></p>

The architecture of our proposed flexible SR network. We use the RRDB equipped with SFT as a basic block. The condition branch takes a style map for reconstruction style as input. This map is used to control the recovery styles of edges and textures for each region through SFT layers.

<p align="center"><img src="figure/fig05_c_RRDB-SFT.png" width="700"></p>

The proposed Basic Block (RRDB equipped with SFT layer)

## Experimental Results

### Visual Evaluation of Flexible SR for Perception-Distortion (FxSR-PD)

<p align="center"><img src="figure/FxSR-PD_4x.png" width="800"></p>

Changes in the result of FxSR-PS 4x SR according to t on DIV2K validation set.

<p align="center"><img src="figure/FxSR-PD_4x_t08.png" width="800"></p>

Visual comparison with state-of-the-art perception-driven SR methods on DIV2K validation set.

### Quantitative Evaluation of Flexible SR for Perception-Distortion (FxSR-PD)

<p align="center"><img src="figure/Quantitative Comparison FxSR-PD.png" width="800"></p>

### Visual Evaluation of Flexible SR for Diverse Styles (FxSR-DS)

<p align="center"><img src="figure/FxSR-DS_4x.png" width="900"></p>

Changes in the result of FxSR-DS 4x SR according to t on DIV2K validation set.

### Per-pixel Style Control
Comparison of the SR results of the conventional method and the FxSR-DS method
<p align="center"><img src="figure/fig_Local_Map_text_SR_v11.png" width="700"></p>
<p align="center"><img src="figure/fig_Local_Map_text_FxSR_v11.png" width="700"></p>
<!--
<p align="center"><img src="figure/fig_Local_Map_texture_SR_v11.png" width="700"></p>
<p align="center"><img src="figure/fig_Local_Map_texture_FxSR_v11.png" width="700"></p>

<p align="center"><img src="figure/fig_Local_Map_depth2_FxSR_v11.png" width="700"></p>
Depth-adaptive FxSR. T-maps is the modified version of the depth map of an image from the Make3D dataset.

<p align="center"><img src="figure/fig_Local_Map_depth_FxSR_v11.png" width="700"></p>
An example of applying a user-created depth map to enhance the perspective feeling with the sharper and richer textured foreground and the background with more reduced camera noise than the ground truth.
-->

We will add more local style control examples.

...

### Ablation Study

Convergence of diversity curve of the proposed FxSR-PD model as the number of training iteration increase
<!--
(a) 16 RBs with SFT, (b) using 23 RRDBs with SFT, (c) The performance comparison between two FxSR-PD version at the 250,000th iteration
-->
<p align="center"><img src="figure/fig_itr_diversity_all.png" width="1200"></p>

# NTIRE 2021 Learning the Super-Resolution Space Challenge <a href="https://github.com/andreas128/NTIRE21_Learning_SR_Space">Link</a> 

We joined the NTIRE 2021 challenge under the name of SSS.
FxSR-DS is the best in terms of LPIPS for both 4x and 8x, 8th in diversity score and 3rd in MOR (Mean Opinion Rank) <a href="https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Lugmayr_NTIRE_2021_Learning_the_Super-Resolution_Space_Challenge_CVPRW_2021_paper.pdf">Link.</a> 
<p align="center"><img src="figure/ntire_4x.png" width="600"></p>
<p align="center"><img src="figure/ntire_8x.png" width="600"></p>


# Acknowledgement
Our work and implementations are inspired by and based on BasicSR <a href="https://github.com/xinntao/BasicSR">[site]</a> 
