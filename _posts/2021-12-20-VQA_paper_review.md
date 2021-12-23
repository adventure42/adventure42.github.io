---
layout: post                          # (require) default post layout
title: "VQA paper review"                   # (require) a string title
date: 2021-12-20       # (require) a post date
categories: [VQA]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [VQA]                      # (custom) tags only for meta `property="article:tag"`


---

<br>

# Video Quality Assessment paper review

<br>

1. Quality Assessment of In-the-Wild Videos by Dingquan Li (2019)
2. KonVid-150k: A Dataset for No-Reference Video Quality Assessment of Videos in-the-Wild by Franz Götz-Hahn (Mar 2021) 
3. UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content by Zhengzhong Tu (Apr 2021)
4. Study on the Assessment of the Quality of  Experience of Streaming Video by Aleksandr Ivchenko(2020)

<br>

<br>

## Quality Assessment of In-the-Wild Videos

새로운 video quality assessment model을 제시함. human visual system (HVS)를 기반의 content-dependency와 temporal-memory effect 요소가 video quality assessment에 끼치는 영향을 고려하여 deep learning model을 설계함. 

- content dependency - pre-trained image classification neural network (inherent content-aware property를 가진)
- temporal-memory effect - temporal hysteresis와 같은 요소가 포함된 long-term dependencies를 network에 GRU(gates recurrent unit) & subjectively-inspired temporal pooling layer와 함께 integrate했음. 

저자의 model을 주요 video databses (: KoNViD-1k, CVD2014, and LIVE-Qualcomm)로 훈련시켜서 assessment 결과를 SROCC, KROCC, PLCC, RMSE 값으로 평가함.

다른 주요 video quality assessment model(BRISQUE, NIQE, CORNIA, VIIDEO, VBLIINDS)과 함께 평가 결과를 비교하여 저자의 model의 월등히 우수하다는 것을 확인함.

<br>

**Pytorch를 통해 model을 구현했고 source code는 [https://github.com/lidq92/VSFA](https://github.com/lidq92/VSFA)**

<br>

computational efficiency - 저자의 실험 환경= Intel Core i7-6700K CPU@4.00 GHz, 12G NVIDIA TITAN Xp GPU and 64 GB RAM in Ubuntu 14.04

<br>

**No Reference Image and Video Quality Assessment 기법들** (설명: [https://live.ece.utexas.edu/research/Quality/nrqa.htm](https://live.ece.utexas.edu/research/Quality/nrqa.htm))

- **BRISQUE**(blind/referenceless image spatial quality evaluator) - use scene statistics of locally normalized luminance coefficients to quantify possible losses of "naturalness" in the image due to the presnce of distortions, leading to a holistic measure of quality

  관련 논문: https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

- **NIQE**(Natural Image Quality Evaluator) - completely blind image quality analyzer that only makes use of measurable deviations from statistical regularities observed in natural images, without training on human-rated distorted images, and, indeed without any exposure to distorted images.

- **CORNIA**(Codebook Representation for No-Reference Image Assessment) - 

  unsupervised feature learning 방식으로 human perceived image quality 를 예측하는 model. computationally efficient- how? used soft-assignment coding with max pooling to obtain effective image representations for quality estimation. 더 상세한 내용은 아래 논문 읽어봐야함.

  관련 논문: https://ieeexplore.ieee.org/document/6247789

- **VIIDEO**(video intrinsic integrity and distortion evaluation oracle) - embodies models of intrinsic statistical regularities that are observed in natural vidoes, which are used to quantify disturbances introduced due to distortions. An algorithm derived from the VIIDEO model is thereby able to predict the quality of distorted videos without any external knowledge about the pristine source, anticipated distortions, or human judgments of video quality.

  관련 논문: https://live.ece.utexas.edu/publications/2016/07332944.pdf

- **VBLIINDS**(Video BLIINDS) - non-distortion specific 방식이다. This approach relies on a spatio-temporal model of video scenes in the discrete cosine transform (DCT) domain, and on a model that characterizes the type of motion occurring in the scenes, to predict video quality. The video quality assessment (VQA) algorithm does not require the presence of a pristine video to compare against in order to predict a quality score. The contributions of this work are three-fold.

- **BLIINDS** (BLind Image Integrity Notator using DCT-Statistics) - 

- **DIIVINE** (Distortion Identification-based Image Verity and INtegrity Evalutation)

- 연구된 타 model: (2019) https://link.springer.com/content/pdf/10.1007/s11760-019-01510-8.pdf 그 외 매우 다양한 새로운 모델들이 개발되고 있다. 

<br>

<br>

## A Dataset for No-Reference Video Quality Assessment of Videos in-the-Wild

왜 Konstanz Natural Video Quality Database? 

Konstanz Natural Video Quality Database (KoNViD-1k) is the only publicly available database that contains sequences with authentic distortions.

논문의 저자는 KonVid-150k라는 새로운 dataset을 만들었다. this dataset consists of coarsely annotated set of 153,841 videos (각 five quality ratings가진) and 1,596 videos (각 최소 89 ratings가진)

저자는 새로운 VQA 방식을 제안한다 - MLSP-VQA relying on multi-level spatially pooled deep features(MLSP) 이 방식은 기존 deep transfer learning 방식보다 큰 scale에서 훈련을 진행하는데에 더 특화되어있다.  저자가 제안하는 방식 중에 MLSP-VQA-FF가 KoNViD-1k dataset으로 평가했을때에 0.82수준의 가장 좋은 SRCC(Spearman rank-order correlation coefficient)를 보여주었다. 

MLSP-VQA models trained on KonVid-150k sets the new state-of-the-art for cross-test performance on KoNViD-1k, LIVEVQC, and LIVE-Qualcomm with a 0.83, 0.75, and 0.64 SRCC, respectively.

<br>

<br>

## Benchmarking Blind Video Quality Assessment for User Generated Content

UGC(User-Generated-Content)가 폭발적으로 많아지고있다. 정확한 video quality assessment(VQA) model를 통해서 UGC/consumer video를 monitor, control, optimize해야 한다. Blind quality prediction of in-the-wild videos is quite challenging, since the quality degradations of UGC videos are unpredictable, complicated, and often commingled. 

저자는 이런 challenege에 대응할 수 있는 blind video quality asssessment model을 만들었다. 먼저 주요 no-reference/ blind video quality assessmnet (BVQA) features와 model를 fixed evaluation architecture를 기반으로 평가해서 전반적인 VQA의 발전 현황을 분석했다. 그리고 subjective video quality studies와 objective VQA model design에 대해 새로운 empirical insights를 찾아냈다. 

새로운 BVQA model을 만들어냈다. BVQA models에 feature selection 전략을 적용해서 기존 methods에 존재하는 763개의 statistical features 중 60개를 선별해서 새로운 fusion-based model을 만들고 이것을 VIDeo quality EVALuator (VIDEVAL)이라는 이름을 붙였다. 이 model은 VQA performance와 efficiency사이의 trade off를 balance한다. 실험을 통해서 다른 주요 model 대비 computational cost는 적지만 높은 수준의 performance를 확보할 수 있다는 것을 확인했다. 

Our study protocol also defines a reliable benchmark for the UGC-VQA problem, which we believe will facilitate further research on deep learning-based VQA modeling, as well as perceptually-optimized efficient UGC video processing, transcoding, and streaming. 

<br>

**source code: https://github.com/vztu/VIDEVAL**

<br>

<br>

## Study on the Assessment of the Quality of  Experience of Streaming Video

Zhengfang Duanmu group의 연구 결과에 더해서 experiment를 진행 함.

the main problem of previous studies was insufficient data sets & not consistent with the actual behavior of the network & the distortion introduced was too artificial. SqoE-III database가 사용되었었는데, 이 database에도 문제가 많음.

이 논문의 저자는 이번 연구에서: 

- analyzed the influence of classic and handcrafted metrics on QoE
- investigated the dependence of MOS curve on the integral indicator of objective quality 
- propose several variants of objective assessment models for VQA

<br>

**current state of art of VQA**:

PSNR(peak signal to noise ratio) is commonly used to quantify reconstruction quality for images and video subject to lossy compression.

<br>

**Signal-based models** use only decoded signal to estimate quality. they are based on an estimate of distortion of signal passing through an unknown system and has 3 types: FR(full reference), RR(reduced reference), NR(no reference)

**-FR:** input signal(reference video)에 대한 모든 정보가 필요하다. frame by frame comparison을 계산한다. (this class includes MSE, PSNR, HVS-PSNR SSIM, MS-SSIM, and ITU-T recommendations)

**-RR:** - input system infortmaion의 부분만 필요하다. (this class includes ITU-T J.246 recommendation) 

**-NR:** - 이 mode는 system의 input information에 의존하지 않고 received signal에만 기반한다. (this class includes DIIVINE, BRISQUE, BLINDS, NIQE)

<br>

**Parametric models** - bases on network layer, and is called QoS-metrics. video playback에 대해서는 정보는 없고, packet headers의 정보(delay, jitter, packet loss and others)가 필요하다.

<br>

**Bitstream models**- takes into account the encoded bit streamed packet layer 정보. features such as bitrate, framerate, quantization parameter (QP), PLR, motion vector, macroblock size(MBS), DCT coefficients 등이 extract되어서 model의 input으로 사용된다. 

<br>

**Hybrid models** - 위 3 가지 모델을 혼합함. 그래서 가장 effective하다. 

<br>

**overview of the database(SqoE-III)**

저자는 SqoE-III database의 content, description of metrics(attributes), MOS에 주는 영향을 분석하고, reference video의 name, FPS, SI, TI, description을 확보했다. 

SI=spatial information - 동일 프레임 내에서 pixel들의 변화 정도로 수치가 높을 수록 frame내에서의 변화가 큼을 의미함.

TI=temporal information - 영상 안에서의 시간벅 변화로 이전프레임과 현재 프레임에서의 픽셀값의 차이. 수치가 낮을 수록 시간별 프레임 값의 변화가 적음을 의미함.

standard quality metrics: 각 항목의 p-value & SRCC(Spearman's rank correlation coefficient) 사용

- initial buffer time (seconds): time from the start of playback initialization to the start of video rendering on the user side.
- rebuffer percentage: ratio of the total duration of  stalling events to the total playback time
- rebuffer count:  the number of rebuffering events
- average rendered bitrate (kbps)
- bitrate switch count
- average bitrate switch magnitude (kbps)
- ratio on highest video quality level
- 그 외 평가 metrics로 13개 더 추가 (advantage numerical quality metrics)

feature engineering - standard client-side QoE media metrics와 이들의 p-value of significance를 사용했다. use a coefficient suitable for a lower level (rank) scale - Spearman's correlation coeffcient- in order to calculate the correlation of absolute and rank scales.



