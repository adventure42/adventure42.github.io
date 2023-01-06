---
layout: post                          # (require) default post layout
title: "Voice Quality Assessment Algorithms"   # (require) a string title
date: 2022-12-23       # (require) a post date
categories: [VoiceQualityAssessment]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [VoiceQualityAssessment]                      # (custom) tags only for meta `property="article:tag"`

---

## Voice/Speech Quality Assessment Algorithms 

### Factors that affect QOE

3 major branches of factors that influence the quality of experience (QoE):

**service & system**

types of channels(mono or stereo), position of microphone, central processing unit(CPU) overload

**network**

Jitter, packet loss, delay of the transmitted speech signal

**content and context of use**

characteristics of speech and voice

location of using particular service (e.g., noisy places as restaurants, stations, airports, etc vs. quite places as homes)

<br>

### Subjective vs. Objective

가장 넓은 범위로 algorithm을 나누는 기준이 subjective vs. objective이다.

#### subjective

voice quality 를 사람이 직접 평가하는 방식. determined by the average quaity index determined by a group of evaluators.

minimum number of participants, cost and time for testing과 같은 disadvantages가 있다. 

대표적인 subjective method:

- P.800: structured as a model to perform subjective quality test in laboratories, aiming to indicate appropriate methods and procedures for the determination of the voice quality in telephony services

- P.862(PESQ): intrusive (full reference: degraded vs. original) objective method that estimates MOS for end-to-end voice quality assessment in Narrow Band (NB) telephone networks. This subjective quality assessment outputs MOS-LQO (Mean Opinion Score Listening Quality Objective)

<br>

#### objective

algorithm input으로 무엇이 넣어지느냐에 따라사 다른 model로 구분된다. Models based on speech use the signal to predict the quality index. 

- Perceptual models are based on speech use the signal to predict the quality index
- Parametric models use different factors, such as network parameters, mainly delay and PLR, voice codec, among others.
- Models based on speech signals are divided into intrusive and non-intrusive methods. Intrusive methods require the original signal as a reference to evaluate the degraded signal. In contrast, non-intrusive methods only need the speech signal at the point at which it is evaluated.

subjective대비 더 적은 cost와 time consume됨. 

대표적인 objective methods: 

- P.862(PESQ): intrusive (full reference: degraded vs. original) objective method that estimates MOS for end-to-end voice quality assessment in Narrow Band (NB) telephone networks. This subjective quality assessment outputs MOS-LQO (Mean Opinion Score Listening Quality Objective)

- P.863(POLQA): objective method that estimates voice quality from NB to Super Wide Band (SWB) 

- P.563: non-intrusive metric for NB (but its performance on networks with PLRs is not satisfactory)

- SDR

<br>

### Perceptual vs. Parametric 

#### perceptual

based on human perception (사람의 지각과 인지를 기반으로 함)

#### parametric

based on parameters from transport layer and client (전송 계층 및 클라이언트 매개변수(페이로드, 비트스트림) 사용)

Perceptual & Intrusive = calculate a perceptually weighted distance between clean reference and the contaminated signal to estimate perceived sound quality (PESQ, POLQA)

Perceptual & Non-intrusive = signal processing algorithm uses only the contaminated signal for evaluation of sound quality (P.563)

<br>

### Intrusive vs. Non-intrusive

#### non-intrusive

=type of objective method that utilizes the contaminated speech signal only or rely on some statistics collected from the network (w/o having to remove the test channel from service and eject test calls. Having the original signal unknown is challenging, but speech quality can be estimated through the following methods - voice payload analysis, Internet protocol analysis, and transmission rating model)

=signal processing algorithm uses only the contaminated signal for evaluation of sound quality
(similar to human listeners)

examples are 3SQM, ITU-T Recommendation P.563

(원음과의 비교 없이 평가음 만을 단독으로 사용해서 음질의 값을 예측하는 평가 방식.)

#### intrusive

=type of objective method that measures MOS by comparing reference signal with the contaminated one, but is not suitable for online, live call quality monitoring purposes (b/c the reference speech is unavailable)

=calculates a perceptually weighted distance between the clean reference and the contaminated signal to estimate 
perceived sound quality
(intrusive methods are considered more accurate as they provide higher correlation with subjective evaluations)
examples are PESQ(Perceptual Evaluation of Speech Quality), ITU-T Recommendation P.862,
and POLQA(Perceptual Objective Listening Quality Assessment) as an update to PESQ to address 
super-wideband speech services as the demand for videoconferening increased

(원음과 평가음을 비교하여 원음에 비해 평가음의 음질이 얼마나 저하되었는지 수치로 나타내는 방식.)

<br>

<br>

# References

