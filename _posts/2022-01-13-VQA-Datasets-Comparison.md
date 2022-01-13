---
layout: post                          # (require) default post layout
title: "VQA Datasets Comparison"                   # (require) a string title
date: 2022-01-13       # (require) a post date
categories: [paperreview]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [VQA]                      # (custom) tags only for meta `property="article:tag"`

---

<br>

# VQA Datasets

## KoNViD-1k

link: [http://database.mmsp-kn.de/konvid-1k-database.html](http://database.mmsp-kn.de/konvid-1k-database.html)

- 다른 datasets대비 reliability관점에서 다소 떨어지지만, content diversity가 높은 강점을 가지고 있음.

- 포함된 features - blur, colorfulness, contrast, SI, TI and NIQE

- video quality research와 연관성이 약한 contents도 포함되어 있음.

- contents들이 original에서부터 clipped되어 540p로 resize됨.

<br>

KoNViD-1k의 특징:

- 장점: 
  - 다른 dataset(LIVE-VQC, YouTube-UGC)대비 더 넓은 contents diversity & 더 다양한 colorfulness 
  - Natural distortion 학습 가능

- 단점: 
  - video의 sharpness와 SI지수가 비교적 낮음. 
  - (큰 DB에서 crawling & sampling 방식이 아닌)crowd sourcing으로 형성된 dataset이기에 video quality 연구와 연관성이 떨어지는 video도 포함됨. 
  - MOS값의 reliability가 조금은 떨어지는 경향이 있음. 

<br>

<br>

## LIVE-VQC

link: [https://live.ece.utexas.edu/research/LIVEVQC/](https://live.ece.utexas.edu/research/LIVEVQC/)

- contents include variety of camera motions, some night scenes(outlier가 될 가능성이 큼)

- resolution이 uniform하게 distribute되어있지 않음.

<br>

LIVE-VQC의 특징:

- 장점:
  - camera motion이 다양하게 포함되어있어 다른 datasets대비 TI 지수가 높은 경향이 있음.
- 단점:
  - video quality assessment에 outlier 영향을 끼칠 수 있는 data가 포함되어있지만, MOS 값의 standard deviation이 주어지지 않음.
  - MOS의 distribution이 높은쪽으로 치우쳐져 있음.

<br>

<br>

## YouTube-UGC

link: [https://media.withyoutube.com/](https://media.withyoutube.com/)

- crowd sourcing보다, large DB로부터 crawling&sampling으로 만들어져서 user camera에서부터 바로 가져온것보다 더 uniformly distributed content diversity 확보됨.

- 포함된 features - spatial, color, temporal, chunk variations

- 가진 contents들이 15가지 category로 나뉨. (e.g. HDR, screen content, animation, gaming, etc)

- wider content diversity than LIVE-VQC

<br>

YouTube-UGC의 특징:

- 장점:
  - 다른 datasets대비 더 다양한 contents를 포함하고있고 feature이 더 uniformly distributed되어 있음.
- 단점:
  - MOS의 distribution이 높은쪽으로 치우쳐져 있음.

<br>

<br>

## MOS comparison:

| dataset         | MOS range    | absolute range              | MOS distribution/  특징                    | num of ratings             |
| --------------- | ------------ | --------------------------- | ------------------------------------------ | -------------------------- |
| **KoNViD-1k**   | 1.22  ~ 4.64 | categoy  of 1~5             | crowdsourcing으로인해  some are unreliable | 136,800 (114 votes/video)  |
| **LIVE-VQC**    |              | continuous  values of 0~100 | skewed  to higher scores                   | 205,000  (240 votes/video) |
| **YouTube-UGC** |              | continuous  values of 1~5   | skewed  to higher scores                   | 170,159 (123 votes/video)  |

<br>

<br>

## low level features comparison:

| dataset         | brightness            | contrast              | colorfulness                                                 | sharpness                                       | SI                                              |
| --------------- | --------------------- | --------------------- | ------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------- |
| **KoNViD-1k**   | YouTube-UGC와  비슷함 | YouTube-UGC와  비슷함 | higher  than LIVE-VQC and YouTube (source가 flickr인 만큼, 다양한 color가 포함되어있음.) | lower쪽으로  치우침 (than LIVE-VQC and YouTube) | lower쪽으로  치우침 (than LIVE-VQC and YouTube) |
| **LIVE-VQC**    |                       |                       |                                                              |                                                 |                                                 |
| **YouTube-UGC** | KoNViD-1K와  비슷함   | KoNViD-1K와  비슷함   |                                                              | wider  range than Konvid and LIVE-VQC           | wider  range than Konvid and LIVE-VQC           |

<br>

<br>

# Reference

1. UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content by Zhengzhong Tu (Apr 2021)