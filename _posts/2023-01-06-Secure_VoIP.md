---
layout: post                          # (require) default post layout
title: "Secure VoIP"   # (require) a string title
date: 2023-01-06       # (require) a post date
categories: [VoiceQualityAssessment]          # (custom) some categories, but make sure these categories already exists inside path of `category/`
tags: [VoiceQualityAssessment]                      # (custom) tags only for meta `property="article:tag"`

---

# Secure VoIP

## Encryption

Algorithm과 key를 사용해서 message content를 숨긴다.

Encryption scheme마다 어떻게 encryption, decryption이 제공되고 key가 사용되는지가 다르다.

Due to the higher computational complexity, asymmetric schemes are typically just used for key agreement and then the actual payload is encrypted with a symmetric scheme.

### CTR(counter) mode of operation

CTR은 random block of data를 access 할 수 있도록 한다. counter function을 사용해서 block의 순서를 결정하는 기준을 설정한다. 그림에서 보이는바와 같이, cipher CIPH와 key K를 사용해서 counter를 encrypt되고, plain text와 exclusive-OR되어 cipher text를 생성한다. 

(exclusive-OR: true if and only if arguments differ)

CTR mode is suitable for streaming data over connection-less transmissions since blocks do not depend on information in other blocks. 

The loss of one packet will hence not impact the decryption of other packets as long as the counter is kept in synchronization

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/encryption_CTR_operation.PNG)

### CBC mode of operation

아래 그림에서 보이는바와 같이, cipher CIPH와 key K를 사용해서 plain text를 encrypt하고 cipher text를 생성한다. 

The first block is exclusive-ORed with an Initialization Vector (IV) which must be random but not necessarily secret. The purpose of the IV is to ensure that the same plain text does not result in the same cipher text when encrypted multiples times.

(exclusive-OR: true if and only if arguments differ)

Previous block에 의존하기때문에, the loss of a packet will impact the decryption of subsequent data.

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/encryption_CBC_operation.PNG)

### Synchronization

session동안 어떻게 encrypt된 packet들이 synchronize 되는지?

SCIP protocol이 사용되는 경우를 설명:

SCIP(Secure Communication Interoperability Protocol)는 CTR mode로 AES(Advanced Encryption Standards)를 사용해서 encryption을 구현한다.

CTR mode에서 counter로 128 bit 크기의 State Vector이 생성됨. cryptographic synchronization을 통해 encoder와 decoder는 반드시 동일한 block을 operate할 때에 동일한 State Vector를 사용하게 된다.

Network에서 packet이 전송될 때, State Vector 정보가 Sync Management Frames의 도움으로 packet과 함께 전송된다.

short term component는 frame의 한 부분으로, long term component는 3개의 연속적인 Sync Management Frames로 나뉜다. 

State vector(counter)가 어떻게 Syn Management Frame들과 mapping되는지 도식화:

 ![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/mapping_of_state_vector_to_sync_managemnet_frame.PNG)

### Packetization

각각의 network packet은 하나의 superframe을 가지고있고, 이 frame의 structure은 사용된 voice-codec에 따라 결정된다.

G.729D superframe은 Sync Management frame으로 시작한다. 그리고 1~8개의 encrypted speech frames들로 이어진다. 각 speech frame은 그림에서 보이는 바와 같이 lowest 8-bits of the counter와 4개의 encrypted G.729D frame들로 구성되어있다. 그래서 각 superframe은 40~320 milisecond의 speech를 포함하고 있다. 이 superframe은 20 milisecond interval로 voice를 packetize하는 cleartext VoIP와 비교된다. 

superframe의 구조:

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/superframe_structure.PNG)

<br>

<br>

# References

1. Parametric Prediction Model for Perceived Voice Quality in Secure VoIP (Andersson, 2019)
