---
layout: post                          # (require) default post layout
title: "GRU Network"                   # (require) a string title
date: 2021-12-17       # (require) a post date
categories: [SNIclassification]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [GRU]                      # (custom) tags only for meta `property="article:tag"`
---



# GRU network

GRU = improved version of standard RNN

vanishing gradients problem을 해결하기위해 2개의 vectors - **update gate**와 **reset gate**를 활용함.

이 두 vectors는 어떤 information이 output으로 pass되어야하는지를 결정함. 오래전 (sequence에서 멀리서)부터 희미해지는 영향 없이 정보를 기억하도록 훈련시킬수도 있고 반대로 prediction에 관련되어 있지 않다고 판단되는 정보는 제외시킬 수 있는 능력이 있다.  

![RNN_with_GRU](C:\SJL\Traffic_classification\figures\RNN_GRU.png)



![GRU_unit](C:\SJL\Traffic_classification\figures\GRU_unit.png)

![](C:\SJL\Traffic_classification\figures\GRU_unit_specifics.png)



### 1. update gate

![update gate](C:\SJL\Traffic_classification\figures\GRU_update_gate.png)



### 2. reset gate

![reset gate](C:\SJL\Traffic_classification\figures\GRU_reset_gate.png)





### 3. current memory content



![current memory](C:\SJL\Traffic_classification\figures\GRU_current_memory.png)



### 4. final memory at current time step



![final_memory](C:\SJL\Traffic_classification\figures\GRU_final_memory.png)



# References

1. Simeon Kostadinov. (2017, December 16). *Understanding GRU Networks.* from https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

