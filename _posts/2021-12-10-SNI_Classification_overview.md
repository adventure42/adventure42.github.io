---
layout: post                          # (require) default post layout
title: "Traffic 및 SNI Classification"                   # (require) a string title
date: 2021-12-10       # (require) a post date
categories: [SNIclassification]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [SNIclassification]                      # (custom) tags only for meta `property="article:tag"`

---





# Traffic/SNI(ESNI) Classification



## 배경



### 왜 traffic/SNI(Server Name Indication) classification이 필요한지?

traffic/SNI classification이 다음 사항들을 위해 매우 중요함:

- monitoring network의 필요성

  HTTPS의 사용이 증가할수록 security, privacy는 개선되지만, ISP의 입장에선 monitoring하기 더 어려워진다. encryption으로 인해 network traffic에 "blind"되어서 network 관리 활동에(e.g., traffic engineering, capacity planning, performance/failure monitoring, caching, etc) 어려움이 생긴다.

  예시) QoS를 통해 critical service의 priority를 조정 할 수 있는데, HTTPS는 QoS measurement의 적용을 방해해서 critical services의 우선순위를 높이는 활동이 어려워 진다. Caching technique를 사용해서 traffic을 이해하고 network latency와 congestion을 감소시킬 수 있지만, 이 또한 encryption으로 인해 방해가 된다.

- security & reliability

  security monitoring이 어려워진다. 이로 인해 anomalies, malicious activities를 선별하는 능력이 감소된다. HTTPS traffic을 analyze할 수 있는 solutions의 demand가 더욱 커지고있다.

- 적절한 resource allocation can be made based on traffic, application type(e.g., streaming, VoIP, FTP, SMTP, HTTP, 등) classification data/이력

  Application or traffic type마다 각각다른 resources (bandwidth, delay, 등)필요함.

  

### HTTPS Traffic and Services Identification

Internet traffic classification에서 HTTPS service monitoring이 어디에 위치하는지:

![HTTPS_service_level](C:\SJL\Traffic_classification\figures\gradularity_of_internet_traffic_classification.PNG)

Web에서 security를 위해 주로 사용되는 security protocol은 SSL과 TLS이다. 각 TCP/IP layers에서 주로 사용되는 security protocol:

![where is SSL,TLS](C:\SJL\Traffic_classification\figures\TCP_IP_stack_security_related_protocols.PNG)









web communication을 안전하게 제공하기위해 HTTPS protocol이 사용된다. HTTPS는 secured TLS/SSL connection위로 HTTP를 사용하는 것임. HTTPS를 통한 website는 SSL certificate을 가진 authenticated server로 host된다. 

note: HTTPS vs. encrypted HTTP - HTTPS는 TLS/SSL를 통한 HTTP를 일컫는다. Encrypted-HTTP는 (VPN, SSH 또는 Tor connections과 같은) higher level encrypted connection을 통한 HTTP를 의미한다.

(The HTTPS only refers to HTTP-over-TLS/SSL, while encrypted-HTTP can be HTTP over any higher level encrypted connection like VPN, SSH or Tor connections)



#### HTTPS의 monitoring이 필요한 이유

- internet상에서 HTTPS traffic이 빠르게 증가하고있다.
- 다양항 종류의 web services에서 HTTPS protocol을 사용해서 privacy와 security를 제공한다.
- HTTPS가 악용될 수 있는 위험이 있다. (used for illegitimate purposes as inappropriate contents or services

TLS는 두개의 authenticated communication parties사이에 secure channel을 제공해서 제 3자가 channel내에서 전송되는 data를 access할 수 없게 방지한다.   

continue with explanation about TLS from Shbair's "Service_Level_Monitoring_of_HTTP_Traffic"





### 왜 HTTPS traffic/SNI classification에 AI가 필요한지?

1. HTTPS의 사용이 보편화 되면서 encryption으로 인해 ISPs나 network administrators가 network의 상세 정보에 둔감해지고 traffic engineering, capacity planning등과 같은 network management 활동을 하기 어려워지고 있다.
2. security와 privacy사이의 trade-off 관계로 인해 어느 한쪽을 손해보아야하는 상황. AI를 기반으로 제한된 정보를 가지고 security에 필요한 classification을 수행할 수 있다.  
3. 기존에 port-based method로 classification이 가능했지만, 지금 recent application들은 다른 상황임. dynamic ports & tunnel을 사용하기때문에 port-based method 사용 불가 함. 현재 Deep packet inspection과 같은 기법으로(predefined signatures를 통해 packet payload를 inspect함.) classification이 가능하지만 다음과 같은 단점들이 있음: 
   - encrypted data handle불가
   - computationally expensive
4. 큰 protocol category의 traffic 분류 보다 더 상세한 identification process가 필요한 경우가 있는데, encrypted traffic을 분류하는 수준보다 더 깊이있게 underlying protocol을 분석할 수 있도록 개선되어야한다.



Machine learning

best result when using decision tree and random forest classifiers



supervised Naive Bayes classifier - encrypted data의 비중이 크지 않았을때에 주로 사용했음.



deep learning

apply RNN and CNN to application level appcations



# References

Wazen M. Shbair, et al (2020). A Survey of HTTPS Traffic and Services Identification Approaches