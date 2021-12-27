---
layout: post                          # (require) default post layout
title: "Encrypted Traffic Classification"                   # (require) a string title
date: 2021-12-27       # (require) a post date
categories: [trafficClassification]          # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [trafficClassification]                      # (custom) tags only for meta `property="article:tag"`

---



# Encrypted Traffic Classification

<br>

<br>

## Background

web application security context에서 web은 standard Internet TCP/IP protocols위에서 실행되는 client/server application을 의미한다. 그래서 안전한 web을 구축하기 위해 security protocols는 client/server architecture와 같은 operating environment, user가 누구인지, 등 몇가지 사항들을 고려해야한다.

client side에서는 web browser를 통해 쉽게 사용 가능하지만, server side에서는 공격에 쉽게 노출될 수 있기 때문에 level of trust & confidence를 높게 유지하는 것이 필수이다. 

<br>

Internet traffic classification을 다음과 같은 구조로 세분화 할 수 있다: 

**Data Network Traffic > Encrypted Traffic > SSL/TLS > HTTPS service-level monitoring** 

![HTTPS_service_level](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/gradularity_of_internet_traffic_classification.PNG)

communication을 secure하는 방법은 여러가지 이다. TCP/IP protocol layer에서 각자의 위치에 관련된 mechanism을 통해 다양한 방법들이 존재한다. 여러 security protocol들이 각각의 각각 장/단점을 가지고있지만, 그 중 security를 위해 주로 사용되는 security protocol은 SSL과 TLS이다. 

각 TCP/IP layers에서 주로 사용되는 security protocol:

![where is SSL,TLS](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TCP_IP_stack_security_related_protocols.PNG)

SSL and TLS security protocols는 application layer와 transport layer사이에 있다. SSL과 TLS는 Public Key Infrastructure (PKI)를 사용해서 authentication을 제공하고 그 다음 confidentiality를 위한 symmetric key를 제공한다. 

<br>

<br>

## Intro to HTTPS protocols & TLS(Traffic Layer Security)

TLS에서 HTTP를 사용하는 것이 HTTPS이다.

HTTPS = HTTP over TLS/SSL. website의 url이 "https://"로 시작된다. website 가 SSL certificate을 소지하고있는 authenticated server에서 host된다. a client과 a server negotiate the connection parameters before exchanging data thanks to HTTP over a dedicated secure link between them. 

HTTPS와는 다른 개념의 encrypted HTTP가 있는데, 다음과 같이 define된다.

encrypted-HTTP = HTTP over any higher level encrypted connection like VPN, SSH or Tor connections. website의 urls이 "http://"로 시작된다. (web server가 스스로 secure connection service를 제공하지는 못하고 server가 아닌 선택된 method안에서 secure link parameters가 configure되는 것을 의미한다.)

<br>

TLS는 다음과 같은 문제들을 방지한다 - eavesdropping, data tampering(deliberately modifying, destroying, manipulating or editing data through unauthorized channels). 즉, 두개의 authenticated communication parties사이에 secure channel을 제공해서 제 3자가 channel내에서 전송되는 data를 access할 수 없게 방지한다.  

Web에서 TLS protocol이 널리 사용되는 이유는 web상에서 unauthorized view of users' information을 방지해주기 때문이다. user쪽에서 configuration이 필요하지 않다. 그래서 web browser에서 쉽게 사용될 수 있다. (all web browsers and web servers fully support TLS natively)

<br>

HTTPS의 monitoring이 필요한 이유

- internet상에서 HTTPS traffic이 빠르게 증가하고있다.
- 다양항 종류의 web services에서 HTTPS protocol을 사용해서 privacy와 security를 제공한다.
- HTTPS가 악용될 수 있는 위험이 있다. (used for illegitimate purposes as inappropriate contents or services

<br>

### TLS architecture

TLS는 두 개의 protocol layers로 형성되어있다. top-layer & lower-layer:

<img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_layer_sub_protocols.PNG" style="zoom:80%;" />

**top-layer:** 세 개의 handshaking sub-protocol로 구성되어있다 -the Handshake, the Change Cipher Specification, and the Alert protocol. 이 protocol들은 TLS exchanges를 관리하는데에 사용된다. allow peers to agree on an encryption algorithm and a shared secret key, to authenticate themselves, and to report errors to each other.

**lower-layer:** TLS Record protocol을 가지고있다. application data와 top layer protocols로 부터 온 TLS messages를 위한 envelop과 같이 표현된다. 이 Record protocol은 data를 chunks로 나누는 것을 담당하고있다. chunk로 split된 data는 optionally compress되고, MAC으로 authenticate되고, encrypted되어서 finally transmit된다. 

<br>

#### TLS handshake protocol

**TLS handshake diagram:**

<img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_handshake_protocol.PNG" style="zoom:67%;" />

handshake는 처음 interactions를 define하고 여러 configuration aspects를 책임지고있다 - cipher suite negotiation, server/client authentication, session key exchanges, 등등

Cipher suite negotiation동안에는 client와 server가 data exchange에 사용되는 cipher suite에 대한 agreement를 만든다. 

Authentication에서는 both parties가 PKI method를 통해서 identity를 prove한다. 

session key exchange에서는 Pre-Master Secret이라고 불리는 random하고 special한 number들을 exchange한다. 이 number들이 Master key를 생성하는데에 사용된다. Master key는 data exchange동안 encryption에 적용된다. 

<br>

handshake과정을 통해 client와 server 양쪽이 "finished" message를 받고 validate되었다면, TLS connection를 통해서 양쪽에서 encrypted application data를 주고받을 수 있다.

<br>

**Full TLS handshake:**

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_handshake_protocol_detailed.PNG)

<br>

**상세 과정:**

1. The **ClientHello** message contains the usable cipher suites, supported extensions and a random number. 
2. The **ServerHello** message holds the selected cipher, supported extensions and a random number. 
3. The **ServerCertificate** message contains a certificate with a server public key. 
4. The **ServerHelloDone** indicates the end of the **ServerHello** and associated messages. If the client receives a request for its certificate, it sends a **ClientCertificate** message. 
5. Based on the server random number, the client generates a random Pre-Master Secret, encrypts it with the public key given in the server’s certificate and sends it to the server. 
6. Both client and server generate a master secret from the Pre-Master Secret and exchanged random values. 
7. The client and the server exchange **ChangeCipherSpec** to start using the new keys for encryption. 
8. The client sends the **Finished** message to verify that the key exchange and authentication processes were successful. 
9. The server sends the **Finished** message to the client to end the handshake phase.

<br>

<img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_handshake_parameters.PNG" style="zoom:67%;" />

<br>

TLS connection이 resume될때 실행 속도를 높이기 위해 간소화된 TLS handshake가 수행된다면, session key들의 recomputing만을 요구할 수 있다. 이렇게 간소화된 방식에서는 server와 client가 이미 서로를 알고있고 master key를 공유하기때문에 ServerCertificate, ServerHelloDone, 그리고 ClientKeyExchange 단계들이 skip되는 것이다.

**TLS handshake resume시:** 

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_handshake_shortened.PNG)

resumed session은 client가 이전 connection의 ClientHello 단계로 부터 존재하는 session ID를 submit하면서 trigger된다. (server는 동일한 session ID로 respond해야한다 ID가 틀리다면, full handshake가 요구된다.)

<br>

 #### change cipher specification protocol

이 protocol은 client와 server가 사용하는 encryption algorithm을 변경한다. subsequent records는 새롭게 negotiate되는 algorithm과 key 아래에서 protect된다. protocol은 single byte를 가진 single message로 구성되어있다. 또한 connection을 다시 negotiate할 필요 없이 TLS session을 변경하도록 허락해준다. ChangeCipherSpec message는 보통 TLS handshake의 마지막에 전송된다. 

<br>

note:

SNI(Server Name Indication)란? 

TLS handshake의 extension이다. TLS handshake는 destination hostname을 가지고있고 ClientHello message로 부터 extract된다. SNI는 HTTPS traffic inspection의 주요 component이다. security를 보존하기 위해 firewalls는 SNI를 inspect해서 server name이 허락되어있는지를 확인한다. Internet services를 censor하기 위해서도 intermediaries가 SNI를 사용한다. SNI 자체는 encrypt되어 있지 않다. 2018년 부터 ESNI(encrypted SNI)가 개발되어서 issue of domain eavesdropping 방지하는데에 기여하고있다.

<br>

<br>

### TLS development의 key role players

<br>

#### TLS implementation libraries

web browser를 통해 access하는 HTTP services를 보면, 각각의 web browser가 다른 implementation of the TLS protocol를 사용한다. 

Mozilla - Network Security Services (NSS) libraries to provide the support for TLS in Firefox web browser and other services (e.g., Thunderbird, SeaMonkey)

Google Chrome - NSS libraries (but Google has developed its own fork of OpenSSL, named BoringSSL)

<br>

#### SSL certificate

HTTPS에서 client와 server는 server의 domain name와 연관된 public-key를 가진 X.509 digital certificate을 보여주면서 TLS handshake를 완성한다. 이 certificate들은 trusted CA로 부터 issue된것으로 간주한다. 

<img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/SSL_certificate.PNG" style="zoom: 67%;" />

위 그림에서 보이는바와 같이 X.509 certificate은 website domain과 link되어있고 temporal validity period, a public key, 그리고 trusted CA로부터 제공된 digital signature를 가지고있다. web browser는 certificate의 identity가 request된 domain name과 match 하는지, certificate이 validity period내에 포함되는지, 그리고 certificate의 digital signature이 valid한지를 확인한다. certificate의 public key는 client가 server와 session secret을 공유해서 end-to-end encrypted channel을 형성하기위해 사용된다. 

SSL certificates는 validation process depth에 따라서 3 가지 type들이 존재한다:

- **Domain Validated certificate (DV)** - asserts that a domain name is mapped to the correct web server (IP address) through DNS. (그러나 이 type은 organization information을 identify하지 못한다. so it should not be used for commercial website)
- **Organization Validated certificate (OV)** - includes additional CA-verified information (such as organization named and postal address) 이렇게 extra validation 이 있어서 OV SSL certificate가 DV certificate보다 더 비싸다. 
- **Extended Validation certificate (EV)** - uses the highest level of authentication, including diligent human validation of a site's identity and business registration details. 이런 extensive validation때문에 EV가 가장 비싼 SSL certificate이다. 

여러개의 domain names가 필요한 web applications의 경우 SSL certificates의 cost를 감소시키고 관리하기위해서 하나의 X.509 certificate이 여러개의 host와 domains에 link 될 수 있다. 이런 방식은 X.509v3의 specification이 SubjectAltName field을 지원하기때문에 가능하다. 이것은 하나의 certificate가 하나 이상의 host 또는 domain name을 specify할 수 있도록 허락한다. 그리고 CommonName과 SubjectAltName field들의 wildcards를 지원한다.

 <br>

<br>

## TLS traffic identification

Encrypted traffic의 여러 type들 중에서 TLS traffic을 감지한다. TLS traffic identification 은 다음과 같이 세가지 방식으로 나누어볼 수 있다. Port-based와 structure-based method는 TLS vs. non-TLS를 구분하는 identification에 집중되어있고, machine learning-based method는 identification외에도 TLS traffic의 분류에도 집중되어있다. 

![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/identification_of_TLS.PNG)

- **port-based method**

  (transport layer port number들이 Internet Assigned Numbers Authority(IANA)로 부터 배정되기 때문에) port-based method는 Internet application들과 protocol들을 identify하는 직관적인 방법이다. 그러나 TLS protocol은 여러 application layer protocols에서 널리 사용되기때문에, port-number 만으로 TLS traffic을 identify하는 것은 불가능하다. (e.g.,  For instance, HTTPS, FTPS and SMTPS protocols use TLS over port 443, 990, 465 respectively. Moreover, 8% of non-TLS traffic use standard TLS ports, while 6.8% of TLS traffic use ports not officially associated with TLS.) 그래서 deeper and more robust identification method for TLS가 필요하다.

- **protocol structure-based method**

  TLS-level DPI(deep packet inspection) 기법은 packet의 payload를 분석해서 TLS format을 알아보도록 활용되어왔다. 더 자세하게는 TLS Record Protocol structure을 기반으로 packet의 payload를 분석해서 TLS traffic을 detect한다. 

  ![](https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_record_format.PNG)

  한가지 방법으로 standard TLS format을 사용해서 ServerHello packets를 detect할 수 있다. (ServerHello packet은 TLS handshake protocol의 한 부분이고, TLS connection의 parameter를(TLS version, Selected Cipher, 등) 설정한다.) 그래서 valid ServerHello packet이 존재한다는 사실은 현재 monitoring하고있는 flow가 TLS라는 알려주는 중요한 근거가 될 수 있다. 

  다른 방법으로는 "TLS Traffic Detector"라로 불리는 기법이 있다. 이 기법은 pure TLS flows를 따로 분리한다. (그후 service 분류를 위해 더 깊이있게 proceed한다.) 이 TLS Detector는 packets payload의 처음 5 bytes를 standard TLS record format과 비교해서 결정을 내린다. 이 기법은 TLS packet payload들이 동일한 structure로 시작한다는 점을 활용하기때문에, 현재 monitoring하고있는 flow내의 어떤 packet이든 payload의 처음 몇 bytes를 확인하는것만으로 충분히 해당 flow가 TLS인지 확인할 수 있다. (이전 기법과 같이 반드시 ServerHello packet을 사용하는것에 제한되지 않아도 됨.)  

  OpenDPI 기법은 DPI를 기반으로 traffic identification을 진행한다. OpenDPI는 TLS Record protocol내의 정보를 활용해서 100%의 정확도로 TLS traffic을 분류하고 TLS flow의 두 phases를 구분할 수 있다. 첫번째 phase에서는 content type과 TLS version을 읽기위해 payload에 있는 valid한 TLS Record Protocol structure를 가진 packet을 detect한다. 두번째 phase에서는 OpenDPI가 반대 방향의 packet을 intercept한다. 만약 하나 또는 그 이상의 TLS Record protocol structure를 가지고있다면, OpenDPI는 이 packet을 TLS로 marking하고 양방향의 모든 packet들을 확인한다.

  또 다른 structure-based method는 Double Record Protocol Structure Detection (DRPSD)이다. 첫 8 packet를 활용해서 TLS traffic을 identify한다. Record Protocol Structure를 기반으로 TLS protocol를 identify하기위해 Record Protocol Structure가 몇개가 detect되었는지가 중요하다. 거의 모든 TLS flow는 하나 또는 그 이상의 packet들이 두 개의 Record Protocol Structure을 가지고있다는 사실을 활용한다. 그래서 packet의 payload를 확인해서, 만약 packet의 payload에서 double Record Protocol Structure가 확인된다면, 해당 flow는 TLS flow로 identify된다. DRPSD 기법의 연구 결과로 99.17% 의 identification 정확도가 확인되었다. [Liu et al. (2012) DRPSD: An novel method of identifying SSL/TLS traffic](https://ieeexplore.ieee.org/document/6321091) 

- **machine learning-based method**

  flow의 feature들을 활용해서 machine learning 기법으로 encrypted traffic을 classify하는 방법들이 많이 연구되었다. 

  <img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/identify_TLS_flow_ML.PNG" style="zoom:80%;" />

  flow duration, packet size, inter-arrival time와 같은 flow의 statistics가 feature들로 사용되어서 TLS protocol을 위한 statistical signature를 만들었다. 

  machine learning method를 사용하기 위해서는 training dataset, relevant statistical features, machine learning algorithm and evaluation techniques가 필요하다.   

  learning process는 세 가지 phases로 나뉜다 -  training, classification, and validation. 

  **machine learning process:**

  <img src="https://raw.githubusercontent.com/adventure42/adventure42.github.io/master/static/img/_posts/TLS_traffic_ML_phases.PNG" style="zoom:80%;" />

  training에서는 statistical feature들과 machine learning algorithms가 훈련되어서 prediction을 만든다. training phase의 output은 classification phase에서 unseen data를 identify할때에 사용되는 model이다. validation phase에서는 classification의 결과가 validate되어서 model의 classification 성능을 측정한다. 

  4 개의 주요 machine learning algorithm은 AdaBoost, C4.5, RIPPER, Naive Bayesian이다. 

  이 algorithm들은 packet size와 packets inter-arrival time을 기반으로 계산된 22개의 statistical features (e.g., mean, standard deviation, max, min, 등)와 함께 evaluate되었다. 연구 결과로  AdaBoost algorithm이 95%의 정확도와 4% False positive rate으로 가장 높은 성능을 보였다. (해당 연구 저자들이 locally 생성한 dataset을 기반으로 50,000개의 TLS flows를 통해 훈련한 모델을 평가한 결과)

  <br>

### 왜 HTTPS traffic classification이 필요한지?

traffic classification이 다음 사항들을 위해 매우 중요함:

- monitoring network의 필요성

  HTTPS의 사용이 증가할수록 security, privacy는 개선되지만, ISP의 입장에선 monitoring하기 더 어려워진다. encryption으로 인해 network traffic에 "blind"되어서 network 관리 활동에(e.g., traffic engineering, capacity planning, performance/failure monitoring, caching, etc) 어려움이 생긴다.

  예시) QoS를 통해 critical service의 priority를 조정 할 수 있는데, HTTPS는 QoS measurement의 적용을 방해해서 critical services의 우선순위를 높이는 활동이 어려워 진다. Caching technique를 사용해서 traffic을 이해하고 network latency와 congestion을 감소시킬 수 있지만, 이 또한 encryption으로 인해 방해가 된다.

- security & reliability

  security monitoring이 어려워진다. 이로 인해 anomalies, malicious activities를 선별하는 능력이 감소된다. HTTPS traffic을 analyze할 수 있는 solutions의 demand가 더욱 커지고있다.

- 적절한 resource allocation can be made based on traffic, application type(e.g., streaming, VoIP, FTP, SMTP, HTTP, 등) classification data/이력

  Application or traffic type마다 각각다른 resources (bandwidth, delay, 등)필요함.




### 왜 HTTPS traffic classification에 AI가 필요한지?

1. HTTPS의 사용이 보편화 되면서 encryption으로 인해 ISPs나 network administrators가 network의 상세 정보에 둔감해지고 traffic engineering, capacity planning등과 같은 network management 활동을 하기 어려워지고 있다.
2. security와 privacy사이의 trade-off 관계로 인해 어느 한쪽을 손해보아야하는 상황. AI를 기반으로 제한된 정보를 가지고 security에 필요한 classification을 수행할 수 있다.  
3. 기존에 port-based method로 classification이 가능했지만, 지금 recent application들은 다른 상황임. dynamic ports & tunnel을 사용하기때문에 port-based method 사용 불가 함. 현재 Deep packet inspection과 같은 기법으로(predefined signatures를 통해 packet payload를 inspect함.) classification이 가능하지만 다음과 같은 단점들이 있음: 
   - encrypted data handle불가
   - computationally expensive
   - privacy challenge
4. 큰 protocol category의 traffic 분류 보다 더 상세한 identification process가 필요한 경우가 있는데, encrypted traffic을 분류하는 수준보다 더 깊이있게 underlying protocol을 분석할 수 있도록 개선되어야한다.

<br>

Machine learning -  encrypted data의 비중이 크지 않았을때에 supervised Naive Bayes classifier가 주로 사용되었음. encrypted data를 다루면서 random forest, AdaBoost등을 중심으로 주요 classifier의 성능이 확인되었음.

Deep learning - CNN과 RNN을 혼합하여 설계한 모델을 중심으로 연구되고 있음.

<br>

<br>

# References

1. Wazen M. Shbair, et al (2020). A Survey of HTTPS Traffic and Services Identification Approaches

2. Wazen M. Shbair. (2017). Service-Level Monitoring of HTTPS Traffic