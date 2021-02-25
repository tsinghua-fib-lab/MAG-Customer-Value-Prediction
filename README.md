# MAG-Customer-Value-Prediction
The official PyTorch implementation of "Predicting Customer Value with Social Relationships via Motif-based Graph Attention Networks" (WebConf '21).
![Model Framework](model_framework.jpg)
## Abstract
Customer value is essential for successful customer relationship management. Although growing evidence suggests that customers' purchase decisions can be influenced by social relationships, social influence is largely overlooked in previous research. In this work, we fill this gap with a novel framework --- Motif-based Multi-view Graph Attention Networks with Gated Fusion (MAG), which jointly considers customer demographics, past behaviors, and social network structures. Specifically, (1) to make the best use of higher-order information in complex social networks, we design a motif-based multi-view graph attention module, which explicitly captures different higher-order structures, along with the attention mechanism auto-assigning high weights to informative ones. (2) To model the complex effects of customer attributes and social influence, we propose a gated fusion module with two gates: one depicts the susceptibility to social influence and the other depicts the dependency of the two factors. Extensive experiments on two large-scale datasets show superior performance of our model over the state-of-the-art baselines. Further, we discover that the increase of motifs does not guarantee better performances and identify how motifs play different roles. These findings shed light on how to understand socio-economic relationships among customers and find high-value customers.

This repository provides a PyTorch implementation of MAG as described in the paper:
Jinghua Piao, Guozhen Zhang, Fengli Xu, Zhilong Chen, Yong Li. 2021.Predicting Customer Value with Social Relationships via Motif-based GraphAttention Networks.


