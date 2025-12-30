# 🎆Welcome to the Official Repo for EARAM! 
Welcome to our repository!

This repo contains the code and data used in the experiments for our paper, "**From Predictions to Analyses: Explainable Rationale-Augmented Fake News Detection with Large Vision-Language Models**".



## 🔧Dependency
+ python 3.7+
+ torch 1.13.0+
+ transformers 4.28.0
+ numpy 1.26.4


## 📦Dataset
We need to use the Pheme, MR2 and Weibo datasets, and their raw datasets can be downloaded directly from the following two links:

https://github.com/THU-BPM/MR2

https://github.com/drivsaf/MFAN

#### Note that the path in the code needs to be modified before it can run


## 🚀Citation
 ```
@inproceedings{10.1145/3696410.3714532,
author = {Zheng, Xiaofan and Zeng, Zinan and Wang, Heng and Bai, Yuyang and Liu, Yuhan and Luo, Minnan},
title = {From Predictions to Analyses: Rationale-Augmented Fake News Detection with Large Vision-Language Models},
year = {2025},
isbn = {9798400712746},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696410.3714532},
doi = {10.1145/3696410.3714532},
abstract = {The rapid development of social media has led to a surge of eye-catching fake news on the Internet, with multimodal news comprising both images and text being particularly prevalent. To address the challenges of Multimodal Fake News Detection (MFND), numerous supervised task-specific Multimodal Small Language Models (MSLMs) have been developed. However, these models lack the breadth of knowledge and the depth of language understanding, which results in unsatisfactory adaptability, generalization, and explainability performance. To address these issues, we attempt to introduce Large Vision-Language Models (LVLMs), aiming to leverage the common sense understanding and logical reasoning abilities of LVLMs for the MFND task. We observed that LVLMs can generate reasonable analyses of news content from specific angles. However, when it comes to synthesizing these analyses for final judgment, their performance declines significantly, failing to meet the accuracy benchmarks set by existing MSLMs detection models. This reflects the need for a more effective way for LVLMs, which have not undergone task-specific training, to utilize their knowledge and capabilities. Based on these findings, we propose the Explainable Adaptive Rationale-Augmented Multimodal (EARAM) framework, which adaptively uses MSLMs to extract useful rationales from the multi-perspective analyses of LVLMs. After making judgments based on these rationales, EARAM then assists LVLMs in generating more reliable explanations. Extensive experiments demonstrate that our model not only achieves state-of-the-art results on widely used datasets but also significantly outperforms other models in terms of generalization and explainability.},
booktitle = {Proceedings of the ACM on Web Conference 2025},
pages = {5364–5375},
numpages = {12},
keywords = {explainable, fake news detection, large vision-language models},
location = {Sydney NSW, Australia},
series = {WWW '25}
}
 ```
