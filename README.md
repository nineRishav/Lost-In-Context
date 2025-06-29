# Lost in Context: The Influence of Context on Feature Attribution Methods for Object Recognition  
**Accepted at ICVGIP 2024**

[![arXiv]( https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2411.02833)


## Overview

![Project Illustration](assets/main.png)


This repository contains the code and supplementary materials for our paper *"Lost in Context: The Influence of Context on Feature Attribution Methods for Object Recognition,"* accepted at ICVGIP 2024. Our research highlights the significant impact of context on feature attribution methods used in explaining object recognition models, providing insights into the vulnerabilities of these techniques in real-world scenarios.

## Abstract

Contextual elements can heavily bias feature attribution methods, leading to unreliable interpretations and reducing the trustworthiness of AI systems. We present a comprehensive analysis of how these biases manifest and offer guidance for developing more robust and context-aware explainability frameworks.

---
## Poster 


![Project Illustration](assets/30_Adhikari_Lost_in_Context_page-0001.jpg)

## 📂 Contents

- **`code/`**: Python scripts and modules for running experiments.
- **`data/`**: Sample datasets used in our analysis.
- **`metric/`**: Metrics calculation scripts with explanations:
  - `M1-metric-object_context-attribution-correct.py`: Evaluates correctly classified images.
  - `M5-metric-object_context-attribution-noises-paper.py`: Assesses performance under noise conditions.
- **`utils/`**: Utility scripts and data files:
  - `imagenet_class_index.json`: Contains ImageNet class labels and indices.
- **`environment.yml`**: Conda environment setup file.

---

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nineRishav/Lost-In-Context.git

2. **Install Dependency**
    ```bash
    pip install -r requirements.txt


## 🗂️ Directory Structure

- `metric/`: Contains scripts for calculating various metrics related to image classification.
  - `M1-metric-object_context-attribution-correct.py`: Calculates metrics for correctly classified images.
  - `M5-metric-object_context-attribution-noises-paper.py`: Calculates metrics for images with noise variants.
  - ... (list other relevant scripts with brief descriptions)
- `utils/`: Utility scripts and JSON files used across the project.
  - `imagenet_class_index.json`: JSON file containing ImageNet class indices.
  - ... (list other relevant utility files)
- `environment.yml`: Conda environment configuration file that lists all dependencies required to run the code.


## 📜 Citation
If you make use of our work, please cite our paper:

```
@inproceedings{10.1145/3702250.3702254,
author = {Adhikari, Sayanta and Kumar, Rishav and Mopuri, Konda Reddy and Pachamuthu, Rajalakshmi},
title = {Lost in Context: The Influence of Context on Feature Attribution Methods for Object Recognition},
year = {2025},
isbn = {9798400710759},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3702250.3702254},
doi = {10.1145/3702250.3702254},
abstract = {Contextual information plays a critical role in object recognition models within computer vision, where changes in context can significantly affect accuracy, underscoring models’ dependence on contextual cues. This study investigates how context manipulation influences both model accuracy and feature attribution, providing insights into the reliance of object recognition models on contextual information as understood through the lens of feature attribution methods. We employ a range of feature attribution techniques to decipher the reliance of deep neural networks on context in object recognition tasks. Using the ImageNet-9 and our curated ImageNet-CS datasets, we conduct experiments to evaluate the impact of contextual variations, analyzed through feature attribution methods. Our findings reveal several key insights: (a) Correctly classified images predominantly emphasize object volume attribution over context volume attribution. (b) The dependence on context remains relatively stable across different context modifications, irrespective of classification accuracy. (c) Context change exerts a more pronounced effect on model performance than Context perturbations. (d) Surprisingly, context attribution in ‘no-information’ scenarios is non-trivial. Our research moves beyond traditional methods by assessing the implications of broad-level modifications on object recognition, either in the object or its context. Code available at https://github.com/nineRishav/Lost-In-Context},
booktitle = {Proceedings of the Fifteenth Indian Conference on Computer Vision Graphics and Image Processing},
articleno = {4},
numpages = {10},
keywords = {Context, Explainable AI (XAI), ImageNet, Feature Attribution, Object Recognition},
location = {
},
series = {ICVGIP '24}
}

```
