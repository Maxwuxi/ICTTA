# ICTTA: Imbalanced Continuous Test-Time Adaptation

[![GitHub](https://img.shields.io/github/stars/Maxwuxi/ICTTA?style=social)](https://github.com/Maxwuxi/ICTTA)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Dynamic Adaptation for Class-Imbalanced Streams: An Imbalanced Continuous Test-Time Framework"** (accepted at a top-tier conference).  
**ICTTA** is a novel framework designed to address class imbalance in dynamically evolving test data streams, significantly improving model robustness under continuous domain shifts.

---

## 📌 Overview

Existing Test-Time Adaptation (TTA) methods often assume balanced test data distributions, which rarely hold in real-world scenarios. **ICTTA** tackles this challenge by:
- **Dynamic Adaptive Loss**: Assigns sample-wise weights to prioritize minority classes while preserving majority class performance.
- **Class-Aware Adaptation**: Leverages confidence margins to adjust optimization focus dynamically.
- **Theoretical Guarantees**: Provably enhances minority class adaptation under imbalanced streaming data.

**Key Results**:
- Achieves **16.5%** mean classification error on CIFAR10-C and **68.1%** on ImageNet-C.
- Outperforms state-of-the-art TTA methods (e.g., TENT, CTTA) in precision, recall, and F1-score.

![ICTTA Framework](https://github.com/Maxwuxi/ICTTA/raw/main/assets/framework.png)

---

## 🚀 Quick Start

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Maxwuxi/ICTTA.git
   cd ICTTA
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Usage
Run ICTTA on CIFAR10-C with a WideResNet-28-10 model:
### CIFAR10-to-CIFAR10C-standard task
```bash
cd cifar
# This includes the comparison of all three methods as well as baseline
bash run_cifar10.sh
### ImageNet-to-ImageNetC task 
```bash
# Tested on RTX3090
cd imagenet
bash run.sh
```
   
Reproducing Experiments
See scripts/ for predefined commands to replicate results on:

CIFAR10-C / ImageNet-C

WideResNet-28-10 / WideResNet-40-2 / ResNet-50

## 📊 Results
CIFAR10-C (Severity 5)
Method	Error Rate (%)	Precision (%)	Recall (%)	F1 (%)
Source	44.1	66.5	56.4	52.9
TENT	16.9	79.3	80.1	78.8
ICTTA	16.5	80.2	79.9	79.4
ImageNet-C (Severity 5)
Method	Error Rate (%)	F1 (%)
Source	82.2	16.0
TENT	68.5	29.2
ICTTA	68.1	29.3
## 📖 Citation
If you find this work useful, please cite:

bibtex
@article{ma2024dynamic,
  title={Dynamic Adaptation for Class-Imbalanced Streams: An Imbalanced Continuous Test-Time Framework},
  author={Ma, Wuxi and Yang, Hao},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}


## 🙏 Acknowledgements
We thank the authors of TENT and CTTA for their foundational work.

Dataset credits: CIFAR10-C and ImageNet-C from Hendrycks & Dietterich (2019).
For questions regarding the code, please contact yanghao@nudt.edu.cn.

