# ICTTA: Imbalanced Continuous Test-Time Adaptation

[![GitHub](https://img.shields.io/github/stars/Maxwuxi/ICTTA?style=social)](https://github.com/Maxwuxi/ICTTA)


Official implementation of the paper **"Dynamic Adaptation for Class-Imbalanced Streams: An Imbalanced Continuous Test-Time Framework"** (under review the visual journel).  
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
```
### ImageNet-to-ImageNetC task 
```bash
cd imagenet
bash run.sh
```
   
Reproducing Experiments
See scripts/ for predefined commands to replicate results on:

CIFAR10-C / ImageNet-C

WideResNet-28-10、WideResNet-40-2 / ResNet-50

## 🙏 Acknowledgements
We thank the authors of TENT and CTTA for their foundational work.

Dataset credits: CIFAR10-C and ImageNet-C from Hendrycks & Dietterich (2019).
For questions regarding the code, please contact yanghao@nudt.edu.cn.
## External data link
+ ImageNet-C [Download](https://zenodo.org/records/2235448)
+ Cifar-10-C [Download](https://zenodo.org/records/2535967)
