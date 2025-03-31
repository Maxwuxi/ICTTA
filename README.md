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
