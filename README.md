# Dynamic Adaptation for Class-Imbalanced Streams: An Imbalanced Continuous Test-Time Framework 
 
## Authors 
**Wuxi Ma**  
Artificial Intelligence Institute, Zhejiang Industry & Trade Vocational College  
No. 717 Fu East Road, Lucheng District, Wenzhou, 325003, Zhejiang, China  
106512084@qq.com  
 
**Hao Yang** (Corresponding Author)  
College of Systems Engineering, National University of Defense Technology  
No. 109 Deya Road, Kaifu District, Changsha, 410073, Hunan, China  
yanghao@nudt.edu.cn  
 
## Abstract 
**Challenge**: Existing Test-Time Adaptation (TTA) methods assume balanced data distributions, while real-world test data often exhibits **dynamic class imbalance**.  
**Solution**: Propose **Imbalanced Continuous Test-Time Adaptation (ICTTA)** framework with:
- Dynamic adaptive imbalanced loss function 
- Class-aware adaptation mechanism  
**Key Results**:
- Achieved **16.5%** error rate on CIFAR10-C 
- **68.1%** error rate on ImageNet-C 
- Outperformed SOTA methods by **14.2%** on imbalanced streams 
 
**Code Availability**: [GitHub Repository](https://github.com/Maxwuxi/ICTTA) 
 
**Keywords**: Test-Time Adaptation, Imbalanced Data, Continuous Adaptation, Dynamic Loss Function 
 
## 1. Introduction 
### Core Challenges 
1. **Persistent Class Imbalance**: Real-world test streams (e.g., medical imaging, autonomous driving) often exhibit dynamic class ratios 
2. **Catastrophic Forgetting**: Continuous adaptation risks losing minority class knowledge 
3. **Entropy Minimization Bias**: Conventional TTA methods overfit majority classes 
 
### Proposed Solution 
**ICTTA Framework Features**:
- Imbalanced perturbation dataset construction 
- Confidence-weighted dynamic loss function 
- Theoretical guarantees for minority class preservation 
 
## 2. Related Work 
### Key Research Areas 
| Category | Representative Methods | Limitations Addressed by ICTTA |
|----------|------------------------|--------------------------------|
| Transfer Learning | DANN, DAN | Static adaptation assumptions |
| Domain Generalization | MetaReg, MASF | No test-time adaptation |
| Test-Time Adaptation | TENT, RMT | Class imbalance ignorance |
| Class Imbalance Learning | Focal Loss, SMOTE | Offline training dependency |
 
## 3. Methodology 
### 3.1 Problem Formulation 
**Test-Time Adaptation Objective**:
```python 
θ* = argmin_θ E_{(x,y)∼D_t}[L(f_θ(x), y)]
