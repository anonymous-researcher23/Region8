# Region8
## Spatially-Aware Quantum Encoding for Flood Patch Classification (ECCV 2026)

Official implementation for the paper:

**Spatially-Aware Quantum Encoding for Flood Patch Classification in Multi-Modal Remote Sensing**

Region8 is a spatially-aware encoding designed for flood classification using multi-modal Sentinel-1 SAR and Sentinel-2 optical imagery.  
Instead of flattening patches before quantum encoding, Region8 preserves coarse spatial layout by pooling a 32×32 image patch into **8 spatial regions**, producing an **8-dimensional vector that maps directly to 8 qubits**.

---

# Contents

1. Introduction  
2. Key Highlights  
3. Dependencies  
4. Dataset Preparation  
5. Feature Extraction  
6. Training  
7. Testing  
8. Results  
9. Acknowledgements  
10. Citation  

---

# Introduction

Flood detection using satellite imagery is important for disaster response and environmental monitoring.  
Multi-modal remote sensing data from **Sentinel-1 SAR** and **Sentinel-2 optical imagery** provides complementary information for detecting flooded regions.

Most quantum learning pipelines flatten image patches and apply PCA before quantum encoding, which removes spatial information.

**Region8** introduces a spatially-aware encoding that preserves locality while remaining compatible with **8-qubit quantum models**.

Pipeline overview:

![PCA8 and Region8](./Figures/Figure1.png)

```
32×32 patch
    ↓
4 × 2 spatial grid
    ↓
8 region features
    ↓
8-dimensional vector
    ↓
Angle encoding
    ↓
8-qubit Variational Quantum Classifier
```

---

# Key Highlights

• **Spatially-aware encoding**  
Region8 preserves spatial layout instead of flattening image patches.

• **Region-to-qubit mapping**  
Each region maps directly to one qubit.

• **Local entanglement topology**  
Quantum circuit connections follow neighboring spatial regions.

• **Fair comparison with classical models**  
Both PCA8 and Region8 produce the same **8-dimensional feature vectors**.

• **Cross-dataset validation**  
Experiments conducted on:

- SEN1FLOODS11  
- DEEPFLOOD  
- SEN12MS  

---

# Dependencies

Tested on:

```
Ubuntu 20.04+
Python 3.10+
```

Required libraries:

```
numpy
pandas
scikit-learn
matplotlib
qiskit
tqdm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Download the following datasets.

### 1. DEEPFLOOD Dataset

Includes Sentinel-1 SAR, Sentinel-2 optical imagery, UAV references, and auxiliary layers.

https://figshare.com/articles/dataset/DEEPFLOOD_DATASET_High-Resolution_Dataset_for_Accurate_Flood_Mappingand_Segmentation/28328339

---

### 2. SEN1FLOODS11 Dataset

https://github.com/cloudtostreet/Sen1Floods11

---

### 3. SEN12MS Dataset

https://mediatum.ub.tum.de/1474000

---

Place datasets in:

```
data/raw/
```

Example structure:

```
data
 ├── raw
 │   ├── sen1floods11
 │   ├── deepflood
 │   └── sen12ms
```

---

# Feature Extraction

Generate feature representations.

### PCA8 baseline

```bash
python scripts/build_pca_features.py
```

### Region8 encoding

```bash
python scripts/build_region8_features.py
```

Region8 divides each patch into **8 spatial regions** and computes region-level statistics combining optical water cues and SAR scattering features.

---

# Training

Train classical and quantum models.

### Logistic Regression

```bash
python src/models/train_logreg.py
```

### RBF-SVM

```bash
python src/models/train_rbf_svm.py
```

### Variational Quantum Classifier

```bash
python src/models/train_vqc.py
```

---

# Testing

Evaluate trained models:

```bash
python scripts/evaluate_models.py
```

Metrics reported:

- AUC  
- Accuracy  
- Precision  
- Recall  
- F1 score  
- Log-loss  

Decision thresholds are selected on the validation set.

---

# Results

Region8 improves classification performance by preserving spatial locality.

Example results (SEN1FLOODS11):

| Model | Feature | AUC | F1 | Accuracy |
|------|------|------|------|------|
| RBF-SVM | Region8 | **0.9574** | **0.8693** | **0.9090** |
| VQC | Region8 | 0.9317 | 0.8184 | 0.8675 |
| Logistic Regression | Region8 | 0.9224 | 0.7834 | 0.8565 |

---

# Acknowledgements

This work uses publicly available datasets:

- SEN1FLOODS11  
- DEEPFLOOD  
- SEN12MS  

We thank the dataset creators and the remote sensing research community.

---




---
