# Region8
# Spatially-Aware Quantum Encoding (Region8) for Flood Patch Classification (ECCV 2026)

This repository contains the official implementation for the ECCV 2026 submission:

**"Spatially-Aware Quantum Encoding for Flood Patch Classification in Multi-Modal Remote Sensing"**.

We propose **Region8**, a spatially-aware and quantum-compatible encoding that preserves coarse spatial layout by pooling a 32×32 multi-modal patch into a fixed **4×2 grid** (8 regions), producing an **8D vector** that maps **one-to-one onto 8 qubits**. We compare against a conventional **PCA8** flattened baseline and evaluate with:
- Logistic Regression
- RBF-SVM
- 8-qubit Variational Quantum Classifier (VQC) with locality-aligned entanglement

## Contents
1. Introduction
2. Key Highlights
3. Dependencies
4. Data Preparation
5. Train
6. Test / Evaluate
7. Reproducing Paper Tables & Figures
8. Acknowledgements
9. Citation

---

## Introduction

Most quantum remote-sensing pipelines flatten image patches and apply PCA before encoding into a VQC, which discards spatial structure.  
**Region8** instead preserves locality by explicit region pooling, maintaining interpretability under strict 8-qubit constraints.

**Region8 summary**
- Patch size: 32×32×C (Sentinel-1 VV/VH + Sentinel-2 B2/B3/B4/B8)
- Grid: 4×2 regions → 8 region features → 8 angles → 8 qubits
- Entanglement: restricted to neighboring regions (spatially aligned)

---

## Key Highlights
- **Spatially-aware quantum encoding:** Region8 preserves coarse layout without increasing feature dimension beyond 8.
- **Direct region-to-qubit mapping:** each qubit corresponds to a specific spatial region.
- **Local entanglement topology:** aligns circuit bias with image locality.
- **Fair comparison:** PCA8 and Region8 are both 8D; identical train/val/test sizes and threshold tuning.
- **Reproducibility-first:** scripts provided to regenerate features, train, and reproduce paper tables.

---

## Dependencies
Tested on:
- Ubuntu 20.04+
- Python 3.10+
- PyTorch (for any optional deep components; not required for classical baselines)
- NumPy, SciPy, scikit-learn
- pandas
- matplotlib
- Qiskit (for VQC)

### Setup (conda)
```bash
conda env create -f environment.yml
conda activate region8
