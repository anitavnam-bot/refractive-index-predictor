# Refractive Index Predictor

## Overview

This project predicts the refractive index of crystalline materials using:

* Chemical composition (formula)
* Crystal structure (CIF file)

The model combines:

1. Magpie composition descriptors
2. CGCNN (Crystal Graph Convolutional Neural Network) structural embeddings
3. An XGBoost regression model

The final deployable model achieves approximately:

* MAE ≈ 1.36
* RMSE ≈ 2.03
* R² ≈ 0.775

using only composition and crystal structure.

---

# Required Inputs

The predictor requires:

1. Chemical formula
2. CIF file describing the crystal structure

Example:

Formula:

```text
TiCuSiAs
```

Structure:

```text
JVASP-90856.cif
```

---

# Environment Setup

Create and activate the environment:

```bash
source ~/venvs/alignn23/bin/activate
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Running the Predictor

Example:

```bash
python predict_ri.py \
  --formula "TiCuSiAs" \
  --cif JVASP-90856.cif
```

Output:

```text
Predicted refractive index: 8.5012
```

---

# How the Pipeline Works

Input:

```text
Formula + CIF
```

↓

Magpie descriptors

↓

CGCNN structural embedding (128-dimensional learned representation)

↓

Multimodal XGBoost model

↓

Predicted refractive index

---

# Files

## Prediction Script

```text
predict_ri.py
```

Main user-facing entry point.

---

## Deployable Model

```text
deployable_refractive_index_model.joblib
```

Trained XGBoost model.

---

## Feature Schema

```text
deployable_feature_columns.joblib
```

Ensures prediction features match training features.

---

## CGCNN Checkpoint

```text
model_best.pth.tar
```

Used to generate crystal structure embeddings.

---

# Notes

The model does NOT require:

* DFT bandgap
* Formation energy
* Energy above hull
* Density

Only composition and crystal structure are required.

This makes the model suitable for screening newly generated materials without running expensive DFT calculations.

