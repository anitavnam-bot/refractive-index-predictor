
This repository contains a multimodal machine learning pipeline for predicting refractive index from:

* Chemical composition (formula)
* Crystal structure (CIF)

The final deployable model combines:

* Magpie composition descriptors
* CGCNN structural embeddings
* XGBoost regression

## Deployable Package

See:

```text
deployment/
```

for the standalone predictor.

Run:

```bash
python predict_ri.py \
    --formula "TiCuSiAs" \
    --cif JVASP-90856.cif
```

to obtain a refractive-index prediction.

