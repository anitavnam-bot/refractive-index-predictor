# Refractive Index Predictor

A deployable machine-learning pipeline for predicting the refractive index of crystalline materials from:

- Chemical formula
- Crystal structure (CIF)

The model combines:

- Magpie composition descriptors
- CGCNN structural embeddings
- XGBoost regression

## Performance

| Model | R² |
|---------|---------|
| Descriptor-only | ~0.665 |
| CGCNN-only | ~0.664 |
| Descriptor + CGCNN embedding | ~0.775 |

## Usage

See:

```text
ri_predictor_release.zip
