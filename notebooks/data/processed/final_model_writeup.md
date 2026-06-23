
Methodology and Results Summary

We evaluated three models for refractive-index prediction. The fair descriptor-only baseline used engineered composition and property descriptors. The structure-only baseline used CGCNN predictions from crystal graphs. The multimodal model combined the descriptor features with CGCNN latent structural embeddings.

Results:
- Descriptor-only (fair): MAE = 1.730, RMSE = 2.496, R² = 0.665
- CGCNN-only: MAE = 1.714, RMSE = 2.488, R² = 0.664
- Multimodal fusion: MAE = 1.294, RMSE = 1.976, R² = 0.786

The multimodal model performed best across all metrics, reducing MAE and RMSE while increasing R² relative to both single-modality baselines. This indicates that learned structural information from CGCNN adds predictive value beyond descriptor-based features alone.

Best model:
- multimodal_fusion
- R² = 0.786
