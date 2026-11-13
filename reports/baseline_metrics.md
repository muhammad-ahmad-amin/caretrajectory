# CareTrajectory — Baseline Model Benchmark Report

Generated across evaluated models on patient-level holdout validation set.

| model_name | AUROC | AUPRC | F1_Score | Sensitivity_Recall | Specificity | Precision_PPV | PhysioNet_Utility | Brier_Score | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.9704 | 0.7856 | 0.8 | 0.6667 | 1.0 | 1.0 | 0.3333 | 0.0326 | 0.0078 |
| Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0009 | 0.8774 |

### Metric Explanations:
- **AUROC**: Overall discrimination across all potential alarm cutoffs.
- **AUPRC**: Primary evaluation metric under extreme clinical class imbalance.
- **Sensitivity_Recall**: Percentage of true deteriorating patients flagged.
- **Specificity**: Percentage of non-deteriorating patients spared false alarms.
- **PhysioNet_Utility**: Early prediction reward versus alert fatigue penalty.
