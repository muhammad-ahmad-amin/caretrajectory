# CareTrajectory — Baseline Model Benchmark Report

Generated across evaluated models on patient-level holdout validation set.

| model_name          |   AUROC |   AUPRC |   F1_Score |   Sensitivity_Recall |   Specificity |   Precision_PPV |   PhysioNet_Utility |   Brier_Score |   threshold |
|:--------------------|--------:|--------:|-----------:|---------------------:|--------------:|----------------:|--------------------:|--------------:|------------:|
| Logistic Regression |  1      |  1      |     1      |                    1 |         1     |          1      |              1      |        0.0007 |      0.956  |
| XGBoost             |  0.9995 |  0.9545 |     0.9767 |                    1 |         0.999 |          0.9545 |              0.9976 |        0.2257 |      0.5249 |

### Metric Explanations:
- **AUROC**: Overall discrimination across all potential alarm cutoffs.
- **AUPRC**: Primary evaluation metric under extreme clinical class imbalance.
- **Sensitivity_Recall**: Percentage of true deteriorating patients flagged.
- **Specificity**: Percentage of non-deteriorating patients spared false alarms.
- **PhysioNet_Utility**: Early prediction reward versus alert fatigue penalty.
