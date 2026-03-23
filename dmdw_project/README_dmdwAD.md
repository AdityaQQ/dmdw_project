# dmdwAD — DMDW Focused ML Project
## Commercial Housing Dataset

## Run
```
pip install -r requirements.txt
python dmdwAD.py
```

## Algorithms & Figures Generated (35 total)

### KNN (fig01–fig06)
- fig01 — Metrics vs K (Accuracy, F1, Precision, Recall for K=1 to 15)
- fig02 — Confusion Matrix (large, annotated)
- fig03 — ROC Curves (one-vs-rest per class)
- fig04 — Per-Class F1 Score
- fig05 — Learning Curve (train vs CV)
- fig06 — Final Metrics Summary bar chart

### Linear Regression (fig07–fig12)
- fig07 — Actual vs Predicted scatter
- fig08 — Residuals Plot
- fig09 — Residual Distribution histogram
- fig10 — OLS vs Ridge vs Lasso comparison + Alpha sweep
- fig11 — Feature Coefficients (which features drive price)
- fig12 — Percentage Error + Q-Q analysis

### K-Means (fig13–fig18)
- fig13 — Elbow Curve (Inertia vs K)
- fig14 — Silhouette Score vs K
- fig15 — PCA 2D Cluster Scatter + True Labels comparison
- fig16 — Cluster Sizes bar chart
- fig17 — Feature Means Heatmap per Cluster
- fig18 — Elbow + Silhouette Combined

### K-Medoids (fig19–fig24)
- fig19 — Silhouette vs K
- fig20 — Cluster Scatter vs K-Means comparison
- fig21 — Cluster Sizes
- fig22 — Medoid House Feature Details
- fig23 — Feature Means Heatmap per Cluster
- fig24 — K-Means vs K-Medoids side by side

### DBSCAN (fig25–fig30)
- fig25 — eps Parameter Sweep (clusters, noise, silhouette)
- fig26 — Cluster Scatter + Noise Highlighted
- fig27 — Noise House Feature Analysis
- fig28 — eps Value Comparison (4 panels)
- fig29 — Cluster Summary + Price Boxplot
- fig30 — min_samples Parameter Sweep

### Confusion Matrix (fig31–fig34)
- fig31 — KNN Confusion Matrix (large detailed)
- fig32 — Normalised Confusion Matrix (%)
- fig33 — Per-Class Binary Breakdown (TP/FP/TN/FN per category)
- fig34 — Error Matrix + Precision/Recall/F1 per class

### Summary (fig35)
- fig35 — All Algorithms Combined Summary
