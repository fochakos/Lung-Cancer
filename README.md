# Lung Cancer Survival Prediction

Predicts whether a lung cancer patient survives from their age, sex, cancer stage, smoking status, treatment type, BMI and other conditions.

## Data

- **Source:** [Lung Cancer dataset](https://www.kaggle.com/datasets/fochako/lung-cancer) (Kaggle), `Lung Cancer.csv`.
- **Size:** 890,000 patients, 17 columns. About 22% of patients survived.

## Approach

1. Encode categorical fields (gender, country, family history, smoking status, treatment type) and map cancer stage I–IV to 1–4.
2. Engineer treatment length in days from the diagnosis and end-of-treatment dates.
3. Explore correlations with a heatmap.
4. Stratified 80/20 split and standardized features.
5. Handle class imbalance with class weights and SMOTE oversampling.
6. Compare logistic regression (with and without SMOTE) and a random forest.

## Results (test set, 178,000 patients)

| Model | Accuracy | Recall (survived) | Macro F1 |
|---|---|---|---|
| Logistic regression, class-weighted | 0.50 | 0.51 | 0.46 |
| Logistic regression + SMOTE | 0.49 | 0.52 | 0.45 |
| Random forest + SMOTE | 0.65 | 0.24 | 0.50 |

**None of the models separate survivors from non-survivors better than chance** (macro averages around 0.50). The correlation heatmap supports this: no feature is meaningfully correlated with survival. The dataset appears to be synthetic, with features generated independently of the outcome.

## What I learned and next steps

- Accuracy can look better (0.65) while recall for the minority class gets worse. Choosing the metric matters.
- Before modeling, check whether any feature relates to the target at all. Here the correlation step already showed that none do.
- Repeating the workflow on a real clinical dataset, such as SEER-based survival data, would test whether the pipeline finds real signal.

## How to run

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
jupyter notebook Lung_Cancer_ML.ipynb
```

## Tools

Python · pandas · scikit-learn · imbalanced-learn (SMOTE) · Seaborn
