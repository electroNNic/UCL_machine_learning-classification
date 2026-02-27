# Classification (Model Benchmark + CV + Tuning)

This repository contains a machine learning experiment for **binary classification** on a rice dataset.

The overall pipeline is applicable to most classification datasets (including datasets with two or more classes).

The workflow includes **data preprocessing**, **10-fold stratified cross-validation**, **model benchmarking**, **confusion matrix analysis**, and **hyperparameter tuning**.

## Dataset

- File name: `rice-final2.csv`
- Format:
  - Features: all columns except the last one
  - Label: the last column (`class1` / `class2`)

> Note: The notebook assumes `rice-final2.csv` is in the same directory as `UCL_ml.ipynb`.

## Methods Overview

### Preprocessing
Missing values/Feature scaling/Label encoding

### Part 1: 10-fold Stratified Cross-Validation
Evaluate average CV accuracy for:
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree (entropy)
- Bagging (DT as base estimator)
- AdaBoost (DT as base estimator)
- Gradient Boosting

**Visualization & analysis (saved under `png/`):**
- Accuracy distribution boxplots (10-fold scores)
- Confusion matrices
- Metrics computed from confusion matrix: Accuracy / Precision / Recall / F1

### Part 2: Cross-Validation with Hyperparameter Tuning (GridSearchCV)
- **KNN**: grid search over
  - `n_neighbors` (k): [1, 3, 5, 7]
  - `p`: [1, 2]
- **Random Forest**: grid search over
  - `n_estimators`: [10, 30, 60, 100]
  - `max_leaf_nodes`: [6, 12]
  - with `criterion='entropy'`, `max_features='sqrt'`, `random_state=0`

Also includes **parameter sensitivity analysis** plots for KNN and RF (saved under `png/`).
