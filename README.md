# Online Signature Verification

An ML pipeline that distinguishes **genuine handwritten signatures from skilled forgeries** using stylus trajectory data (position, pressure, altitude, azimuth). Trained and compared multiple classifiers across 5 international signature datasets.

---

## Results

**MCYT Dataset Benchmark:**

| Model | Accuracy | F1 Score | EER |
|-------|----------|----------|-----|
| **Random Forest** | **92.3%** | **93.7%** | **0.075** |
| MLP (Neural Net) | 92.0% | 93.3% | 0.078 |
| XGBoost | 90.8% | 92.1% | 0.095 |
| KNN | 90.3% | 91.3% | 0.110 |

Evaluated across **5 datasets**: MCYT, Chinese, Dutch, German, and SVC. Random Forest and MLP consistently performed at the top across all datasets.

---

## How It Works

```
Raw Signature Data (x, y, pressure, altitude, azimuth)
        |
        v
Feature Engineering (summary statistics, spatial features)
        |
        v
Model Training (SVM, RF, XGBoost, MLP)
        |
        v
Ensemble (Voting + Stacking) --> Genuine / Forgery
```

1. **Feature Engineering** — Extract statistical features from pressure, altitude, azimuth, and spatial coordinates per signature sample
2. **Model Training** — Train and compare Logistic Regression, SVM, Random Forest, KNN, XGBoost, and MLP classifiers
3. **Ensemble Methods** — Combine top models via soft voting and stacking for improved accuracy
4. **Cross-Dataset Evaluation** — Benchmark generalization across 5 different signature datasets with different acquisition protocols

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=fff)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=fff)
![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=fff)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=fff)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=fff)

---

## Project Structure

```
├── 1.ipynb    Exploratory data analysis (distributions, trajectory viz, correlation heatmaps)
├── 2.ipynb    Baseline pipeline (Logistic Regression, SVM, Random Forest)
├── 3.ipynb    Production pipeline with EER evaluation (RF vs KNN)
├── 4.ipynb    XGBoost, MLP, and soft voting ensemble
├── 5.ipynb    Hyperparameter tuning (GridSearchCV) + stacking ensemble
├── 6.ipynb    Cross-dataset benchmarking (MCYT, Chinese, Dutch, German, SVC)
└── *.pdf      Project report and reference materials
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Al-RaoushBasel/online-signature-verification-.git
cd online-signature-verification-

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

# Run notebooks in order
jupyter notebook 1.ipynb
```

Notebooks expect signature datasets in Google Drive paths — adjust the data-loading cells to point to your local copies, or open directly in [Google Colab](https://colab.research.google.com/).

---

## Key Takeaways

- **Pressure and azimuth features** are the strongest discriminators between genuine and forged signatures
- **Random Forest** benefits most from including spatial coordinates (X, Y)
- **Stacking ensembles** outperform individual models across all datasets
- **Cross-dataset generalization** varies significantly — models trained on MCYT don't transfer well to Chinese signatures without retraining
