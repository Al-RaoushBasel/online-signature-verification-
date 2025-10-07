# Online Signature Verification Project

## Overview
This repository collects a series of Jupyter notebooks that walk through the design, evaluation, and progressive improvement of machine-learning pipelines for online signature verification. The work explores classical feature engineering on stylus trajectory data (position, pressure, altitude, and azimuth), multiple classifier families, and ensemble techniques to distinguish genuine signatures from skilled forgeries. In addition to exploratory analysis, the notebooks document incremental experiments, model tuning, and cross-dataset comparisons that can be reproduced locally or in Google Colab.

## Repository structure
| File | Description |
| --- | --- |
| `1.ipynb` | Exploratory data analysis of the MCYT-style training set, including distribution checks, trajectory visualization, pressure trends, directional angles, and correlation heatmaps. |
| `2.ipynb` | Baseline ML pipeline that engineers summary statistics per signature and compares logistic regression, SVM, and random forest classifiers with and without spatial (`X`, `Y`) features. |
| `3.ipynb` | Production-style pipeline with refreshed preprocessing, scaling, and evaluation of Random Forest vs. KNN, including Equal Error Rate (EER) estimation and confusion matrices. |
| `4.ipynb` | Experiments with XGBoost, a multilayer perceptron (MLP), and a soft voting ensemble evaluated via accuracy, F1 score, EER, and confusion matrices. |
| `5.ipynb` | Hyper-parameter tuning (GridSearchCV) for XGBoost and MLP, plus weighted voting and stacking ensembles aimed at boosting verification accuracy. |
| `6.ipynb` | Comparative study of four models across five signature datasets (MCYT, Chinese, Dutch, German, SVC) with consolidated metrics tables and charts. |
| `OSV-1.pdf` | Reference material describing signature verification concepts and experiments. |
| `Online Signature Verification Using Machine Learning .pdf` | Project report summarizing methodology and findings. |

## Highlights
* **Feature engineering focus** – Notebook 2 shows how summary statistics for pressure, altitude, and azimuth drive baseline performance, with Random Forest benefiting most when spatial coordinates are included (`F1 = 0.605`, `accuracy = 0.705`).
* **Verification metrics** – Notebook 3 emphasises verification-specific evaluation such as Equal Error Rate; the Random Forest model achieves `accuracy = 0.837`, `F1 = 0.847`, and `EER = 0.159`.
* **Model scaling** – Later notebooks introduce gradient boosting, neural networks, voting/stacking ensembles, and per-dataset benchmarking to understand generalisation across acquisition protocols.

## Getting started
1. **Environment** – Use Python 3.9+ with common data-science libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`). Installing `jupyter` is recommended for local execution; alternatively, open the notebooks directly in Google Colab.
2. **Data access** – The notebooks expect signature datasets stored in Google Drive (e.g., `mcytTraining.txt`). Upload the raw files to your drive or adjust the paths under the data-loading cells to point to local copies.
3. **Execution order** – Work through the notebooks numerically (`1.ipynb` → `6.ipynb`) to follow the narrative from exploratory analysis through increasingly sophisticated models.
4. **Reproducibility** – Many notebooks include random seeds, but results may vary slightly between runs because of stochastic model components. When tuning or stacking models, enable GPU acceleration in Colab if available for faster training.

## Extending the project
* Integrate temporal dynamics (e.g., velocity, acceleration) in feature engineering to capture signing behaviour over time.
* Experiment with deep learning architectures for sequence modelling such as recurrent or transformer-based networks.
* Apply cross-validation and class-imbalance techniques (resampling or class weighting) to improve robustness on imbalanced datasets.
* Automate the workflow by exporting reusable preprocessing and model-training utilities into Python modules.

## Citation
If you build on this work, please cite the MCYT and other signature datasets according to their respective licenses and acknowledge the original project report included in this repository.
