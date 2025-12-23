// ...existing code...

# Insurance Modeling — ML Notebook

This repository contains a Jupyter notebook that implements a complete machine-learning pipeline for an insurance purchase prediction problem.

Project files
- notebooks/ML-Project.ipynb — full ML pipeline and experiments
- data/
  - insurance_3_TRAINING.csv — training set
  - insurance_3_SCORE.csv — scoring set
- README.md
- requirements.txt (optional)

Overview
- The notebook loads training and scoring CSVs, performs feature engineering (target encoding, binning, one‑hot encoding), handles outliers, scales numeric features, and addresses severe class imbalance (~18% positive) with class weights and stratified CV.
- Multiple models are trained and compared: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN, plus custom neural-network experiments (manual MLP implementations and an Adam optimizer variant).
- Final scoring produces a CSV with predicted label and buy probability.

Quick start (macOS)
1. Create & activate virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or, if no requirements file:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow jupyter
   ```
3. Place dataset CSVs in data/ (see Project files).
4. Open and run the notebook:
   - Open notebooks/ML-Project.ipynb in Jupyter or VS Code Jupyter extension and run cells top to bottom.

Key findings (summary)
- Severe class imbalance: use class weights, stratified splits, and metrics that account for imbalance (AUC, F1).
- Target encoding for high‑cardinality fields (region) reduces dimensionality while preserving signal.
- Neural network (Adam) and LightGBM delivered the strongest results in the notebook experiments; tune hyperparameters and persist preprocessors for production use.
- Accuracy is not a reliable metric here — prefer AUC and F1.

Recommended next steps
- Persist fitted preprocessors (scalers, encoders) and save trained model artifacts.
- Perform hyperparameter optimization (GridSearch, RandomSearch, or Optuna) for top models.
- Wrap preprocessing + model inference into a reproducible scoring script or small API.
- Add unit/integration tests and CI to validate preprocessing and scoring outputs.

Usage notes
- Ensure consistent preprocessing between train and score datasets (the notebook aligns columns after one-hot encoding).
- Use stratified cross-validation and class-weight or scale_pos_weight for imbalanced learners.

License & contact
- No license file provided. For questions, inspect notebook comments and code cells.
```// filepath: /Users/hallaodeh/python-projects/Machine Learning REPO/README.md
// ...existing code...

# Insurance Modeling — ML Notebook

This repository contains a Jupyter notebook that implements a complete machine-learning pipeline for an insurance purchase prediction problem.

Project files
- notebooks/ML-Project.ipynb — full ML pipeline and experiments
- data/
  - bzan6357_insurance_3_TRAINING.csv — training set
  - bzan6357_insurance_3_SCORE.csv — scoring set
- README.md
- requirements.txt (optional)

Overview
- The notebook loads training and scoring CSVs, performs feature engineering (target encoding, binning, one‑hot encoding), handles outliers, scales numeric features, and addresses severe class imbalance (~18% positive) with class weights and stratified CV.
- Multiple models are trained and compared: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN, plus custom neural-network experiments (manual MLP implementations and an Adam optimizer variant).
- Final scoring produces a CSV with predicted label and buy probability.

Quick start (macOS)
1. Create & activate virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or, if no requirements file:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tensorflow jupyter
   ```
3. Place dataset CSVs in data/ (see Project files).
4. Open and run the notebook:
   - Open notebooks/ML-Project.ipynb in Jupyter or VS Code Jupyter extension and run cells top to bottom.

Key findings (summary)
- Severe class imbalance: use class weights, stratified splits, and metrics that account for imbalance (AUC, F1).
- Target encoding for high‑cardinality fields (region) reduces dimensionality while preserving signal.
- Neural network (Adam) and LightGBM delivered the strongest results in the notebook experiments; tune hyperparameters and persist preprocessors for production use.
- Accuracy is not a reliable metric here — prefer AUC and F1.

Recommended next steps
- Persist fitted preprocessors (scalers, encoders) and save trained model artifacts.
- Perform hyperparameter optimization (GridSearch, RandomSearch, or Optuna) for top models.
- Wrap preprocessing + model inference into a reproducible scoring script or small API.
- Add unit/integration tests and CI to validate preprocessing and scoring outputs.

Usage notes
- Ensure consistent preprocessing between train and score datasets (the notebook aligns columns after one-hot encoding).
- Use stratified cross-validation and class-weight or scale_pos_weight for imbalanced learners.

License & contact
- No license file provided. For questions, inspect notebook comments and code cells.