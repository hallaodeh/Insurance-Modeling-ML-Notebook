# EDA Project — Customer Churn Analysis

Summary
- Exploratory data analysis and preprocessing on a telco churn dataset.
- Goal: prepare data for downstream modeling and identify key issues (collinearity, distributions) that affect model design.

Dataset
- Source: `churn_dataset.csv`
- Notes: `phone` treated as an ID column; `area_code` converted to categorical.

Environment & Dependencies
- Python (3.8+ recommended)
- Core packages: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels
- Example install:
  pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

Files
- `EDA-Project.ipynb` — analytical notebook containing preprocessing, EDA, visualizations and statistical tests.
- `churn_dataset.csv` — input data (not included here).

Preprocessing
- Created working copy `df_churn1`.
- Converted `churn` and `vmail_plan` to numeric binary flags (0/1).
- Standardized `day_mins` (saved as `dat_mins`) using z-score.
- Replaced `night_mins` with z-score standardized values.
- Min–Max normalized `intl_calls`.
- Generated dummy/flag variables for `area_code` and produced `df_churn2`.

Exploratory Analysis & Visualizations
- Converted `intl_plan` to binary for analysis.
- Computed correlation matrix and plotted a half-matrix heatmap to detect collinearity.
- Created a 2x2 scatterplot panel (day_mins vs day_charge, eve_mins, night_mins, intl_mins) colored by churn.
- Plotted histograms for standardized `night_mins` and normalized `intl_calls`.

Statistical Tests (Train/Test Split)
- Performed train/test split on original data (test_size=0.4, random_state=30).
- Two-sample t-test on `eve_calls` (H0: means equal, α=0.05) → p-value > α => fail to reject H0.
- Two-proportion z-test on `vmail_plan` (H0: proportions equal, α=0.05) → p-value > α => fail to reject H0.

Key Findings & Recommendations
- Perfect collinearity detected between minutes and corresponding charges:
  - day_mins ↔ day_charge
  - eve_mins ↔ eve_charge
  - intl_mins ↔ intl_charge
  - night_mins ↔ night_charge
  Recommendation: drop one variable from each pair (prefer keeping `_mins` for interpretability).
- High correlation between `vmail_plan` and `vmail_message` (≈ 0.957). Investigate before including both in models.
- `night_mins` approximately normal after z-score scaling; `intl_calls` is right-skewed after min-max scaling.
- Train/test split appears balanced for tested variables.

Next Steps
- Feature selection: drop collinear `_charge` variables; consider removing or combining highly correlated features.
- Encode remaining categorical variables and proceed to model building (logistic regression, tree-based models).
- Evaluate models with cross-validation and inspect feature importance/coefficients to validate preprocessing decisions.
- Document reproducible pipeline and save preprocessing transformers for production use.

How to run
1. Ensure dependencies are installed.
2. Place `churn_dataset.csv` alongside the notebook.
3. Open and run `EDA-Project.ipynb` in JupyterLab / Jupyter Notebook.

License & Contact
- No license specified. For questions about the analysis, refer to the notebook comments and code cells.