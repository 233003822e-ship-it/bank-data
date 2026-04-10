# Term Deposit Uptake Prediction with Logistic Regression

Term Deposit Uptake Prediction with Stacking Ensemble
Objective

Banks invest heavily in telemarketing to promote term deposits. This project applies a Stacking Ensemble Learning approach—combining multiple base models—to historical campaign data so that the bank can more accurately estimate which customers are most likely to subscribe to a term deposit.

Dataset Details

The data comes from the UCI Bank Marketing dataset — 45,211 customer records across 17 fields.
| Field | Kind | What it captures |
|-------|------|------------------|
| age | Numeric | Customer's age in years |
| job | Categorical | Employment category |
| marital | Categorical | Marriage status |
| education | Categorical | Highest qualification |
| default | Binary | Credit default flag |
| balance | Numeric | Mean annual balance (EUR) |
| housing | Binary | Housing-loan flag |
| loan | Binary | Personal-loan flag |
| contact | Categorical | Communication channel used |
| day | Numeric | Calendar day of last call |
| month | Categorical | Calendar month of last call |
| duration | Numeric | Length of last call (sec) |
| campaign | Numeric | Contacts made in this campaign |
| pdays | Numeric | Days elapsed since prior campaign contact |
| previous | Numeric | Contacts made in earlier campaigns |
| poutcome | Categorical | Result of the prior campaign |
| **y** | **Binary** | **Opened a term deposit? yes / no** |

## Methodology

Phase 1 — Exploration:

Inspected data quality (verified absence of missing values and duplicates). Visualised feature distributions using histograms for numerical variables and bar charts for categorical variables. Analysed relationships between predictors and the target variable through stacked proportion plots and box plots. Generated a correlation heatmap to identify linear relationships and highlight potentially important predictors.

Special attention was given to class imbalance (~88% “no”, ~12% “yes”), as this significantly impacts model performance and evaluation strategy.

Phase 2 — Preprocessing:

Transformed categorical variables into numerical representations:

Binary features (e.g., yes/no) were manually encoded (0/1)
Multi-class categorical variables were encoded using LabelEncoder

Split the dataset into training (75%) and testing (25%) sets using stratified sampling to preserve class distribution.

Applied feature scaling using StandardScaler to normalise numerical features, ensuring compatibility with models sensitive to feature magnitude.

Phase 3 — Modelling (Stacking Ensemble):

Implemented a Stacking Ensemble model, which combines multiple base learners to improve predictive performance.

🔹 Base Models:
Logistic Regression
Decision Tree
Random Forest
🔹 Meta-Model:
Logistic Regression used as the final estimator to combine predictions from base models.

Two configurations were tested:

Standard Stacking Model — trained on original data
Balanced Stacking Model — incorporated class imbalance handling (via class weights or resampling)

This approach leverages:

Linear relationships (Logistic Regression)
Non-linear patterns (Decision Tree)
Robust ensemble learning (Random Forest)
Phase 4 — Evaluation:

Evaluated model performance using multiple metrics:

Accuracy
Precision
Recall
F1-score
ROC-AUC

Generated confusion matrices and plotted the ROC curve to assess classification performance visually.

To ensure model robustness and generalisation, performed 5-fold stratified cross-validation, maintaining consistent class distribution across folds.

1. **Imbalance effect** — ~88 % of customers did not subscribe, so high accuracy alone does not mean the model is useful; recall and AUC matter more.
2. **Call duration dominates** — the length of the last marketing call is, by far, the strongest predictor of conversion.
3. **Prior success recurs** — a customer who subscribed in a previous campaign is very likely to subscribe again.
4. **Demographic sweet spots** — retirees, students, single individuals, and those with a tertiary education show above-average conversion.
5. **Channel matters** — cellular contact outperforms telephone and unknown.
6. **Weighted model trade-off** — it recovers substantially more true positives (higher recall) while giving up only a modest amount of precision, making it the better choice when missing a potential subscriber is costly.
7. **Consistent across folds** — cross-validation scores exhibit low variance, confirming the model is not over-fitting.

## Files

```
.
├── bank-data/
│   └── bank-full.csv               # Source data
├── term_deposit_analysis.py         # Full analysis script
├── requirements.txt                 # pip dependencies
├── README.md                        # Documentation
└── images/                          # Generated charts (after running)
```

## Execution

```bash
pip install -r requirements.txt
python term_deposit_analysis.py
```

Charts are written to the `images/` directory automatically.

## Libraries Used

Python 3.8+, pandas, numpy, matplotlib, seaborn, scikit-learn
