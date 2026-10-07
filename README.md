# Customer Churn Prediction — Telecom Retention Intelligence System

Final year B.Tech project. End-to-end machine learning pipeline that predicts
which telecom subscribers are at risk of churning, and quantifies the revenue
impact of acting on those predictions.

---

## Results

Evaluated on a held-out test set of 1,409 customers that the model never saw
during training or threshold tuning.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced Acc | MCC |
|---|---|---|---|---|---|---|---|
| **Random Forest** (selected) | **84.99%** | **69.90%** | **60.55%** | **71.72%** | **65.66%** | **76.73%** | **0.510** |
| Logistic Regression | 85.87% | 70.20% | 61.09% | 70.96% | 65.65% | 76.64% | 0.511 |
| Stacking Ensemble | 84.40% | 70.08% | 57.63% | 76.26% | 65.65% | 77.17% | 0.505 |
| Gradient Boosting | 85.20% | 70.37% | 55.42% | 77.53% | 64.63% | 76.57% | 0.489 |
| Extra Trees | 83.70% | 68.59% | 59.22% | 72.98% | 65.38% | 76.67% | 0.504 |

The dataset's Bayes-optimal AUC ceiling is **0.897** — computed by scoring
customers with the true generating probability. At 0.850 the model captures
about 95% of the theoretically available signal; the remaining gap is
irreducible noise representing unobservable churn drivers (competitor offers,
relocation, personal circumstances).

### Business impact on the test set

| Metric | Value |
|---|---|
| Churners correctly identified | 284 of 396 (71.7%) |
| False alarms | 185 |
| Revenue protected (24-mo CLV, 30% save rate) | $125,147 |
| Retention campaign cost ($50/contact) | $23,450 |
| **Net benefit** | **$101,697** |
| **ROI** | **434%** |

---

## Why accuracy is not reported as a headline metric

The dataset is imbalanced at 28.1% churn. A model that predicts "no churn" for
every customer scores 71.9% accuracy while identifying zero churners and
generating zero business value. Accuracy rewards the majority class and hides
complete failure on the class that matters.

This project evaluates on **precision, recall, F1, PR-AUC, balanced accuracy,
and MCC** — all of which are sensitive to minority-class performance. Accuracy
is recorded in `analytics.json` for completeness but is never used for model
selection.

---

## Pipeline

### 1. Dataset (`generate_dataset.py`)

7,043 synthetic customers generated through a **logistic link function**, so
coefficients are interpretable log-odds rather than arbitrary probability
offsets. Three design decisions make the data learnable:

- **Realistic effect sizes.** A month-to-month contract adds +0.95 to the
  log-odds (≈2.6× odds multiplier), calibrated so segment churn rates match
  published IBM Telco figures.
- **Feature interactions.** Churn risk is not additive. Fiber optic on a
  month-to-month contract carries an extra +0.45 penalty beyond either factor
  alone; a new customer with no security or support add-ons gets +0.28.
- **Correlated features.** Fiber customers pay more, long-tenure customers hold
  longer contracts, `TotalCharges` is derived from tenure × monthly charge.
  This gives tree models genuine structure to discover.

Calibration check against the real IBM Telco dataset:

| Segment | This dataset | Real IBM Telco |
|---|---|---|
| Overall churn | 28.1% | 26.5% |
| Month-to-month | 49.4% | 42.7% |
| Two year | 5.2% | 2.8% |
| Fiber optic | 40.6% | 41.9% |
| DSL | 21.0% | 19.0% |
| Electronic check | 45.3% | 45.3% |

### 2. Feature engineering (10 derived features)

| Feature | Hypothesis it encodes |
|---|---|
| `AvgChargePerMonth` | Historical spend level |
| `ChargeIncreaseRatio` | Current bill vs historical average — detects price rises |
| `NumServices` | Breadth of engagement with the provider |
| `HasProtection` | Security/support add-ons create switching friction |
| `TenureBand` | Lifecycle stage as an ordinal |
| `IsNewCustomer` | Inside the high-risk onboarding window |
| `ContractLength` | Commitment strength, ordinal |
| `AutoPayment` | Payment friction reduces passive churn |
| `HighRiskCombo` | The known worst-case interaction, made explicit |
| `ChargePerService` | Perceived value for money |

### 3. Class imbalance — SMOTE

`imbalanced-learn` is unavailable in the target environment, so SMOTE is
implemented directly from Chawla et al. (2002): for each minority sample, find
its *k* = 5 nearest minority neighbours and generate synthetic points by
interpolating along the connecting line segments.

SMOTE is applied **to the training fold only**. Synthetic samples never enter
the validation or test sets — doing so would leak information and inflate
scores.

### 4. Three-way split

```
Train (64%)  →  SMOTE  →  model fitting + hyperparameter search
Validation (16%)        →  decision threshold tuning
Test (20%)              →  final evaluation, touched exactly once
```

Tuning the threshold on the test set would be a form of leakage. It is tuned on
validation and then applied unchanged to test.

### 5. Hyperparameter search

`RandomizedSearchCV` with stratified 5-fold cross-validation, scored on ROC-AUC,
across Random Forest, Gradient Boosting, and Extra Trees. A stacking ensemble
combines all three tuned tree models under a logistic regression meta-learner.

### 6. Threshold tuning

The default 0.5 cutoff is wrong for imbalanced data. The threshold is selected
by scanning the precision-recall curve for the point maximising F1 on the churn
class. The selected model uses **0.425**, not 0.5.

### 7. Explanation — permutation importance

Model-agnostic importance computed on the test set over 5 repeats. Unlike
tree-based `feature_importances_`, permutation importance is not biased toward
high-cardinality features and measures actual predictive contribution.

Top features: `ContractLength` (0.046), `Contract` (0.038), `TenureBand`
(0.009), `AvgChargePerMonth` (0.006), `ChargePerService` (0.006).

---

## Running it

```bash
pip install flask pandas numpy scikit-learn joblib

python generate_dataset.py    # creates data/telco_churn.csv
python train_model.py         # trains, tunes, saves models/ + analytics.json
python app.py                 # serves dashboard at http://localhost:5050
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/api/predict` | POST | Score one customer — returns probability, risk band, recommendations |
| `/api/analytics` | GET | Precomputed EDA and model metrics |
| `/api/customers` | GET | Paginated customer records |
| `/api/batch_predict` | POST | Score a cohort, returned ranked by risk |

Predictions apply the tuned threshold from `models/decision_threshold.pkl`.
Each recommendation ships with the segment statistic that justifies it.

---

## Files

```
churn_project/
├── generate_dataset.py     # calibrated data generator + ceiling diagnostic
├── train_model.py          # full training pipeline
├── app.py                  # Flask REST API
├── data/
│   ├── telco_churn.csv
│   └── analytics.json
└── models/
    ├── best_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── feature_columns.pkl
    └── decision_threshold.pkl
```

---

## Limitations

- The dataset is synthetic. Segment rates are calibrated to published IBM Telco
  figures, but real customer behaviour contains patterns no generator captures.
- No temporal features. Real churn prediction benefits substantially from
  month-over-month usage trends and support-ticket history.
- Business impact figures depend on assumed CLV (24 months), retention success
  rate (30%), and contact cost ($50). These are stated assumptions, not measured
  values, and would need calibration against a real retention programme.
