"""
train_model.py — Churn prediction training pipeline.

Pipeline stages:
  1. Load dataset, engineer domain features
  2. Encode + scale, stratified hold-out split
  3. SMOTE on training fold only (manual implementation)
  4. Hyperparameter search (RandomizedSearchCV, stratified 5-fold, scored on ROC-AUC)
  5. Train 4 base models + a stacking ensemble
  6. Tune decision threshold on a validation split (never on test)
  7. Final evaluation on untouched hold-out test set
  8. Permutation importance for model explanation
  9. Persist artefacts + analytics.json
"""
import os
import numpy as np
import pandas as pd
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, StackingClassifier)
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score,
                             precision_recall_curve, balanced_accuracy_score,
                             matthews_corrcoef, brier_score_loss)

SEED = 42
np.random.seed(SEED)
BASE = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    """Domain-driven features. Each one encodes a hypothesis about churn."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Average spend per month of tenure — detects customers whose bill grew
    df['AvgChargePerMonth'] = np.where(df['tenure'] > 0,
                                       df['TotalCharges'] / df['tenure'],
                                       df['MonthlyCharges'])

    # Is current bill above their historical average? (recent price increase)
    df['ChargeIncreaseRatio'] = df['MonthlyCharges'] / (df['AvgChargePerMonth'] + 1e-6)

    # Count of value-added services subscribed
    svc_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = sum((df[c] == 'Yes').astype(int) for c in svc_cols)

    # Protective services specifically (security + support = stickiness)
    df['HasProtection'] = ((df['OnlineSecurity'] == 'Yes') |
                           (df['TechSupport'] == 'Yes')).astype(int)

    # Tenure lifecycle stage
    df['TenureBand'] = pd.cut(df['tenure'], [-1, 6, 12, 24, 48, 72],
                              labels=[0, 1, 2, 3, 4]).astype(int)

    # Is the customer still inside the high-risk onboarding window?
    df['IsNewCustomer'] = (df['tenure'] < 12).astype(int)

    # Contract commitment as an ordinal (0 = none, 2 = strongest)
    df['ContractLength'] = df['Contract'].map(
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    # Automatic payment = lower friction = stickier
    df['AutoPayment'] = df['PaymentMethod'].isin(
        ['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)

    # The known high-risk combination, as an explicit flag
    df['HighRiskCombo'] = ((df['Contract'] == 'Month-to-month') &
                           (df['InternetService'] == 'Fiber optic') &
                           (df['tenure'] < 12)).astype(int)

    # Price per service — are they getting value for money?
    df['ChargePerService'] = df['MonthlyCharges'] / (df['NumServices'] + 1)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. SMOTE (manual — imbalanced-learn unavailable offline)
# ═══════════════════════════════════════════════════════════════════════════
def smote(X_min, n_synthetic, k=5, seed=SEED):
    """Chawla et al. (2002): interpolate between minority samples and their
    k nearest minority neighbours to create synthetic examples."""
    rng = np.random.RandomState(seed)
    X_min = np.asarray(X_min)
    n_min = len(X_min)
    k = min(k, n_min - 1)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, nbrs = nn.kneighbors(X_min)

    out = np.zeros((n_synthetic, X_min.shape[1]))
    for i in range(n_synthetic):
        idx = rng.randint(n_min)
        nbr = nbrs[idx, rng.randint(1, k + 1)]
        out[i] = X_min[idx] + rng.rand() * (X_min[nbr] - X_min[idx])
    return out


def apply_smote(X, y, seed=SEED):
    X, y = np.asarray(X), np.asarray(y)
    X_min, X_maj = X[y == 1], X[y == 0]
    n_gen = len(X_maj) - len(X_min)
    if n_gen <= 0:
        return X, y
    X_syn = smote(X_min, n_gen, seed=seed)
    X_bal = np.vstack([X_maj, X_min, X_syn])
    y_bal = np.concatenate([np.zeros(len(X_maj)), np.ones(len(X_min)), np.ones(n_gen)])
    perm = np.random.RandomState(seed).permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOAD + PREPARE
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STAGE 1 — Load dataset and engineer features")
print("=" * 72)

df_raw = pd.read_csv(f'{BASE}/data/telco_churn.csv')
df = engineer_features(df_raw)
df = df.drop('customerID', axis=1)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop('Churn', axis=1)
y = df['Churn']
feature_names = list(X.columns)

print(f"Rows: {len(df)}   Features: {X.shape[1]} "
      f"({X.shape[1] - 19} engineered)   Churn rate: {y.mean()*100:.2f}%")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# Three-way split: train / validation (threshold tuning) / test (final, untouched)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=SEED, stratify=y_temp)

print(f"Train: {len(X_train)}   Validation: {len(X_val)}   Test: {len(X_test)}")
print("  (threshold tuned on validation — test set never touched until the end)")

# SMOTE on training fold only
X_train_bal, y_train_bal = apply_smote(X_train, y_train)
print(f"\nSMOTE: train churn {y_train.mean()*100:.1f}% -> {y_train_bal.mean()*100:.1f}% "
      f"(+{len(X_train_bal) - len(X_train)} synthetic samples)")
X_train_bal = pd.DataFrame(X_train_bal, columns=feature_names)


# ═══════════════════════════════════════════════════════════════════════════
# 4. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STAGE 2 — Hyperparameter search (RandomizedSearchCV, 5-fold, ROC-AUC)")
print("=" * 72)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

search_spaces = {
    'Random Forest': (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {'n_estimators': [150, 250],
         'max_depth': [10, 16],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4],
         'max_features': ['sqrt', 'log2']}
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=SEED),
        {'n_estimators': [150, 250],
         'learning_rate': [0.03, 0.05, 0.1],
         'max_depth': [3, 4],
         'subsample': [0.8, 0.9, 1.0],
         'min_samples_leaf': [1, 5, 10]}
    ),
    'Extra Trees': (
        ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
        {'n_estimators': [150, 250],
         'max_depth': [12, None],
         'min_samples_split': [2, 5, 10],
         'max_features': ['sqrt', 'log2']}
    ),
}

tuned = {}
for name, (est, space) in search_spaces.items():
    rs = RandomizedSearchCV(est, space, n_iter=20, cv=cv, scoring='roc_auc',
                            random_state=SEED, n_jobs=-1, verbose=0)
    rs.fit(X_train_bal, y_train_bal)
    tuned[name] = rs.best_estimator_
    print(f"{name:<20} CV AUC: {rs.best_score_:.4f}")
    print(f"{'':<20} best: {rs.best_params_}")

# Logistic Regression — no heavy search, but validate
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
lr_cv = cross_val_score(lr, X_train_bal, y_train_bal, cv=cv, scoring='roc_auc')
print(f"{'Logistic Regression':<20} CV AUC: {lr_cv.mean():.4f}")
tuned['Logistic Regression'] = lr.fit(X_train_bal, y_train_bal)

# Stacking ensemble — combines the tuned tree models with an LR meta-learner
print("\nBuilding stacking ensemble (RF + GB + ET -> Logistic meta-learner)...")
stack = StackingClassifier(
    estimators=[('rf', tuned['Random Forest']),
                ('gb', tuned['Gradient Boosting']),
                ('et', tuned['Extra Trees'])],
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=3, n_jobs=-1)
stack.fit(X_train_bal, y_train_bal)
tuned['Stacking Ensemble'] = stack
print("done.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. THRESHOLD TUNING ON VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════
def tune_threshold(y_true, y_prob):
    """Pick the threshold maximising F1 on the positive (churn) class."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best = np.argmax(f1[:-1])
    return float(thr[best])


def evaluate(model, X, y, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y, prob)
    return {
        'auc':        round(roc_auc_score(y, prob) * 100, 2),
        'pr_auc':     round(average_precision_score(y, prob) * 100, 2),
        'accuracy':   round(accuracy_score(y, pred) * 100, 2),
        'bal_acc':    round(balanced_accuracy_score(y, pred) * 100, 2),
        'precision':  round(precision_score(y, pred, zero_division=0) * 100, 2),
        'recall':     round(recall_score(y, pred, zero_division=0) * 100, 2),
        'f1':         round(f1_score(y, pred, zero_division=0) * 100, 2),
        'mcc':        round(matthews_corrcoef(y, pred), 3),
        'brier':      round(brier_score_loss(y, prob), 4),
        'threshold':  round(threshold, 3),
        'cm':         confusion_matrix(y, pred).tolist(),
        'fpr':        [round(v, 4) for v in fpr[::max(1, len(fpr)//60)]],
        'tpr':        [round(v, 4) for v in tpr[::max(1, len(tpr)//60)]],
    }


print("\n" + "=" * 72)
print("STAGE 3 — Threshold tuning on validation, evaluation on test")
print("=" * 72)

# Evaluate on validation first (for model selection), then on test (for reporting)
val_results = {}
for name, model in tuned.items():
    val_prob = model.predict_proba(X_val)[:, 1]
    thr = tune_threshold(y_val, val_prob)
    val_results[name] = {'threshold': thr, **evaluate(model, X_val, y_val, thr)}

# Select best model by VALIDATION F1 — never peek at test for selection
best_name = max(val_results, key=lambda k: val_results[k]['f1'])
best_model = tuned[best_name]
best_thr = val_results[best_name]['threshold']

# Now evaluate ALL models on the untouched test set for final reporting
results = {}
for name, model in tuned.items():
    thr = val_results[name]['threshold']
    results[name] = evaluate(model, X_test, y_test, thr)
    r = results[name]
    print(f"\n{name}")
    print(f"  ROC-AUC {r['auc']}%  |  PR-AUC {r['pr_auc']}%  |  threshold {r['threshold']}")
    print(f"  Precision {r['precision']}%  |  Recall {r['recall']}%  |  F1 {r['f1']}%")
    print(f"  Balanced Acc {r['bal_acc']}%  |  MCC {r['mcc']}  |  Brier {r['brier']}")

print("\n" + "=" * 72)
print(f"SELECTED MODEL (highest validation F1 on churn class): {best_name}")
print("=" * 72)
r = results[best_name]
print(f"  ROC-AUC:           {r['auc']}%   (dataset ceiling ~89.7%)")
print(f"  PR-AUC:            {r['pr_auc']}%")
print(f"  Precision:         {r['precision']}%")
print(f"  Recall:            {r['recall']}%")
print(f"  F1:                {r['f1']}%")
print(f"  Balanced Accuracy: {r['bal_acc']}%")
print(f"  MCC:               {r['mcc']}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STAGE 4 — Permutation importance (model-agnostic, on test set)")
print("=" * 72)

perm = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                              random_state=SEED, scoring='roc_auc', n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
imp_std = pd.Series(perm.importances_std, index=feature_names)

print(f"{'Feature':<24} {'Importance':>12} {'Std':>8}")
print("-" * 46)
for f in imp.head(12).index:
    print(f"{f:<24} {imp[f]:>12.4f} {imp_std[f]:>8.4f}")

feature_importance = {k: round(float(v), 4) for k, v in imp.head(12).items() if v > 0}


# ═══════════════════════════════════════════════════════════════════════════
# 7. BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════════════════
tn, fp, fn, tp = np.array(results[best_name]['cm']).ravel()
avg_monthly = df_raw['MonthlyCharges'].mean()
CLV_MONTHS, RETENTION_COST, SUCCESS_RATE = 24, 50, 0.30

revenue_saved = tp * avg_monthly * CLV_MONTHS * SUCCESS_RATE
campaign_cost = (tp + fp) * RETENTION_COST
missed = fn * avg_monthly * CLV_MONTHS

print("\n" + "=" * 72)
print("STAGE 5 — Business impact on the 1,409-customer test set")
print("=" * 72)
print(f"  Churners correctly identified:  {tp} of {tp+fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"  False alarms (wasted outreach): {fp}")
print(f"  Revenue protected (24mo CLV, 30% save rate): ${revenue_saved:,.0f}")
print(f"  Retention campaign cost:                     ${campaign_cost:,.0f}")
print(f"  Net benefit:                                 ${revenue_saved - campaign_cost:,.0f}")
print(f"  Revenue still lost to missed churners:       ${missed:,.0f}")
print(f"  ROI:                                         {(revenue_saved-campaign_cost)/campaign_cost*100:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 8. ANALYTICS + PERSIST
# ═══════════════════════════════════════════════════════════════════════════
d = df_raw
seg = lambda col: d.groupby(col)['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).round(2).to_dict()
tb = pd.cut(d['tenure'], [-1, 12, 24, 48, 72], labels=['0-12m', '13-24m', '25-48m', '49-72m'])
mb = pd.cut(d['MonthlyCharges'], [0, 35, 60, 85, 125], labels=['<$35', '$35-60', '$60-85', '$85+'])

analytics = {
    'total_customers': int(len(d)),
    'churned': int((d['Churn'] == 'Yes').sum()),
    'retained': int((d['Churn'] == 'No').sum()),
    'churn_rate': round((d['Churn'] == 'Yes').mean() * 100, 2),
    'avg_tenure': round(d['tenure'].mean(), 1),
    'avg_monthly': round(d['MonthlyCharges'].mean(), 2),
    'avg_total': round(d['TotalCharges'].mean(), 2),
    'churn_by_contract': seg('Contract'),
    'churn_by_internet': seg('InternetService'),
    'churn_by_payment': seg('PaymentMethod'),
    'churn_by_techsupport': seg('TechSupport'),
    'churn_by_tenure': {str(k): round(v, 2) for k, v in
                        d.groupby(tb)['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).items()},
    'churn_by_monthly': {str(k): round(v, 2) for k, v in
                         d.groupby(mb)['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).items()},
    'feature_importance': feature_importance,
    'model_results': results,
    'best_model': best_name,
    'best_threshold': best_thr,
    'feature_columns': feature_names,
    'business_impact': {
        'true_positives': int(tp), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_negatives': int(tn),
        'revenue_protected': round(revenue_saved),
        'campaign_cost': round(campaign_cost),
        'net_benefit': round(revenue_saved - campaign_cost),
        'roi_percent': round((revenue_saved - campaign_cost) / campaign_cost * 100),
    },
}

joblib.dump(best_model, f'{BASE}/models/best_model.pkl', compress=3)
joblib.dump(scaler, f'{BASE}/models/scaler.pkl', compress=3)
joblib.dump(le_dict, f'{BASE}/models/label_encoders.pkl', compress=3)
joblib.dump(feature_names, f'{BASE}/models/feature_columns.pkl', compress=3)
joblib.dump(best_thr, f'{BASE}/models/decision_threshold.pkl', compress=3)
with open(f'{BASE}/data/analytics.json', 'w') as f:
    json.dump(analytics, f, indent=2)

print("\nArtefacts saved to models/ and data/analytics.json")
