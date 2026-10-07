"""
Diagnose why the old pipeline was stuck at AUC ~0.68.

The old generator computed a true churn probability p for each customer,
then drew the label as Bernoulli(p). If we score customers using the TRUE p
(the best any model could possibly do — the Bayes-optimal ranker), the AUC
we get is the hard ceiling for that dataset. No model can beat it.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 7043

# ── Reproduce the OLD generator's probability formula ────────────────────────
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[.55, .21, .24])
internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[.34, .44, .22])
tenure   = np.random.randint(0, 73, n)
monthly  = np.random.normal(65, 30, n).clip(18, 120)
senior   = np.random.choice([0, 1], n, p=[.84, .16])
partner  = np.random.choice(['Yes', 'No'], n, p=[.48, .52])
paperless= np.random.choice(['Yes', 'No'], n, p=[.59, .41])
payment  = np.random.choice(['Electronic check', 'Mailed check',
                             'Bank transfer (automatic)', 'Credit card (automatic)'],
                            n, p=[.34, .23, .22, .21])
online_sec = np.random.choice(['Yes', 'No'], n)
tech_sup   = np.random.choice(['Yes', 'No'], n)

p_old = (
    0.15
    + 0.10 * (contract == 'Month-to-month')
    - 0.08 * (contract == 'Two year')
    + 0.08 * (internet == 'Fiber optic')
    - 0.05 * (internet == 'No')
    + 0.05 * (online_sec == 'No')
    - 0.04 * (tech_sup == 'Yes')
    - 0.003 * tenure
    + 0.001 * monthly
    + 0.04 * senior
    - 0.03 * (partner == 'Yes')
    + 0.02 * (paperless == 'Yes')
    + 0.03 * (payment == 'Electronic check')
).clip(0.02, 0.92)

y_old = np.random.binomial(1, p_old)

print("=" * 68)
print("DIAGNOSIS OF THE OLD DATASET")
print("=" * 68)
print(f"Churn rate:                  {y_old.mean()*100:.1f}%")
print(f"True probability range:      {p_old.min():.3f} to {p_old.max():.3f}")
print(f"True probability std dev:    {p_old.std():.4f}")
print()
print(f"BAYES-OPTIMAL AUC (ceiling): {roc_auc_score(y_old, p_old):.4f}")
print()
print("This is the AUC you'd get scoring with the TRUE generating probability.")
print("No model — not XGBoost, not a neural net — can exceed this number.")
print("The 0.68 was never a modelling failure. The signal was never in the data.")
print()
print("Root cause: the coefficients are tiny relative to the Bernoulli noise.")
print(f"  Probabilities only span {p_old.std():.3f} std dev around the mean,")
print(f"  so most customers look nearly identical to any classifier.")
