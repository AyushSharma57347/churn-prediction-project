"""
generate_dataset.py — Realistic Telco churn dataset generator.

Calibrated against the published segment-level churn rates of the IBM Telco
Customer Churn dataset. Three things make this different from a naive generator:

  1. LOGIT-SCALE EFFECTS. Churn is generated through a logistic link, so
     coefficients are log-odds and effect sizes are realistic (a month-to-month
     contract multiplies churn odds ~4x, not "+10 percentage points").

  2. INTERACTIONS. Real churn risk is not additive. Fiber optic on a
     month-to-month contract is far worse than either factor alone. New
     customers on expensive plans churn at extreme rates.

  3. CORRELATED FEATURES. Fiber customers pay more. Long-tenure customers
     hold longer contracts. TotalCharges is a function of tenure and monthly
     charge. This gives tree models real structure to find.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 7043


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n=N):
    # ── Demographics ────────────────────────────────────────────────────────
    gender     = RNG.choice(['Male', 'Female'], n)
    senior     = RNG.choice([0, 1], n, p=[0.84, 0.16])
    partner    = RNG.choice(['Yes', 'No'], n, p=[0.48, 0.52])
    # dependents correlate with having a partner
    dependents = np.where(
        partner == 'Yes',
        RNG.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
        RNG.choice(['Yes', 'No'], n, p=[0.13, 0.87]),
    )

    # ── Tenure: bimodal. Many brand-new customers, many long-term loyalists ──
    tenure = np.where(
        RNG.random(n) < 0.42,
        RNG.gamma(1.6, 7.0, n),           # newer cohort
        RNG.normal(52, 16, n),            # established cohort
    ).clip(0, 72).round().astype(int)

    # ── Contract: strongly tied to tenure (loyal customers sign longer) ─────
    contract = np.empty(n, dtype=object)
    for i, t in enumerate(tenure):
        if t < 12:
            contract[i] = RNG.choice(['Month-to-month', 'One year', 'Two year'], p=[0.82, 0.13, 0.05])
        elif t < 30:
            contract[i] = RNG.choice(['Month-to-month', 'One year', 'Two year'], p=[0.60, 0.26, 0.14])
        elif t < 50:
            contract[i] = RNG.choice(['Month-to-month', 'One year', 'Two year'], p=[0.38, 0.30, 0.32])
        else:
            contract[i] = RNG.choice(['Month-to-month', 'One year', 'Two year'], p=[0.20, 0.25, 0.55])

    # ── Services ────────────────────────────────────────────────────────────
    phone = RNG.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    multi = np.where(phone == 'Yes', RNG.choice(['Yes', 'No'], n, p=[0.47, 0.53]), 'No phone service')
    internet = RNG.choice(['Fiber optic', 'DSL', 'No'], n, p=[0.44, 0.34, 0.22])

    def addon(base_p_dsl, base_p_fiber):
        """Add-on uptake differs by internet type; None if no internet."""
        out = np.empty(n, dtype=object)
        for i, net in enumerate(internet):
            if net == 'No':
                out[i] = 'No internet service'
            else:
                p = base_p_fiber if net == 'Fiber optic' else base_p_dsl
                out[i] = 'Yes' if RNG.random() < p else 'No'
        return out

    online_sec = addon(0.42, 0.27)
    online_bak = addon(0.44, 0.33)
    dev_prot   = addon(0.43, 0.34)
    tech_sup   = addon(0.44, 0.26)
    stream_tv  = addon(0.28, 0.52)
    stream_mv  = addon(0.28, 0.52)

    paperless = np.where(
        internet == 'Fiber optic',
        RNG.choice(['Yes', 'No'], n, p=[0.74, 0.26]),
        RNG.choice(['Yes', 'No'], n, p=[0.49, 0.51]),
    )

    # Payment method correlates with contract length
    payment = np.empty(n, dtype=object)
    for i, c in enumerate(contract):
        if c == 'Month-to-month':
            payment[i] = RNG.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                p=[0.51, 0.22, 0.14, 0.13])
        else:
            payment[i] = RNG.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                p=[0.17, 0.20, 0.32, 0.31])

    # ── Monthly charges: built from actual service mix (realistic pricing) ──
    monthly = np.full(n, 0.0)
    monthly += np.where(phone == 'Yes', 20.0, 0.0)
    monthly += np.where(multi == 'Yes', 5.5, 0.0)
    monthly += np.where(internet == 'Fiber optic', 45.0, np.where(internet == 'DSL', 25.0, 0.0))
    for svc in (online_sec, online_bak, dev_prot, tech_sup):
        monthly += np.where(svc == 'Yes', 5.0, 0.0)
    for svc in (stream_tv, stream_mv):
        monthly += np.where(svc == 'Yes', 10.0, 0.0)
    monthly += RNG.normal(0, 2.0, n)
    monthly = monthly.clip(18.25, 120.0).round(2)

    # TotalCharges = tenure * monthly, with drift from historical price changes
    total = (tenure * monthly * RNG.normal(1.0, 0.06, n)).clip(0)
    total = np.where(tenure == 0, monthly, total).round(2)

    # ── CHURN via logistic model (log-odds scale) ───────────────────────────
    z = np.full(n, -0.80)   # baseline intercept (calibrated to ~26.5% churn)

    # Contract — the dominant driver
    z += np.where(contract == 'Month-to-month',  0.95, 0.0)
    z += np.where(contract == 'One year',       -0.45, 0.0)
    z += np.where(contract == 'Two year',       -1.40, 0.0)

    # Tenure — non-linear decay in early months (survival-curve shape)
    z += -1.45 * (1.0 - np.exp(-tenure / 14.0))

    # Internet service
    z += np.where(internet == 'Fiber optic',  0.62, 0.0)
    z += np.where(internet == 'No',          -0.80, 0.0)

    # Support / security add-ons are protective
    z += np.where(online_sec == 'Yes', -0.42, 0.0)
    z += np.where(tech_sup   == 'Yes', -0.40, 0.0)
    z += np.where(online_bak == 'Yes', -0.18, 0.0)
    z += np.where(dev_prot   == 'Yes', -0.16, 0.0)

    # Billing / payment friction
    z += np.where(payment == 'Electronic check', 0.48, 0.0)
    z += np.where(paperless == 'Yes',            0.26, 0.0)

    # Demographics
    z += 0.32 * senior
    z += np.where(partner    == 'Yes', -0.20, 0.0)
    z += np.where(dependents == 'Yes', -0.26, 0.0)

    # Price sensitivity — scaled, continuous
    z += 0.014 * (monthly - 65.0)

    # ── INTERACTIONS: where the real predictive lift lives ──────────────────
    m2m   = (contract == 'Month-to-month')
    fiber = (internet == 'Fiber optic')
    new   = (tenure < 12)

    z += 0.45 * (m2m & fiber)                      # fiber on rolling contract
    z += 0.50 * (m2m & new)                        # new + no commitment
    z += 0.35 * (fiber & (monthly > 90))           # premium price shock
    z += 0.30 * (m2m & (payment == 'Electronic check'))
    z += -0.45 * ((contract == 'Two year') & (tenure > 48))   # entrenched loyalist
    z += 0.28 * (new & (online_sec == 'No') & (tech_sup == 'No'))  # unsupported newcomer

    # Irreducible noise — real churn is driven by unobservable factors
    # (competitor offers, relocation, service outages, personal circumstances).
    # This is what keeps the ceiling realistic rather than artificially perfect.
    z += RNG.normal(0, 1.15, n)

    p_churn = sigmoid(z)
    churn = RNG.binomial(1, p_churn)

    df = pd.DataFrame({
        'customerID': [f"{RNG.integers(1000,9999)}-{''.join(RNG.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),5))}"
                       for _ in range(n)],
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi,
        'InternetService': internet, 'OnlineSecurity': online_sec, 'OnlineBackup': online_bak,
        'DeviceProtection': dev_prot, 'TechSupport': tech_sup, 'StreamingTV': stream_tv,
        'StreamingMovies': stream_mv, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total,
        'Churn': np.where(churn == 1, 'Yes', 'No'),
    })
    return df, p_churn


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    df, p_true = generate()
    y = (df['Churn'] == 'Yes').astype(int)

    BASE = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(BASE, 'data', 'telco_churn.csv'), index=False)

    print("=" * 68)
    print("NEW DATASET GENERATED")
    print("=" * 68)
    print(f"Rows: {len(df)}   Churn rate: {y.mean()*100:.2f}%")
    print(f"Bayes-optimal AUC (ceiling): {roc_auc_score(y, p_true):.4f}")
    print(f"  (old dataset ceiling was 0.7135)")
    print()
    print("Segment churn rates — compare against real IBM Telco figures:")
    print()
    for col in ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']:
        print(f"  {col}:")
        rates = df.groupby(col)['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).round(1)
        for k, v in rates.items():
            print(f"     {k:<26} {v:>5.1f}%")
        print()
    bands = pd.cut(df['tenure'], [-1, 12, 24, 48, 72], labels=['0-12m', '13-24m', '25-48m', '49-72m'])
    print("  Tenure band:")
    for k, v in df.groupby(bands)['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).round(1).items():
        print(f"     {str(k):<26} {v:>5.1f}%")
