"""
app.py — Flask REST API serving the churn prediction model.

Key detail: predictions use the tuned decision threshold loaded from
models/decision_threshold.pkl, NOT the default 0.5. The threshold was
optimised on a held-out validation set to maximise F1 on the churn class.
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import joblib

app = Flask(__name__)
BASE = '.'

# ── Load artefacts once at startup ──────────────────────────────────────────
MODEL        = joblib.load(f'{BASE}/models/best_model.pkl')
SCALER       = joblib.load(f'{BASE}/models/scaler.pkl')
LE_DICT      = joblib.load(f'{BASE}/models/label_encoders.pkl')
FEATURE_COLS = joblib.load(f'{BASE}/models/feature_columns.pkl')
THRESHOLD    = joblib.load(f'{BASE}/models/decision_threshold.pkl')

with open(f'{BASE}/data/analytics.json') as f:
    ANALYTICS = json.load(f)

DF = pd.read_csv(f'{BASE}/data/telco_churn.csv')


@app.template_filter('format_number')
def format_number(v):
    return f"{int(v):,}"


# ── Feature engineering — must exactly mirror train_model.py ────────────────
def engineer_features(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['AvgChargePerMonth'] = np.where(df['tenure'] > 0,
                                       df['TotalCharges'] / df['tenure'],
                                       df['MonthlyCharges'])
    df['ChargeIncreaseRatio'] = df['MonthlyCharges'] / (df['AvgChargePerMonth'] + 1e-6)
    svc = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
           'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = sum((df[c] == 'Yes').astype(int) for c in svc)
    df['HasProtection'] = ((df['OnlineSecurity'] == 'Yes') |
                           (df['TechSupport'] == 'Yes')).astype(int)
    df['TenureBand'] = pd.cut(df['tenure'], [-1, 6, 12, 24, 48, 72],
                              labels=[0, 1, 2, 3, 4]).astype(int)
    df['IsNewCustomer'] = (df['tenure'] < 12).astype(int)
    df['ContractLength'] = df['Contract'].map(
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['AutoPayment'] = df['PaymentMethod'].isin(
        ['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    df['HighRiskCombo'] = ((df['Contract'] == 'Month-to-month') &
                           (df['InternetService'] == 'Fiber optic') &
                           (df['tenure'] < 12)).astype(int)
    df['ChargePerService'] = df['MonthlyCharges'] / (df['NumServices'] + 1)
    return df


def preprocess_input(data: dict) -> pd.DataFrame:
    row = {
        'gender': data['gender'],
        'SeniorCitizen': int(data['SeniorCitizen']),
        'Partner': data['Partner'],
        'Dependents': data['Dependents'],
        'tenure': int(data['tenure']),
        'PhoneService': data['PhoneService'],
        'MultipleLines': data['MultipleLines'],
        'InternetService': data['InternetService'],
        'OnlineSecurity': data['OnlineSecurity'],
        'OnlineBackup': data['OnlineBackup'],
        'DeviceProtection': data['DeviceProtection'],
        'TechSupport': data['TechSupport'],
        'StreamingTV': data['StreamingTV'],
        'StreamingMovies': data['StreamingMovies'],
        'Contract': data['Contract'],
        'PaperlessBilling': data['PaperlessBilling'],
        'PaymentMethod': data['PaymentMethod'],
        'MonthlyCharges': float(data['MonthlyCharges']),
        'TotalCharges': float(data['TotalCharges']),
    }
    df = engineer_features(pd.DataFrame([row]))
    for col, le in LE_DICT.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = 0
    df = df[FEATURE_COLS]
    return pd.DataFrame(SCALER.transform(df), columns=FEATURE_COLS)


def risk_band(prob, threshold):
    """Risk bands anchored to the tuned threshold, not arbitrary cutoffs."""
    if prob >= threshold * 1.45:
        return 'High'
    if prob >= threshold:
        return 'Medium'
    if prob >= threshold * 0.55:
        return 'Low'
    return 'Minimal'


def recommendations(data, prob):
    recs = []
    if data.get('Contract') == 'Month-to-month':
        recs.append(("Offer a discounted 1-year contract upgrade",
                     "Month-to-month customers churn at 49.4% vs 5.2% on two-year contracts"))
    if int(data.get('tenure', 0)) < 12:
        recs.append(("Enrol in the 90-day onboarding success programme",
                     "First-year customers churn at 53.5% — the highest of any tenure band"))
    if data.get('TechSupport') == 'No':
        recs.append(("Bundle TechSupport free for 6 months",
                     "Customers with TechSupport churn 8.6 points lower"))
    if data.get('OnlineSecurity') == 'No':
        recs.append(("Provide a free OnlineSecurity trial",
                     "Security add-ons measurably increase account stickiness"))
    if data.get('PaymentMethod') == 'Electronic check':
        recs.append(("Incentivise switching to automatic payment",
                     "Electronic check users churn at 45.3% vs ~19% on auto-pay"))
    if float(data.get('MonthlyCharges', 0)) > 85:
        recs.append(("Run a plan-optimisation review",
                     "High monthly charges correlate strongly with price-driven churn"))
    if not recs:
        recs.append(("Maintain standard engagement cadence",
                     "No elevated risk factors detected for this profile"))
    return [{'action': a, 'rationale': r} for a, r in recs[:4]]


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', analytics=ANALYTICS)


@app.route('/api/analytics')
def analytics():
    return jsonify(ANALYTICS)


REQUIRED_FIELDS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges',
]


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400
        X = preprocess_input(data)
        prob = float(MODEL.predict_proba(X)[0][1])
        pred = int(prob >= THRESHOLD)
        return jsonify({
            'churn': pred,
            'probability': round(prob * 100, 1),
            'threshold': round(THRESHOLD * 100, 1),
            'risk_level': risk_band(prob, THRESHOLD),
            'recommendations': recommendations(data, prob),
            'model': ANALYTICS['best_model'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/customers')
def customers():
    page = int(request.args.get('page', 1))
    per = 50
    chunk = DF.iloc[(page - 1) * per: page * per]
    return jsonify({'customers': chunk.to_dict('records'),
                    'total': len(DF), 'page': page})


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Score a sample of customers and return them ranked by risk —
    this is how a retention team would actually consume the model."""
    try:
        n = int(request.json.get('n', 100)) if request.json else 100
        sample = DF.sample(min(n, len(DF)), random_state=42)
        out = []
        for _, row in sample.iterrows():
            try:
                p = float(MODEL.predict_proba(preprocess_input(row.to_dict()))[0][1])
                out.append({'customerID': row['customerID'],
                            'probability': round(p * 100, 1),
                            'risk_level': risk_band(p, THRESHOLD),
                            'actual': row['Churn']})
            except Exception:
                continue
        out.sort(key=lambda r: r['probability'], reverse=True)
        return jsonify({'customers': out, 'threshold': round(THRESHOLD * 100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print(f"Model: {ANALYTICS['best_model']}  |  Threshold: {THRESHOLD:.3f}")
    app.run(debug=True, port=5050)
