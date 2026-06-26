"""
model_1x2_logistic.py
=====================
Nowy dedykowany model 1X2 oparty na regresji logistycznej (Multinomial).

Używamy cech z modelu xG (lambda_home, lambda_away, diff, total).
Uczymy model bezpośrednio przewidywać wynik H/D/A.
To jest obecnie najpoważniejsza próba poprawy 1X2.

Zapisuje:
- model.joblib
- scaler.joblib
- wyniki testowe
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import joblib
from pathlib import Path

# ====================== KONFIG ======================
DATA_PATH = Path("data/processed/backtesting_wyniki_xg_v1.csv")
MODEL_PATH = Path("models/1x2_logistic_model.joblib")
SCALER_PATH = Path("models/1x2_logistic_scaler.joblib")
RESULTS_PATH = Path("data/processed/results_1x2_logistic.csv")

Path("models").mkdir(exist_ok=True)

# ====================== WCZYTANIE DANYCH ======================
df = pd.read_csv(DATA_PATH)

# Tworzymy cechy
df = df.copy()
df['lambda_diff'] = df['lambda_home'] - df['lambda_away']
df['lambda_total'] = df['lambda_home'] + df['lambda_away']

features = ['lambda_home', 'lambda_away', 'lambda_diff', 'lambda_total']

X = df[features].values
y = df['wynik_1x2'].values

print(f"Dane załadowane: {len(df)} meczów")
print(f"Rozkład wyników: {df['wynik_1x2'].value_counts().to_dict()}")

# ====================== SKALOWANIE + MODEL ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=0.8,           # lekkie regularyzacja
    max_iter=1000,
    random_state=42
)

model.fit(X_scaled, y)

# Zapis modelu
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model zapisany: {MODEL_PATH}")

# ====================== PREDYKCJE I OCENA ======================
probs = model.predict_proba(X_scaled)
pred = model.predict(X_scaled)

# Przypisanie prawdopodobieństw do kolumn
class_order = model.classes_  # np. ['A', 'D', 'H']
prob_dict = {cls: probs[:, i] for i, cls in enumerate(class_order)}

df['p_home_log'] = prob_dict.get('H', np.zeros(len(df)))
df['p_draw_log'] = prob_dict.get('D', np.zeros(len(df)))
df['p_away_log'] = prob_dict.get('A', np.zeros(len(df)))
df['pred_log'] = pred

# Log-loss
def get_prob(row):
    if row['wynik_1x2'] == 'H': return row['p_home_log']
    if row['wynik_1x2'] == 'D': return row['p_draw_log']
    return row['p_away_log']

df['log_loss_logistic'] = df.apply(get_prob, axis=1)
df['log_loss_logistic'] = -np.log(np.maximum(df['log_loss_logistic'], 1e-10))

print("\n" + "="*60)
print("WYNIKI MODELU LOGISTYCZNEGO 1X2")
print("="*60)
print(f"Log-loss : {df['log_loss_logistic'].mean():.4f}")
print(f"Accuracy : {(df['pred_log'] == df['wynik_1x2']).mean():.3%}")

print("\nRozkład typowań (argmax):")
print(df['pred_log'].value_counts())

print("\nŚrednie prawdopodobieństwa:")
print(f"H = {df['p_home_log'].mean():.3f}")
print(f"D = {df['p_draw_log'].mean():.3f}")
print(f"A = {df['p_away_log'].mean():.3f}")

# Zapis wyników
df.to_csv(RESULTS_PATH, index=False)
print(f"\nWyniki zapisane do: {RESULTS_PATH}")