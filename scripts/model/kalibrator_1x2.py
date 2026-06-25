import pandas as pd
import numpy as np
from scipy.optimize import minimize

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def loss_function(params, p_raw, y_true):
    T, b_h, b_d, b_a = params
    total_ll = 0
    for i in range(len(p_raw)):
        # Przejście do logitów (log-space)
        logits = np.log(np.maximum(p_raw[i], 1e-10))
        # Kalibracja: (Logit + Bias) / T
        calibrated_logits = (logits + np.array([b_h, b_d, b_a])) / T
        probs = softmax(calibrated_logits)
        
        actual_idx = y_true[i]
        total_ll -= np.log(max(probs[actual_idx], 1e-10))
    return total_ll / len(p_raw)

def trenuj_kalibrator(input_path):
    df = pd.read_csv(input_path)
    p_raw = df[['p_home', 'p_draw', 'p_away']].values
    y_true = df['wynik_1x2'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    # Inicjalizacja: T=1.0, biasy=0
    res = minimize(loss_function, [1.0, 0.0, 0.0, 0.0], args=(p_raw, y_true), method='L-BFGS-B')
    
    T, b_h, b_d, b_a = res.x
    print(f"--- PARAMETRY KALIBRACJI ---")
    print(f"Temperatura (T): {T:.4f} (T>1 oznacza studzenie modelu)")
    print(f"Bias H: {b_h:.4f} | Bias D: {b_d:.4f} | Bias A: {b_a:.4f}")
    
    return res.x

if __name__ == "__main__":
    trenuj_kalibrator("data/processed/backtesting_wyniki_xg_v1.csv")