import pandas as pd
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def apply_calibration(row, T, b_h, b_d, b_a):
    # Logity z surowych prawdopodobieństw
    p_raw = np.array([row['p_home'], row['p_draw'], row['p_away']])
    logits = np.log(np.maximum(p_raw, 1e-10))
    
    # Aplikacja kalibracji
    calibrated_logits = (logits + np.array([b_h, b_d, b_a])) / T
    p_cal = softmax(calibrated_logits)
    
    return pd.Series({
        'p_h_cal': p_cal[0],
        'p_d_cal': p_cal[1],
        'p_a_cal': p_cal[2]
    })

def run_calibrated_backtest():
    # Dane wejściowe z v2
    df = pd.read_csv("data/processed/backtesting_wyniki_v2.csv")
    
    # Parametry które wyliczyłeś
    T = 2.8602
    b_h, b_d, b_a = 0.6037, -0.2032, -0.4005
    
    # Kalibrujemy
    cal_cols = df.apply(apply_calibration, axis=1, args=(T, b_h, b_d, b_a))
    df = pd.concat([df, cal_cols], axis=1)
    
    # Wyznaczamy nowe typy (wynik_pred_cal)
    def get_pred(row):
        probs = [row['p_h_cal'], row['p_d_cal'], row['p_a_cal']]
        idx = np.argmax(probs)
        return ['H', 'D', 'A'][idx]
    
    df['wynik_pred_cal'] = df.apply(get_pred, axis=1)
    
    # Liczymy nowy log-loss
    def get_ll(row):
        p = row['p_h_cal'] if row['wynik_1x2'] == 'H' else (row['p_d_cal'] if row['wynik_1x2'] == 'D' else row['p_a_cal'])
        return -np.log(max(p, 1e-10))
    
    df['ll_cal'] = df.apply(get_ll, axis=1)
    
    # --- RAPORT ---
    print(f"--- PORÓWNANIE PO KALIBRACJI ---")
    print(f"Log-loss RAW: {df['log_loss'].mean():.4f}")
    print(f"Log-loss CAL: {df['ll_cal'].mean():.4f} ({'✅ POPRAWA' if df['ll_cal'].mean() < df['log_loss'].mean() else '❌ GORZEJ'})")
    
    print(f"\nRozkład typowań (RAW vs CAL vs RZECZ):")
    for res in ['H', 'D', 'A']:
        n_raw = (df['wynik_pred'] == res).sum()
        n_cal = (df['wynik_pred_cal'] == res).sum()
        n_rzec = (df['wynik_1x2'] == res).sum()
        print(f"  {res}: RAW={n_raw:3d} | CAL={n_cal:3d} | RZECZ={n_rzec:3d}")

    acc_raw = (df['wynik_pred'] == df['wynik_1x2']).mean()
    acc_cal = (df['wynik_pred_cal'] == df['wynik_1x2']).mean()
    print(f"\nAccuracy RAW: {acc_raw:.1%}")
    print(f"Accuracy CAL: {acc_cal:.1%}")

    df.to_csv("data/processed/backtesting_wyniki_v2_calibrated.csv", index=False)

if __name__ == "__main__":
    run_calibrated_backtest()