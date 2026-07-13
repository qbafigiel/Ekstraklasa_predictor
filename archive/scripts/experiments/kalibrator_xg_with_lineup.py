import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

INPUT_XG_PATH = Path("data/processed/backtesting_wyniki_xg_v1.csv")
LINEUP_PATH = Path("data/processed/match_lineup_values.csv")
OUTPUT_PATH = Path("data/processed/backtesting_wyniki_xg_lineup_calibrated.csv")

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def log_loss_multi(probs, targets):
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    # y to one-hot encoding albo indeks
    if targets.ndim == 1:
        # indeks od 0 do 2
        return -np.mean(np.log(probs[np.arange(len(targets)), targets]))
    return -np.mean(np.sum(targets * np.log(probs), axis=1))

def obj_func_base(params, logits_raw, y_idx):
    T, bH, bD, bA = params
    if T <= 0.1:
        return 999.0
    
    b = np.array([bH, bD, bA])
    logits_cal = (logits_raw + b) / T
    probs = softmax(logits_cal)
    return log_loss_multi(probs, y_idx)

def obj_func_lineup(params, logits_raw, diff_offense, y_idx):
    T, bH, bD, bA, gamma = params
    if T <= 0.1:
        return 999.0
    
    b = np.array([bH, bD, bA])
    # Dodamy wplyw lineupu: H dostaje +gamma*diff, A dostaje -gamma*diff, D bez zmian
    mod = np.column_stack([
        gamma * diff_offense,
        np.zeros_like(diff_offense),
        -gamma * diff_offense
    ])
    
    logits_cal = (logits_raw + b + mod) / T
    probs = softmax(logits_cal)
    return log_loss_multi(probs, y_idx)

def main():
    print("1. Wczytuję surowe predykcje i lineup values...")
    df_xg = pd.read_csv(INPUT_XG_PATH)
    df_lineup = pd.read_csv(LINEUP_PATH)

    # df_xg z model_xg_poisson.py nie ma match_id, musimy zjoinować po gosp, gosc i kolejce dla sezonu 2025/26
    # Najpierw weźmy match_id z matches żeby dokładnie zmapować
    import sqlite3
    conn = sqlite3.connect("db/ekstraklasa.db")
    matches = pd.read_sql_query("SELECT match_id, sezon, kolejka, gospodarz, gosc FROM matches WHERE sezon='2025/26'", conn)
    conn.close()

    # Łączymy predykcje z match_id
    df_merged = pd.merge(df_xg, matches, left_on=["kolejka", "gospodarz", "gosc"], right_on=["kolejka", "gospodarz", "gosc"])
    
    # Teraz łączymy z lineup values
    df_final = pd.merge(df_merged, df_lineup[["match_id", "diff_lineup_offense"]], on="match_id", how="inner")

    # Przygotowujemy dane do optymalizacji
    p_raw = df_final[["p_home", "p_draw", "p_away"]].values
    logits_raw = np.log(np.maximum(p_raw, 1e-12))
    
    y_map = {"H": 0, "D": 1, "A": 2}
    y_idx = df_final["wynik_1x2"].map(y_map).values
    diff_off = df_final["diff_lineup_offense"].fillna(0.0).values

    print(f"Ilość meczów w analizie: {len(df_final)}")
    print(f"Log-loss RAW (przed kalibracją): {log_loss_multi(p_raw, y_idx):.4f}")

    # OPTYMALIZACJA 1: Tylko T i b (Stary model)
    res_base = minimize(
        obj_func_base, 
        x0=[1.5, 0.1, 0.0, -0.1], 
        args=(logits_raw, y_idx),
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2)]
    )

    # OPTYMALIZACJA 2: T, b oraz gamma (Nowy model z Lineup)
    res_lineup = minimize(
        obj_func_lineup, 
        x0=[1.5, 0.1, 0.0, -0.1, 0.01], 
        args=(logits_raw, diff_off, y_idx),
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2), (-0.5, 0.5)]
    )

    ll_base = res_base.fun
    ll_lineup = res_lineup.fun

    print("\n==================================================")
    print("PORÓWNANIE MODELI NA SEZONIE 2025/26 (306 meczów)")
    print("==================================================")
    print(f"1. Model stary (T + biasy):       Log-Loss = {ll_base:.4f}")
    print(f"2. Model NOWY (+ Lineup Offense): Log-Loss = {ll_lineup:.4f}")
    print("--------------------------------------------------")
    diff = ll_base - ll_lineup
    if diff > 0:
        print(f"✔ SUKCES! Lineup poprawił model o {diff:.4f} w dół!")
        print(f"✔ Optymalna wartość gamma = {res_lineup.x[4]:.5f}")
    else:
        print(f"❌ Brak poprawy. Różnica: {diff:.4f}")
    print("==================================================")

    # Zapiszmy predykcje nowego modelu
    T, bH, bD, bA, gamma = res_lineup.x
    mod = np.column_stack([gamma * diff_off, np.zeros_like(diff_off), -gamma * diff_off])
    logits_cal = (logits_raw + np.array([bH, bD, bA]) + mod) / T
    probs_cal = softmax(logits_cal)

    df_final["p_home_cal"] = probs_cal[:, 0]
    df_final["p_draw_cal"] = probs_cal[:, 1]
    df_final["p_away_cal"] = probs_cal[:, 2]

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nZapisano predykcje do: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()