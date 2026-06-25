import pandas as pd
import numpy as np

def audyt_kalibracji(input_path):
    df = pd.read_csv(input_path)
    
    print(f"--- AUDYT KALIBRACJI 1X2 (N={len(df)}) ---")
    
    for kat in ['home', 'draw', 'away']:
        p_col = f'p_{kat}'
        r_col = 'wynik_1x2'
        rzecz_val = kat[0].upper()
        
        avg_p = df[p_col].mean()
        avg_r = (df[r_col] == rzecz_val).mean()
        
        print(f"{kat.upper():5s} | Model: {avg_p:.3f} | Rzecz: {avg_r:.3f} | Delta: {avg_r - avg_p:+.3f}")

    # Analiza bucketów dla faworytów (P_home > 0.6)
    faworyci = df[df['p_home'] > 0.6]
    if len(faworyci) > 0:
        acc_faw = (faworyci['wynik_pred'] == faworyci['wynik_1x2']).mean()
        print(f"\nFaworyci (P>0.60): Model mówi {faworyci['p_home'].mean():.1%}, Rzeczywistość: {acc_faw:.1%}")

if __name__ == "__main__":
    audyt_kalibracji("data/processed/backtesting_wyniki_v2.csv")