"""
evaluate_value_bets.py
======================
Ewaluuje ile z rekomendowanych value bets faktycznie trafiło.

Bierze rekomendacje z recommend_value_bets.py (te same filtry)
i porównuje z rzeczywistymi wynikami z matches.

Użycie:
python scripts/audit/evaluate_value_bets.py --sezon 2026/27 --kolejka 3
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")
PRED_DIR = Path("data/processed/predictions")

# Te same konfiguracje co w recommend_value_bets.py
RELIABLE_MARKETS = {
    "corners": {"mu_col": "mu_corners", "prefix": "corners", "label": "Kornery",
                "actual_h": "rozne_gosp", "actual_a": "rozne_gosc"},
    "shots": {"mu_col": "mu_shots", "prefix": "shots", "label": "Strzały",
              "actual_h": "strzaly_gosp", "actual_a": "strzaly_gosc"},
    "sot": {"mu_col": "mu_sot", "prefix": "sot", "label": "Strzały celne",
            "actual_h": "celne_gosp", "actual_a": "celne_gosc"},
    "offsides": {"mu_col": "mu_offsides", "prefix": "offsides", "label": "Spalone",
                 "actual_h": "spalone_gosp", "actual_a": "spalone_gosc"},
    "yc": {"mu_col": "mu_yc", "prefix": "yc", "label": "Żółte kartki",
           "actual_h": "zk_gosp", "actual_a": "zk_gosc"},
}

RELIABLE_LINES = {
    "corners": [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
    "shots": [22.5, 23.5, 24.5, 25.5, 26.5],
    "sot": [6.5, 7.5, 8.5, 9.5, 10.5],
    "offsides": [1.5, 2.5, 3.5],
    "yc": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
}

MU_RANGE = {
    "corners": 3.0, "shots": 4.0, "sot": 2.0, "offsides": 1.5, "yc": 2.0,
}

MIN_CONFIDENCE = 0.58


def load_pred(sezon, kolejka):
    season_part = sezon.replace("/", "-")
    return pd.read_csv(PRED_DIR / f"predict_round_{season_part}_K{kolejka:02d}.csv")


def load_actuals(sezon, kolejka):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches WHERE sezon=? AND kolejka=?",
                            conn, params=(sezon, kolejka))
    conn.close()
    return df


def get_scheduled_fixtures(sezon, kolejka):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT gospodarz, gosc FROM fixtures_upcoming
        WHERE sezon=? AND kolejka=? AND status='played'
    """, conn, params=(sezon, kolejka))
    conn.close()
    return set((r["gospodarz"], r["gosc"]) for _, r in df.iterrows())


def analyze_match_bets(pred_row, actual_row):
    """Zwraca listę bets z pewnością modelu i info czy trafił."""
    bets = []
    
    for market_key, cfg in RELIABLE_MARKETS.items():
        prefix = cfg["prefix"]
        label = cfg["label"]
        mu = pred_row.get(cfg["mu_col"])
        
        if pd.isna(mu):
            continue
        
        actual_h = actual_row.get(cfg["actual_h"])
        actual_a = actual_row.get(cfg["actual_a"])
        
        if pd.isna(actual_h) or pd.isna(actual_a):
            continue
        
        actual_total = float(actual_h) + float(actual_a)
        mu_range = MU_RANGE.get(prefix, 2.0)
        
        for line in RELIABLE_LINES.get(prefix, []):
            if abs(line - float(mu)) > mu_range:
                continue
            
            line_int = int(line)
            line_dec = int(round((line - line_int) * 10))
            col_over = f"{prefix}_p_over_{line_int}_{line_dec}"
            col_under = f"{prefix}_p_under_{line_int}_{line_dec}"
            
            p_over = pred_row.get(col_over)
            p_under = pred_row.get(col_under)
            
            if pd.isna(p_over) or pd.isna(p_under):
                continue
            
            if p_over > p_under:
                side = "OVER"
                p = p_over
            else:
                side = "UNDER"
                p = p_under
            
            if p < MIN_CONFIDENCE:
                continue
            
            actual_over = actual_total > line
            correct = (side == "OVER" and actual_over) or (side == "UNDER" and not actual_over)
            
            bets.append({
                "mecz": f"{pred_row['gospodarz']} vs {pred_row['gosc']}",
                "rynek": label,
                "typ": f"{side} {line:.1f}",
                "pewność": p,
                "μ": mu,
                "actual_total": actual_total,
                "trafiony": correct,
            })
    
    return bets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True)
    parser.add_argument("--kolejka", type=int, required=True)
    args = parser.parse_args()
    
    preds = load_pred(args.sezon, args.kolejka)
    actuals = load_actuals(args.sezon, args.kolejka)
    scheduled = get_scheduled_fixtures(args.sezon, args.kolejka)
    
    all_bets = []
    for _, pred in preds.iterrows():
        if (pred["gospodarz"], pred["gosc"]) not in scheduled:
            continue
        act = actuals[(actuals["gospodarz"] == pred["gospodarz"]) &
                       (actuals["gosc"] == pred["gosc"])]
        if len(act) == 0:
            continue
        bets = analyze_match_bets(pred, act.iloc[0])
        all_bets.extend(bets)
    
    if not all_bets:
        print("Brak rekomendacji do ewaluacji.")
        return
    
    df = pd.DataFrame(all_bets).sort_values("pewność", ascending=False).reset_index(drop=True)
    
    print("=" * 110)
    print(f"EWALUACJA VALUE BETS - {args.sezon} K{args.kolejka:02d}")
    print("=" * 110)
    print(f"Wszystkich rekomendacji: {len(df)}")
    print(f"Trafionych: {df['trafiony'].sum()}/{len(df)} = {df['trafiony'].mean()*100:.1f}%")
    print()
    
    # Per rynek
    print("=" * 110)
    print("PER RYNEK")
    print("=" * 110)
    for rynek in df["rynek"].unique():
        sub = df[df["rynek"] == rynek]
        print(f"  {rynek:<20} {sub['trafiony'].sum()}/{len(sub)} = {sub['trafiony'].mean()*100:.1f}%")
    
    # Per strona (OVER/UNDER)
    print("\n" + "=" * 110)
    print("PER STRONA")
    print("=" * 110)
    for side in ["OVER", "UNDER"]:
        sub = df[df["typ"].str.startswith(side)]
        if len(sub) == 0:
            continue
        print(f"  {side:<10} {sub['trafiony'].sum()}/{len(sub)} = {sub['trafiony'].mean()*100:.1f}%")
    
    # Per przedział pewności
    print("\n" + "=" * 110)
    print("PER PRZEDZIAŁ PEWNOŚCI")
    print("=" * 110)
    ranges = [(0.80, 1.0, "≥80%"), (0.70, 0.80, "70-80%"), (0.65, 0.70, "65-70%"), (0.58, 0.65, "58-65%")]
    for low, high, label in ranges:
        sub = df[(df["pewność"] >= low) & (df["pewność"] < high)]
        if len(sub) == 0:
            continue
        print(f"  {label:<10} {sub['trafiony'].sum()}/{len(sub)} = {sub['trafiony'].mean()*100:.1f}%")
    
    # Wszystkie rekomendacje
    print("\n" + "=" * 110)
    print("WSZYSTKIE REKOMENDACJE (sortowane po pewności)")
    print("=" * 110)
    print(f"{'#':<3} {'Mecz':<40} {'Rynek':<15} {'Typ':<12} {'Pewność':<10} {'Wynik':<12} {'✓/✗'}")
    print("-" * 110)
    for i, r in df.iterrows():
        marker = "✓" if r["trafiony"] else "✗"
        print(f"{i+1:<3} {r['mecz']:<40} {r['rynek']:<15} {r['typ']:<12} "
              f"{r['pewność']*100:>6.1f}%    total={r['actual_total']:>5.1f}  {marker}")


if __name__ == "__main__":
    main()