"""
test_availability_signal.py
===========================
Sprawdza czy availability_score koreluje z realnym xG drużyny.

Test 1: Korelacja Pearsona i Spearmana
Test 2: Bucket analysis (avg xG per bucket availability)
Test 3: Regresja liniowa xG ~ availability_home + availability_away
Test 4: Analiza per sezon (czy sygnał jest stabilny)
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = Path("db/ekstraklasa.db")

MIN_HISTORY = 6


def main():
    conn = sqlite3.connect(DB_PATH)

    # Połączenie: mecze + xG + availability home + availability away
    df = pd.read_sql("""
        SELECT
            m.sezon,
            m.match_id,
            m.data_meczu,
            m.kolejka,
            m.gospodarz,
            m.gosc,
            m.xg_gosp,
            m.xg_gosc,
            m.gole_gosp,
            m.gole_gosc,
            ah.availability_score AS avail_home,
            ah.history_matches AS hist_home,
            aa.availability_score AS avail_away,
            aa.history_matches AS hist_away
        FROM matches m
        JOIN match_availability ah ON ah.sezon = m.sezon AND ah.match_id = m.match_id AND ah.team_side = 'home'
        JOIN match_availability aa ON aa.sezon = m.sezon AND aa.match_id = m.match_id AND aa.team_side = 'away'
        WHERE ah.history_matches >= ?
          AND aa.history_matches >= ?
          AND m.xg_gosp IS NOT NULL
          AND m.xg_gosc IS NOT NULL
    """, conn, params=(MIN_HISTORY, MIN_HISTORY))

    conn.close()

    print("=" * 70)
    print("TEST SYGNAŁU AVAILABILITY")
    print("=" * 70)
    print(f"\nMecze do analizy: {len(df)}")
    print(f"Sezony: {sorted(df['sezon'].unique())}")

    # ================================================
    # TEST 1: Korelacje bezpośrednie
    # ================================================
    print("\n" + "=" * 70)
    print("[1] KORELACJE — czy availability wpływa na xG?")
    print("=" * 70)

    tests = [
        ("xg_gosp  vs  avail_home  (własne availability -> własne xG)", df["avail_home"], df["xg_gosp"]),
        ("xg_gosp  vs  avail_away  (rywala availability -> własne xG)", df["avail_away"], df["xg_gosp"]),
        ("xg_gosc  vs  avail_away  (własne availability -> własne xG)", df["avail_away"], df["xg_gosc"]),
        ("xg_gosc  vs  avail_home  (rywala availability -> własne xG)", df["avail_home"], df["xg_gosc"]),
    ]

    for name, x, y in tests:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        print(f"\n{name}")
        print(f"  Pearson  r={pearson_r:+.4f}  p={pearson_p:.4f}")
        print(f"  Spearman r={spearman_r:+.4f}  p={spearman_p:.4f}")

    # ================================================
    # TEST 2: Bucket analysis
    # ================================================
    print("\n" + "=" * 70)
    print("[2] BUCKET ANALYSIS — xG per bucket availability")
    print("=" * 70)

    def bucket(x):
        if x >= 0.95: return "A: 0.95-1.00 (pełny)"
        if x >= 0.90: return "B: 0.90-0.95"
        if x >= 0.85: return "C: 0.85-0.90"
        if x >= 0.80: return "D: 0.80-0.85"
        if x >= 0.70: return "E: 0.70-0.80"
        return "F: <0.70 (osłabiony)"

    # Dla gospodarza
    df["bucket_home"] = df["avail_home"].apply(bucket)
    b_home = df.groupby("bucket_home").agg(
        n=("xg_gosp", "count"),
        avg_xg_own=("xg_gosp", "mean"),
        avg_xg_opp=("xg_gosc", "mean"),
    ).round(3)
    print("\nGOSPODARZ (pogrupowany po SWOIM availability):")
    print(b_home.to_string())

    # Dla gościa
    df["bucket_away"] = df["avail_away"].apply(bucket)
    b_away = df.groupby("bucket_away").agg(
        n=("xg_gosc", "count"),
        avg_xg_own=("xg_gosc", "mean"),
        avg_xg_opp=("xg_gosp", "mean"),
    ).round(3)
    print("\nGOŚĆ (pogrupowany po SWOIM availability):")
    print(b_away.to_string())

    # ================================================
    # TEST 3: Regresja liniowa
    # ================================================
    print("\n" + "=" * 70)
    print("[3] REGRESJA LINIOWA")
    print("=" * 70)

    # Model dla xG gospodarza
    print("\n[A] xg_gosp = a + b1*avail_home + b2*avail_away")
    X = df[["avail_home", "avail_away"]].values
    y = df["xg_gosp"].values
    X_with_const = np.column_stack([np.ones(len(X)), X])
    coef, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    y_pred = X_with_const @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  intercept    = {coef[0]:+.4f}")
    print(f"  b_avail_home = {coef[1]:+.4f}  (spodziewane: DODATNIE — więcej dostępnych = więcej xG)")
    print(f"  b_avail_away = {coef[2]:+.4f}  (spodziewane: UJEMNE — słaby rywal = więcej xG dla nas)")
    print(f"  R^2          = {r2:.4f}")

    # Model dla xG gościa
    print("\n[B] xg_gosc = a + b1*avail_home + b2*avail_away")
    y = df["xg_gosc"].values
    coef, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    y_pred = X_with_const @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  intercept    = {coef[0]:+.4f}")
    print(f"  b_avail_home = {coef[1]:+.4f}  (spodziewane: UJEMNE)")
    print(f"  b_avail_away = {coef[2]:+.4f}  (spodziewane: DODATNIE)")
    print(f"  R^2          = {r2:.4f}")

    # ================================================
    # TEST 4: Stabilność per sezon
    # ================================================
    print("\n" + "=" * 70)
    print("[4] STABILNOŚĆ PER SEZON")
    print("=" * 70)

    for sezon in sorted(df["sezon"].unique()):
        sub = df[df["sezon"] == sezon]
        r1, _ = stats.pearsonr(sub["avail_home"], sub["xg_gosp"])
        r2, _ = stats.pearsonr(sub["avail_away"], sub["xg_gosc"])
        r3, _ = stats.pearsonr(sub["avail_away"], sub["xg_gosp"])
        r4, _ = stats.pearsonr(sub["avail_home"], sub["xg_gosc"])
        print(f"\n{sezon}  (n={len(sub)}):")
        print(f"  avail_home -> xg_gosp:  r={r1:+.4f}  (chcemy DODATNIA)")
        print(f"  avail_away -> xg_gosc:  r={r2:+.4f}  (chcemy DODATNIA)")
        print(f"  avail_away -> xg_gosp:  r={r3:+.4f}  (chcemy UJEMNA)")
        print(f"  avail_home -> xg_gosc:  r={r4:+.4f}  (chcemy UJEMNA)")


if __name__ == "__main__":
    main()