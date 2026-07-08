"""
test_availability_within_team.py
================================
Prawidłowy test wpływu availability: within-team, nie between-teams.

Dla każdej drużyny:
1. Liczymy jej średnie xG_own i xG_conceded w sezonie
2. Dla każdego meczu liczymy DEVIATION od średniej
3. Sprawdzamy czy deviation koreluje z availability

To pokazuje: "czy drużyna X gra GORZEJ gdy JEJ skład jest osłabiony
             względem tego jak SAMA sobie zwykle radzi"
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = Path("db/ekstraklasa.db")


def main():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            m.sezon, m.match_id, m.kolejka, m.data_meczu,
            m.gospodarz, m.gosc,
            m.xg_gosp, m.xg_gosc,
            ah.availability_score AS avail_home,
            aa.availability_score AS avail_away,
            ah.history_matches AS hist_home,
            aa.history_matches AS hist_away
        FROM matches m
        JOIN match_availability ah ON ah.sezon=m.sezon AND ah.match_id=m.match_id AND ah.team_side='home'
        JOIN match_availability aa ON aa.sezon=m.sezon AND aa.match_id=m.match_id AND aa.team_side='away'
        WHERE m.xg_gosp IS NOT NULL AND m.xg_gosc IS NOT NULL
          AND ah.history_matches >= 6 AND aa.history_matches >= 6
    """, conn)
    conn.close()

    # Zbuduj długi format: jeden wiersz per (mecz, drużyna)
    home_rows = df.rename(columns={
        "gospodarz": "team", "gosc": "opponent",
        "xg_gosp": "xg_own", "xg_gosc": "xg_conceded",
        "avail_home": "avail_own", "avail_away": "avail_opp",
        "hist_home": "hist_own",
    })[["sezon", "match_id", "kolejka", "team", "opponent",
        "xg_own", "xg_conceded", "avail_own", "avail_opp", "hist_own"]]
    home_rows["side"] = "home"

    away_rows = df.rename(columns={
        "gosc": "team", "gospodarz": "opponent",
        "xg_gosc": "xg_own", "xg_gosp": "xg_conceded",
        "avail_away": "avail_own", "avail_home": "avail_opp",
        "hist_away": "hist_own",
    })[["sezon", "match_id", "kolejka", "team", "opponent",
        "xg_own", "xg_conceded", "avail_own", "avail_opp", "hist_own"]]
    away_rows["side"] = "away"

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    print("=" * 70)
    print("TEST WITHIN-TEAM: czy własna słabość składu -> gorsze wyniki?")
    print("=" * 70)
    print(f"Wierszy (mecz × drużyna): {len(long)}")
    print(f"Sezony: {sorted(long['sezon'].unique())}")

    # ================================================
    # KROK 1: Dla każdej pary (drużyna, sezon) — średnie xG
    # ================================================
    team_avg = long.groupby(["team", "sezon"]).agg(
        avg_xg_own=("xg_own", "mean"),
        avg_xg_conceded=("xg_conceded", "mean"),
        n_matches=("match_id", "count"),
    ).reset_index()

    # ================================================
    # KROK 2: Deviation od średniej per mecz
    # ================================================
    long = long.merge(team_avg, on=["team", "sezon"])
    long["xg_own_dev"] = long["xg_own"] - long["avg_xg_own"]
    long["xg_conceded_dev"] = long["xg_conceded"] - long["avg_xg_conceded"]

    # ================================================
    # KROK 3: Korelacje within-team
    # ================================================
    print("\n" + "=" * 70)
    print("[1] KORELACJE WITHIN-TEAM (deviation od własnej średniej)")
    print("=" * 70)

    tests = [
        ("xg_own_dev  vs  avail_own   (własna słabość -> spadek xG względem WŁASNEJ średniej)",
         long["avail_own"], long["xg_own_dev"], "+"),
        ("xg_conceded_dev  vs  avail_own   (własna słabość -> tracimy WIĘCEJ xG)",
         long["avail_own"], long["xg_conceded_dev"], "-"),
        ("xg_own_dev  vs  avail_opp   (rywala słabość -> zdobywamy więcej xG)",
         long["avail_opp"], long["xg_own_dev"], "-"),
        ("xg_conceded_dev  vs  avail_opp   (rywala słabość -> tracimy mniej xG)",
         long["avail_opp"], long["xg_conceded_dev"], "+"),
    ]

    for name, x, y, expected in tests:
        # remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_c, y_c = x[mask], y[mask]

        pearson_r, pearson_p = stats.pearsonr(x_c, y_c)
        spearman_r, spearman_p = stats.spearmanr(x_c, y_c)

        expected_txt = "DODATNI" if expected == "+" else "UJEMNY"
        actual_ok = (pearson_r > 0 and expected == "+") or (pearson_r < 0 and expected == "-")
        marker = "✓" if actual_ok and pearson_p < 0.05 else ("~" if actual_ok else "✗")

        print(f"\n{name}")
        print(f"  spodziewany kierunek: {expected_txt}")
        print(f"  {marker} Pearson  r={pearson_r:+.4f}  p={pearson_p:.4f}")
        print(f"    Spearman r={spearman_r:+.4f}  p={spearman_p:.4f}")

    # ================================================
    # KROK 4: Bucket analysis within-team
    # ================================================
    print("\n" + "=" * 70)
    print("[2] BUCKET ANALYSIS WITHIN-TEAM")
    print("=" * 70)

    def bucket(x):
        if x >= 0.95: return "A: 0.95-1.00 (pełny)"
        if x >= 0.90: return "B: 0.90-0.95"
        if x >= 0.85: return "C: 0.85-0.90"
        if x >= 0.80: return "D: 0.80-0.85"
        if x >= 0.70: return "E: 0.70-0.80"
        return "F: <0.70 (osłabiony)"

    long["bucket_own"] = long["avail_own"].apply(bucket)
    b = long.groupby("bucket_own").agg(
        n=("xg_own_dev", "count"),
        avg_own_deviation=("xg_own_dev", "mean"),
        avg_conc_deviation=("xg_conceded_dev", "mean"),
    ).round(3)
    print("\nDrużyna pogrupowana po SWOIM availability:")
    print("(deviation = odchylenie od WŁASNEJ średniej w sezonie)")
    print(b.to_string())
    print("\nOczekujemy:")
    print("  im niższy bucket, tym MNIEJSZE avg_own_deviation (gorsze wyniki niż zwykle)")
    print("  im niższy bucket, tym WIĘKSZE avg_conc_deviation (tracimy więcej niż zwykle)")

    # ================================================
    # KROK 5: Regresja z fixed effects (dummy per drużyna)
    # ================================================
    print("\n" + "=" * 70)
    print("[3] REGRESJA Z KONTROLĄ SIŁY DRUŻYNY")
    print("=" * 70)
    print("Model: xg_own ~ avail_own + avail_opp + team_dummy + opp_dummy")
    print("(kontrolujemy siłę drużyn dummy variables)")

    # One-hot encoding drużyn
    teams = sorted(set(long["team"]) | set(long["opponent"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}

    n = len(long)
    X_team = np.zeros((n, len(teams)))
    X_opp = np.zeros((n, len(teams)))
    for i, row in enumerate(long.itertuples()):
        X_team[i, team_to_idx[row.team]] = 1
        X_opp[i, team_to_idx[row.opponent]] = 1

    X_avail = long[["avail_own", "avail_opp"]].values
    y = long["xg_own"].values

    # Usuń ostatnią kolumnę każdego dummy (unikamy multikolinearności)
    X = np.column_stack([np.ones(n), X_avail, X_team[:, :-1], X_opp[:, :-1]])

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    print(f"\n  b_avail_own = {coef[1]:+.4f}  (chcemy DODATNI)")
    print(f"  b_avail_opp = {coef[2]:+.4f}  (chcemy UJEMNY)")
    print(f"  R² = {r2:.4f}  (bez kontroli było 0.006)")
    print(f"\n  Znaczy: po kontroli siły drużyn, availability {'DAJE' if abs(coef[1]) > 0.15 else 'NIE DAJE'} sygnału")


if __name__ == "__main__":
    main()