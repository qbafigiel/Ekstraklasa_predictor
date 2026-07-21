"""
predict_round.py
================
Generuje predykcje dla wszystkich meczow z wybranej kolejki.

Uzycie:
    python scripts/predict/predict_round.py --sezon 2026/27 --kolejka 1
    python scripts/predict/predict_round.py --sezon 2026/27 --kolejka 1 --status scheduled
    python scripts/predict/predict_round.py --sezon 2026/27 --kolejki 1-5

Wyjscie:
    - data/processed/predictions/predict_round_{sezon}_K{kolejka}.csv
    - tabela tekstowa w konsoli (skrocona)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "predict"))

# Importujemy funkcje z predict_match.py
from predict_match import (
    load_matches,
    load_referee_features,
    build_team_name_map,
    resolve_team,
    normalize_text,
    fit_poisson_mle,
    apply_priors,
    apply_prior_new_team,
    predict_goals_matrix,
    softmax_cal,
    fit_cal_1x2,
    fit_btts_shift,
    run_val_predictions,
    build_rolling_state,
    get_referee_features,
    compute_stat_predictions,
)


DB_PATH = ROOT / "db" / "ekstraklasa.db"
OUTPUT_DIR = ROOT / "data" / "processed" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_kolejki_arg(arg):
    if not arg:
        return None
    if "-" in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in arg.split(",")]


def load_fixtures(conn, sezon, kolejki, status_filter=None):
    where = ["sezon = ?"]
    params = [sezon]

    if kolejki:
        placeholders = ",".join("?" * len(kolejki))
        where.append(f"kolejka IN ({placeholders})")
        params.extend(kolejki)

    if status_filter:
        where.append("status = ?")
        params.append(status_filter)

    query = f"""
        SELECT fixture_id, sezon, kolejka, gospodarz, gosc,
               data_planowana, godzina, referee_full_name, status
        FROM fixtures_upcoming
        WHERE {' AND '.join(where)}
        ORDER BY kolejka, data_planowana, godzina, gospodarz
    """

    return pd.read_sql_query(query, conn, params=params)


def build_prediction_row(fixture, home, away, ref_name, ref, goals, cal, stats):
    p_H_cal, p_D_cal, p_A_cal = cal["p_H"], cal["p_D"], cal["p_A"]
    p_btts = min(max(goals["p_btts_yes"] + cal["btts_shift"], 0.01), 0.99)

    row = {
        "fixture_id": fixture["fixture_id"],
        "sezon": fixture["sezon"],
        "kolejka": fixture["kolejka"],
        "data_planowana": fixture["data_planowana"],
        "godzina": fixture["godzina"],
        "gospodarz": home,
        "gosc": away,
        "referee_full_name": ref_name if ref_name else "",
        "referee_known": int(ref.get("ref_known", False)),
        "referee_n_matches": ref.get("ref_matches", 0),

        "alpha_home": round(goals["alpha_home"], 4),
        "alpha_away": round(goals["alpha_away"], 4),
        "beta_home": round(goals["beta_home"], 4),
        "beta_away": round(goals["beta_away"], 4),

        "lambda_home": round(goals["lambda_home"], 3),
        "lambda_away": round(goals["lambda_away"], 3),

        "p_H": round(p_H_cal, 4),
        "p_D": round(p_D_cal, 4),
        "p_A": round(p_A_cal, 4),

        "p_over_05": round(goals["p_over_05"], 4),
        "p_over_15": round(goals["p_over_15"], 4),
        "p_over_25": round(goals["p_over_25"], 4),
        "p_over_35": round(goals["p_over_35"], 4),

        "p_btts_yes": round(p_btts, 4),
        "p_btts_no": round(1 - p_btts, 4),

        "mu_corners": round(stats["corners"]["mu_total"], 2),
        "mu_shots": round(stats["shots"]["mu_total"], 2),
        "mu_sot": round(stats["sot"]["mu_total"], 2),
        "mu_offsides": round(stats["offsides"]["mu_total"], 2),
        "mu_fouls": round(stats["fouls"]["mu_total"], 2),
        "mu_yc": round(stats["yc"]["mu_total"], 2),
    }

    # Dodajemy wszystkie linie O/U dla kazdego rynku
    for market_key, market_data in [
        ("corners", stats["corners"]),
        ("shots", stats["shots"]),
        ("sot", stats["sot"]),
        ("offsides", stats["offsides"]),
        ("fouls", stats["fouls"]),
        ("yc", stats["yc"]),
    ]:
        for k, v in market_data.items():
            if k.startswith("p_over_") or k.startswith("p_under_"):
                row[f"{market_key}_{k}"] = round(v, 4)

    return row


def format_short_line(row):
    """Krotki output do konsoli."""
    date_str = row["data_planowana"] or "brak_daty"
    time_str = row["godzina"] or "--:--"
    ref_str = row["referee_full_name"] if row["referee_full_name"] else "brak"

    return (
        f"  K{row['kolejka']:02d} {date_str} {time_str}  "
        f"{row['gospodarz']:<26} vs {row['gosc']:<26} | "
        f"1X2: {row['p_H']*100:4.1f}/{row['p_D']*100:4.1f}/{row['p_A']*100:4.1f}  "
        f"Gole: {row['lambda_home']:.2f}+{row['lambda_away']:.2f}={row['lambda_home']+row['lambda_away']:.2f}  "
        f"BTTS: {row['p_btts_yes']*100:.0f}%  "
        f"Sedzia: {ref_str}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True, help="np. 2026/27")
    parser.add_argument("--kolejka", type=int, help="Jedna kolejka")
    parser.add_argument("--kolejki", type=str, help="Zakres np. '1-5' albo '1,3,7'")
    parser.add_argument("--status", default=None, help="Filtr statusu (np. scheduled)")
    args = parser.parse_args()

    if args.kolejka:
        kolejki = [args.kolejka]
    elif args.kolejki:
        kolejki = parse_kolejki_arg(args.kolejki)
    else:
        print("Blad: podaj --kolejka albo --kolejki")
        sys.exit(1)

    print(f"Sezon:    {args.sezon}")
    print(f"Kolejki:  {kolejki}")
    print(f"Status:   {args.status or 'wszystkie'}")
    print()

    # Ladujemy dane
    print("1. Laduje mecze historyczne...")
    df_all = load_matches()
    print(f"   {len(df_all)} meczow")

    print("2. Buduje mape nazw druzyn...")
    team_map = build_team_name_map(df_all)

    print("3. Laduje profile sedziow...")
    df_ref = load_referee_features()

    print("4. Fit Poisson MLE (raz na cala sesje)...")
    params = fit_poisson_mle(df_all)
    params = apply_priors(params, df_all)

    print("5. Kalibracja 1X2 i BTTS...")
    df_val_preds = run_val_predictions(df_all, params)
    T, bH, bD, bA = fit_cal_1x2(df_val_preds)
    btts_shift = fit_btts_shift(df_val_preds)
    print(f"   T={T:.3f} bH={bH:.3f} bD={bD:.3f} bA={bA:.3f} | BTTS shift={btts_shift:+.3f}")

    print("6. Buduje rolling state druzyn...")
    state, g = build_rolling_state(df_all)

    print("7. Laduje fixture z bazy...")
    conn = sqlite3.connect(DB_PATH)
    df_fixtures = load_fixtures(conn, args.sezon, kolejki, args.status)
    conn.close()

    if df_fixtures.empty:
        print(f"Brak fixture dla podanych filtrow.")
        sys.exit(0)

    print(f"   Fixture: {len(df_fixtures)}")
    print()

    # Predykcje
    print("8. Licze predykcje...")
    rows = []
    for _, fixture in df_fixtures.iterrows():
        home_raw = fixture["gospodarz"]
        away_raw = fixture["gosc"]
        ref_name = fixture["referee_full_name"] if pd.notna(fixture["referee_full_name"]) else None

        home, home_known = resolve_team(home_raw, team_map)
        away, away_known = resolve_team(away_raw, team_map)

        # Zapewniamy priors dla nieznanych druzyn
        apply_prior_new_team(params, home)
        apply_prior_new_team(params, away)

        home_is_new = not home_known
        away_is_new = not away_known

        goals = predict_goals_matrix(params, home, away)
        p_H_cal, p_D_cal, p_A_cal = softmax_cal(
            goals["p_H"], goals["p_D"], goals["p_A"], T, bH, bD, bA
        )
        cal = {"p_H": p_H_cal, "p_D": p_D_cal, "p_A": p_A_cal, "btts_shift": btts_shift}

        ref = get_referee_features(ref_name, df_ref, g["fouls"] / 2.0, g["yc"] / 2.0)

        stats = compute_stat_predictions(
            state=state, g=g,
            home=home, away=away,
            home_is_new=home_is_new, away_is_new=away_is_new,
            ref_fouls_log_ratio=ref["ref_fouls_log_ratio"],
            ref_yc_log_ratio=ref["ref_yc_log_ratio"],
        )

        row = build_prediction_row(fixture, home, away, ref_name, ref, goals, cal, stats)
        rows.append(row)

    df_preds = pd.DataFrame(rows)

    # Zapis CSV
    if len(kolejki) == 1:
        fname = f"predict_round_{args.sezon.replace('/', '-')}_K{kolejki[0]:02d}.csv"
    else:
        fname = f"predict_round_{args.sezon.replace('/', '-')}_K{kolejki[0]:02d}-K{kolejki[-1]:02d}.csv"
    out_path = OUTPUT_DIR / fname
    df_preds.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Krotki output do konsoli
    print()
    print("=" * 120)
    print(f"PREDYKCJE — sezon {args.sezon}, kolejki {kolejki}")
    print("=" * 120)
    current_kolejka = None
    for _, r in df_preds.iterrows():
        if r["kolejka"] != current_kolejka:
            print(f"\n--- KOLEJKA {r['kolejka']} ---")
            current_kolejka = r["kolejka"]
        print(format_short_line(r))

    print(f"\n{'=' * 120}")
    print(f"Zapisano: {out_path}")
    print(f"Predykcji: {len(df_preds)}")


if __name__ == "__main__":
    main()