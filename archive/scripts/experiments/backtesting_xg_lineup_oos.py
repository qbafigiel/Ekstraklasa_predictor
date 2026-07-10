"""
backtesting_xg_lineup_oos.py
============================
Uczciwy test out-of-sample dla feature:
    diff_lineup_offense

Schemat:
1) Generujemy SUROWE predykcje Poisson(xG) dla sezonu 2024/25
   - trening: 2023/24 + poprzednie kolejki 2024/25
   - xG 2023/24 fallback -> gole
2) Generujemy SUROWE predykcje Poisson(xG) dla sezonu 2025/26
   - trening: 2023/24 + 2024/25 + poprzednie kolejki 2025/26
3) Kalibrujemy parametry:
   A) baseline: T, bH, bD, bA
   B) lineup:   T, bH, bD, bA, gamma
   na sezonie 2024/25
4) Testujemy oba modele na 2025/26
5) Porównujemy log-loss out-of-sample

To jest właściwy test:
    czy diff_lineup_offense naprawdę wnosi sygnał,
    czy tylko overfitował 2025/26.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings

warnings.filterwarnings("ignore")

DB_PATH = Path("db/ekstraklasa.db")
LINEUP_PATH = Path("data/processed/match_lineup_values.csv")
OUTPUT_PATH = Path("data/processed/backtesting_xg_lineup_oos_2025_26.csv")
REPORT_PATH = Path("data/reports/model/backtesting_xg_lineup_oos_report.txt")

MAX_GOLE = 10

SEASON_ORDER = {
    "2023/24": 1,
    "2024/25": 2,
    "2025/26": 3,
}

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}


# =============================================================================
# MODEL Poisson(xG) — zgodny z backtesting_xg_v1.py
# =============================================================================

def przygotuj_dane_xg(df_trening):
    df_trening = df_trening.copy()
    df_trening["xg_gosp_final"] = df_trening["xg_gosp"].fillna(df_trening["gole_gosp"])
    df_trening["xg_gosc_final"] = df_trening["xg_gosc"].fillna(df_trening["gole_gosc"])

    druzyny = sorted(
        set(df_trening["gospodarz"].unique()) |
        set(df_trening["gosc"].unique())
    )
    n = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    return {
        "n_druzyn": n,
        "idx_to_team": i2t,
        "home_idx": df_trening["gospodarz"].map(t2i).values,
        "away_idx": df_trening["gosc"].map(t2i).values,
        "goals_home": df_trening["xg_gosp_final"].values.astype(float),
        "goals_away": df_trening["xg_gosc_final"].values.astype(float),
        "weights": df_trening["waga_sezonu"].values.astype(float),
    }


def neg_log_likelihood(theta, data):
    N = data["n_druzyn"]

    log_alpha = np.zeros(N)
    log_beta = np.zeros(N)

    log_alpha[:N - 1] = theta[2:2 + N - 1]
    log_alpha[N - 1] = -np.sum(log_alpha[:N - 1])

    log_beta[:N - 1] = theta[2 + N - 1:2 + 2 * (N - 1)]
    log_beta[N - 1] = -np.sum(log_beta[:N - 1])

    mu_home = np.exp(theta[0])
    mu_away = np.exp(theta[1])

    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)

    lh = np.maximum(
        mu_home * alpha[data["home_idx"]] * beta[data["away_idx"]],
        1e-10
    )

    la = np.maximum(
        mu_away * alpha[data["away_idx"]] * beta[data["home_idx"]],
        1e-10
    )

    ll = np.sum(data["weights"] * (
        data["goals_home"] * np.log(lh) - lh +
        data["goals_away"] * np.log(la) - la
    ))

    return -ll


def trenuj_model(data):
    N = data["n_druzyn"]

    theta0 = np.zeros(2 + 2 * (N - 1))
    theta0[0] = np.log(np.mean(data["goals_home"]))
    theta0[1] = np.log(np.mean(data["goals_away"]))

    result = minimize(
        neg_log_likelihood,
        theta0,
        args=(data,),
        method="L-BFGS-B",
        options={"maxiter": 10000}
    )

    return result.x


def ekstrahuj_parametry(theta, data):
    N = data["n_druzyn"]

    log_alpha = np.zeros(N)
    log_beta = np.zeros(N)

    log_alpha[:N - 1] = theta[2:2 + N - 1]
    log_alpha[N - 1] = -np.sum(log_alpha[:N - 1])

    log_beta[:N - 1] = theta[2 + N - 1:2 + 2 * (N - 1)]
    log_beta[N - 1] = -np.sum(log_beta[:N - 1])

    return {
        "mu_home": np.exp(theta[0]),
        "mu_away": np.exp(theta[1]),
        "alpha": {
            data["idx_to_team"][i]: np.exp(log_alpha[i])
            for i in range(N)
        },
        "beta": {
            data["idx_to_team"][i]: np.exp(log_beta[i])
            for i in range(N)
        },
    }


def przewiduj_mecz(params, gospodarz, gosc):
    if gospodarz not in params["alpha"] or gosc not in params["alpha"]:
        return None

    lh = (
        params["mu_home"]
        * params["alpha"][gospodarz]
        * params["beta"][gosc]
    )

    la = (
        params["mu_away"]
        * params["alpha"][gosc]
        * params["beta"][gospodarz]
    )

    macierz = np.zeros((MAX_GOLE, MAX_GOLE))

    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)

    total = macierz.sum()
    if total <= 0:
        return None

    macierz /= total

    p_home = float(np.sum(np.tril(macierz, -1)))
    p_draw = float(np.sum(np.diag(macierz)))
    p_away = float(np.sum(np.triu(macierz, 1)))

    return {
        "lambda_home": lh,
        "lambda_away": la,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
    }


# =============================================================================
# HELPERS
# =============================================================================

def wynik_1x2(gh, ga):
    if gh > ga:
        return "H"
    elif gh == ga:
        return "D"
    return "A"


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def log_loss_1x2(p_vec, wynik):
    idx = {"H": 0, "D": 1, "A": 2}[wynik]
    return -np.log(max(float(p_vec[idx]), 1e-12))


def previous_seasons(target_season):
    target_order = SEASON_ORDER[target_season]
    return [s for s, o in SEASON_ORDER.items() if o < target_order]


# =============================================================================
# SUROWE PREDYKCJE ROLLING
# =============================================================================

def run_raw_backtesting(df_all, target_season):
    df_test = df_all[df_all["sezon"] == target_season].copy()
    df_hist = df_all[df_all["sezon"].isin(previous_seasons(target_season))].copy()

    kolejki = sorted(df_test["kolejka"].unique())
    wyniki = []

    print(f"\n--- Generowanie surowych predykcji: {target_season} ---")
    print(f"Historyczne sezony treningowe: {', '.join(previous_seasons(target_season))}")
    print(f"Mecze historyczne: {len(df_hist)}")
    print(f"Mecze target season: {len(df_test)}")
    print(f"Kolejki: {len(kolejki)}")

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_trening = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane_xg(df_trening)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)

        df_kolejka = df_test[df_test["kolejka"] == kolejka]

        n_ok = 0
        n_brak = 0

        for _, mecz in df_kolejka.iterrows():
            gosp = mecz["gospodarz"]
            gosc = mecz["gosc"]
            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])

            pred = przewiduj_mecz(params, gosp, gosc)
            if pred is None:
                n_brak += 1
                continue

            wyniki.append({
                "sezon": target_season,
                "match_id": mecz["match_id"],
                "kolejka": int(kolejka),
                "gospodarz": gosp,
                "gosc": gosc,
                "wynik_1x2": wynik_1x2(gh, ga),
                "p_home_raw": pred["p_home"],
                "p_draw_raw": pred["p_draw"],
                "p_away_raw": pred["p_away"],
                "lambda_home": pred["lambda_home"],
                "lambda_away": pred["lambda_away"],
                "diff_lineup_offense": float(mecz["diff_lineup_offense"]),
            })
            n_ok += 1

        print(f"  K{kolejka:02d}: {n_ok} predykcji | {n_brak} brak")

    return pd.DataFrame(wyniki)


# =============================================================================
# KALIBRACJA
# =============================================================================

def calibrate_baseline(p_home, p_draw, p_away, T, bH, bD, bA):
    logits = np.log(np.maximum(np.array([p_home, p_draw, p_away]), 1e-12))
    logits_cal = (logits + np.array([bH, bD, bA])) / T
    return softmax(logits_cal)


def calibrate_lineup(p_home, p_draw, p_away, diff_offense, T, bH, bD, bA, gamma):
    logits = np.log(np.maximum(np.array([p_home, p_draw, p_away]), 1e-12))
    mod = np.array([
        gamma * diff_offense,
        0.0,
        -gamma * diff_offense
    ])
    logits_cal = (logits + np.array([bH, bD, bA]) + mod) / T
    return softmax(logits_cal)


def objective_baseline(params, df_calib):
    T, bH, bD, bA = params
    if T <= 0.1:
        return 999.0

    ll = []
    for row in df_calib.itertuples(index=False):
        p = calibrate_baseline(
            row.p_home_raw,
            row.p_draw_raw,
            row.p_away_raw,
            T, bH, bD, bA
        )
        ll.append(log_loss_1x2(p, row.wynik_1x2))

    return float(np.mean(ll))


def objective_lineup(params, df_calib):
    T, bH, bD, bA, gamma = params
    if T <= 0.1:
        return 999.0

    ll = []
    for row in df_calib.itertuples(index=False):
        p = calibrate_lineup(
            row.p_home_raw,
            row.p_draw_raw,
            row.p_away_raw,
            row.diff_lineup_offense,
            T, bH, bD, bA, gamma
        )
        ll.append(log_loss_1x2(p, row.wynik_1x2))

    return float(np.mean(ll))


def fit_baseline(df_calib):
    result = minimize(
        objective_baseline,
        x0=[1.5, 0.1, 0.0, -0.1],
        args=(df_calib,),
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2)]
    )
    return result


def fit_lineup(df_calib):
    result = minimize(
        objective_lineup,
        x0=[1.5, 0.1, 0.0, -0.1, 0.01],
        args=(df_calib,),
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2), (-0.1, 0.1)]
    )
    return result


# =============================================================================
# EWALUACJA
# =============================================================================

def apply_baseline_to_df(df_in, params):
    T, bH, bD, bA = params
    rows = []

    for row in df_in.itertuples(index=False):
        p = calibrate_baseline(
            row.p_home_raw,
            row.p_draw_raw,
            row.p_away_raw,
            T, bH, bD, bA
        )
        rows.append({
            "p_home_cal_base": p[0],
            "p_draw_cal_base": p[1],
            "p_away_cal_base": p[2],
            "log_loss_base": log_loss_1x2(p, row.wynik_1x2),
        })

    return pd.concat([df_in.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def apply_lineup_to_df(df_in, params):
    T, bH, bD, bA, gamma = params
    rows = []

    for row in df_in.itertuples(index=False):
        p = calibrate_lineup(
            row.p_home_raw,
            row.p_draw_raw,
            row.p_away_raw,
            row.diff_lineup_offense,
            T, bH, bD, bA, gamma
        )
        rows.append({
            "p_home_cal_lineup": p[0],
            "p_draw_cal_lineup": p[1],
            "p_away_cal_lineup": p[2],
            "log_loss_lineup": log_loss_1x2(p, row.wynik_1x2),
        })

    return pd.concat([df_in.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuję matches...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()

    required_match_cols = {
        "match_id", "sezon", "kolejka", "gospodarz", "gosc",
        "gole_gosp", "gole_gosc", "xg_gosp", "xg_gosc"
    }
    missing_match = required_match_cols - set(df.columns)
    if missing_match:
        raise RuntimeError(f"Brak wymaganych kolumn w matches: {sorted(missing_match)}")

    print("2. Wczytuję lineup values...")
    lineup = pd.read_csv(LINEUP_PATH)

    required_lineup_cols = {"match_id", "sezon", "diff_lineup_offense"}
    missing_lineup = required_lineup_cols - set(lineup.columns)
    if missing_lineup:
        raise RuntimeError(f"Brak wymaganych kolumn w {LINEUP_PATH}: {sorted(missing_lineup)}")

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)
    df = df.merge(
        lineup[["match_id", "sezon", "diff_lineup_offense"]],
        on=["match_id", "sezon"],
        how="left"
    )
    df["diff_lineup_offense"] = pd.to_numeric(df["diff_lineup_offense"], errors="coerce").fillna(0.0)

    print("3. Generuję surowe predykcje dla 2024/25 (kalibracja)...")
    df_calib_raw = run_raw_backtesting(df, "2024/25")

    print("\n4. Generuję surowe predykcje dla 2025/26 (test)...")
    df_test_raw = run_raw_backtesting(df, "2025/26")

    if len(df_calib_raw) == 0 or len(df_test_raw) == 0:
        raise RuntimeError("Brak predykcji w sezonie kalibracji lub teście.")

    print("\n5. Fit kalibracji baseline na 2024/25...")
    res_base = fit_baseline(df_calib_raw)

    print("6. Fit kalibracji lineup na 2024/25...")
    res_lineup = fit_lineup(df_calib_raw)

    params_base = res_base.x
    params_lineup = res_lineup.x

    print("7. Aplikuję oba modele na 2025/26...")
    df_test_base = apply_baseline_to_df(df_test_raw, params_base)
    df_test_both = apply_lineup_to_df(df_test_base, params_lineup)

    ll_test_base = df_test_both["log_loss_base"].mean()
    ll_test_lineup = df_test_both["log_loss_lineup"].mean()

    ll_calib_base = res_base.fun
    ll_calib_lineup = res_lineup.fun

    T_A, bH_A, bD_A, bA_A = params_base
    T_B, bH_B, bD_B, bA_B, gamma_B = params_lineup

    df_test_both.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=" * 78)
    lines.append("OUT-OF-SAMPLE TEST — Poisson(xG) + diff_lineup_offense")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. ZBIORY")
    lines.append("-" * 78)
    lines.append("  Kalibracja parametrów: 2024/25")
    lines.append("  Test out-of-sample:    2025/26")
    lines.append(f"  Predykcje 2024/25:     {len(df_calib_raw)}")
    lines.append(f"  Predykcje 2025/26:     {len(df_test_raw)}")
    lines.append("")
    lines.append("2. KALIBRACJA NA 2024/25")
    lines.append("-" * 78)
    lines.append(f"  Baseline log-loss:     {ll_calib_base:.4f}")
    lines.append(f"  Lineup log-loss:       {ll_calib_lineup:.4f}")
    lines.append(f"  Delta:                 {ll_calib_base - ll_calib_lineup:+.4f}")
    lines.append("")
    lines.append("3. TEST OOS NA 2025/26")
    lines.append("-" * 78)
    lines.append(f"  Baseline log-loss:     {ll_test_base:.4f}")
    lines.append(f"  Lineup log-loss:       {ll_test_lineup:.4f}")
    lines.append(f"  Delta (base-lineup):   {ll_test_base - ll_test_lineup:+.4f}")
    lines.append(f"  Benchmark losowy:      {np.log(3):.4f}")
    lines.append("")
    lines.append("4. PARAMETRY")
    lines.append("-" * 78)
    lines.append(f"  Baseline: T={T_A:.4f}, bH={bH_A:.4f}, bD={bD_A:.4f}, bA={bA_A:.4f}")
    lines.append(f"  Lineup:   T={T_B:.4f}, bH={bH_B:.4f}, bD={bD_B:.4f}, bA={bA_B:.4f}, gamma={gamma_B:.6f}")
    lines.append("")
    lines.append("5. DIFF_LINEUP_OFFENSE")
    lines.append("-" * 78)
    lines.append(f"  2024/25 mean: {df_calib_raw['diff_lineup_offense'].mean():.2f}")
    lines.append(f"  2024/25 std:  {df_calib_raw['diff_lineup_offense'].std():.2f}")
    lines.append(f"  2025/26 mean: {df_test_raw['diff_lineup_offense'].mean():.2f}")
    lines.append(f"  2025/26 std:  {df_test_raw['diff_lineup_offense'].std():.2f}")
    lines.append("")
    lines.append("6. WNIOSEK ROBOCZY")
    lines.append("-" * 78)

    delta_oos = ll_test_base - ll_test_lineup
    if delta_oos > 0.0010:
        lines.append("  SYGNAŁ POZYTYWNY OOS")
        lines.append("  Lineup offense poprawia model także poza próbką kalibracyjną.")
        lines.append("  Kandydat do wdrożenia produkcyjnego.")
    elif delta_oos > 0:
        lines.append("  MINIMALNA POPRAWA OOS")
        lines.append("  Jest lepiej, ale bardzo delikatnie.")
        lines.append("  Wdrożenie tylko jeśli chcemy wycisnąć ostatnie tysięczne log-loss.")
    else:
        lines.append("  BRAK POPRAWY OOS")
        lines.append("  In-sample improvement był najpewniej overfitem.")
        lines.append("  Nie wdrażać do produkcji.")
    lines.append("")
    lines.append("7. PLIKI")
    lines.append("-" * 78)
    lines.append(f"  Predykcje testowe: {OUTPUT_PATH}")
    lines.append(f"  Raport:            {REPORT_PATH}")

    report_text = "\n".join(lines)

    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano predykcje: {OUTPUT_PATH}")
    print(f"Zapisano raport:    {REPORT_PATH}")


if __name__ == "__main__":
    main()