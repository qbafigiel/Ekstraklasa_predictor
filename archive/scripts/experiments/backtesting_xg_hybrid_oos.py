"""
backtesting_xg_hybrid_oos.py
=============================
Test OOS: czy użycie hybrid xG dla 2023/24
(observed Flash + constrained imputation)
zamiast fallback na gole poprawia log-loss modelu?

Kalibracja: 2024/25
Test OOS:   2025/26

Porównanie:
  A) Baseline: 2023/24 = gole (obecny pipeline)
  B) Hybrid:   2023/24 = observed Flash xG + imputed constrained xG
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
HYBRID_PATH = Path("data/processed/matches_2023_24_xg_hybrid.csv")
OUTPUT_PATH = Path("data/processed/backtesting_xg_hybrid_oos_2025_26.csv")
REPORT_PATH = Path("data/reports/model/backtesting_xg_hybrid_oos_report.txt")

MAX_GOLE = 10

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

SEASON_ORDER = {
    "2023/24": 1,
    "2024/25": 2,
    "2025/26": 3,
}


# =============================================================================
# POISSON MLE
# =============================================================================

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
        mu_home * alpha[data["home_idx"]] * beta[data["away_idx"]], 1e-10
    )
    la = np.maximum(
        mu_away * alpha[data["away_idx"]] * beta[data["home_idx"]], 1e-10
    )

    ll = np.sum(data["weights"] * (
        data["goals_home"] * np.log(lh) - lh +
        data["goals_away"] * np.log(la) - la
    ))
    return -ll


def przygotuj_dane(df, use_hybrid):
    df = df.copy()

    if use_hybrid and "xg_gosp_hybrid" in df.columns and "xg_gosc_hybrid" in df.columns:
        # 2023/24: hybrid xG
        # 2024/25, 2025/26: meczowe xG z bazy
        df["xg_final_home"] = np.where(
            df["sezon"] == "2023/24",
            df["xg_gosp_hybrid"],
            df["xg_gosp"].fillna(df["gole_gosp"])
        )
        df["xg_final_away"] = np.where(
            df["sezon"] == "2023/24",
            df["xg_gosc_hybrid"],
            df["xg_gosc"].fillna(df["gole_gosc"])
        )
    else:
        # baseline: fallback gole dla 2023/24
        df["xg_final_home"] = df["xg_gosp"].fillna(df["gole_gosp"])
        df["xg_final_away"] = df["xg_gosc"].fillna(df["gole_gosc"])

    druzyny = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    n = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    return {
        "n_druzyn": n,
        "idx_to_team": i2t,
        "home_idx": df["gospodarz"].map(t2i).values,
        "away_idx": df["gosc"].map(t2i).values,
        "goals_home": df["xg_final_home"].values.astype(float),
        "goals_away": df["xg_final_away"].values.astype(float),
        "weights": df["waga_sezonu"].values.astype(float),
    }


def trenuj_model(data):
    N = data["n_druzyn"]
    theta0 = np.zeros(2 + 2 * (N - 1))
    theta0[0] = np.log(np.mean(data["goals_home"]))
    theta0[1] = np.log(np.mean(data["goals_away"]))
    result = minimize(
        neg_log_likelihood, theta0, args=(data,),
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
        "alpha": {data["idx_to_team"][i]: np.exp(log_alpha[i]) for i in range(N)},
        "beta": {data["idx_to_team"][i]: np.exp(log_beta[i]) for i in range(N)},
    }


def przewiduj_mecz(params, gospodarz, gosc):
    a = params["alpha"]
    b = params["beta"]
    if gospodarz not in a or gosc not in a:
        return None

    lh = params["mu_home"] * a[gospodarz] * b[gosc]
    la = params["mu_away"] * a[gosc] * b[gospodarz]

    macierz = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)

    total = macierz.sum()
    if total <= 0:
        return None
    macierz /= total

    return {
        "p_home": float(np.sum(np.tril(macierz, -1))),
        "p_draw": float(np.sum(np.diag(macierz))),
        "p_away": float(np.sum(np.triu(macierz, 1))),
        "lambda_home": lh,
        "lambda_away": la,
    }


# =============================================================================
# KALIBRACJA
# =============================================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def calibrate(p_home, p_draw, p_away, T, bH, bD, bA):
    logits = np.log(np.maximum(np.array([p_home, p_draw, p_away]), 1e-12))
    return softmax((logits + np.array([bH, bD, bA])) / T)


def log_loss_1x2(p_vec, wynik):
    idx = {"H": 0, "D": 1, "A": 2}[wynik]
    return -np.log(max(float(p_vec[idx]), 1e-12))


def wynik_1x2(gh, ga):
    if gh > ga: return "H"
    elif gh == ga: return "D"
    return "A"


def optimize_calibration(df_raw):
    def objective(params):
        T, bH, bD, bA = params
        if T <= 0.1:
            return 999.0
        ll = []
        for row in df_raw.itertuples(index=False):
            p = calibrate(row.p_home_raw, row.p_draw_raw, row.p_away_raw, T, bH, bD, bA)
            ll.append(log_loss_1x2(p, row.wynik_1x2))
        return float(np.mean(ll))

    result = minimize(
        objective,
        x0=[1.5, 0.1, 0.0, -0.1],
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2)]
    )
    return result.x, result.fun


# =============================================================================
# ROLLING BACKTESTING
# =============================================================================

def run_rolling(df_all, target_season, use_hybrid, label=""):
    target_order = SEASON_ORDER[target_season]
    prev_seasons = [s for s, o in SEASON_ORDER.items() if o < target_order]

    df_test = df_all[df_all["sezon"] == target_season].copy()
    df_hist = df_all[df_all["sezon"].isin(prev_seasons)].copy()

    kolejki = sorted(df_test["kolejka"].unique())
    wyniki = []

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_trening = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane(df_trening, use_hybrid)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)

        for _, mecz in df_test[df_test["kolejka"] == kolejka].iterrows():
            gosp = mecz["gospodarz"]
            gosc = mecz["gosc"]
            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])

            pred = przewiduj_mecz(params, gosp, gosc)
            if pred is None:
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
                "use_hybrid": int(use_hybrid),
            })

        if kolejka % 5 == 1:
            print(f"  {label} K{kolejka:02d} OK")

    return pd.DataFrame(wyniki)


# =============================================================================
# MAIN
# =============================================================================

def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuję matches z DB...")
    conn = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()

    df_db["waga_sezonu"] = df_db["sezon"].map(WAGI_SEZONOW)

    print("2. Wczytuję hybrid xG 2023/24...")
    hybrid = pd.read_csv(HYBRID_PATH)
    hybrid_cols = ["match_id", "xg_gosp_hybrid", "xg_gosc_hybrid"]
    hybrid = hybrid[hybrid_cols].copy()

    df_all_base = df_db.copy()
    df_all_hybrid = df_db.merge(hybrid, on="match_id", how="left")

    # Sanity check: hybrid powinien mieć wartości dla wszystkich meczów 2023/24
    mask_2023 = df_all_hybrid["sezon"] == "2023/24"
    n_hybrid_ok = df_all_hybrid.loc[mask_2023, "xg_gosp_hybrid"].notna().sum()
    print(f"   Mecze 2023/24 z hybrid xG: {n_hybrid_ok}/306")

    # --- KALIBRACJA NA 2024/25 ---
    print("\n3. Kalibracja na 2024/25 — Baseline (gole fallback)...")
    df_calib_base = run_rolling(df_all_base, "2024/25", use_hybrid=False, label="CALIB_BASE")
    params_base, ll_calib_base = optimize_calibration(df_calib_base)

    print(f"\n4. Kalibracja na 2024/25 — Hybrid xG...")
    df_calib_hybrid = run_rolling(df_all_hybrid, "2024/25", use_hybrid=True, label="CALIB_HYB")
    params_hybrid, ll_calib_hybrid = optimize_calibration(df_calib_hybrid)

    print(f"\n   CALIB Baseline:  {ll_calib_base:.4f}")
    print(f"   CALIB Hybrid:    {ll_calib_hybrid:.4f}")

    # --- TEST OOS NA 2025/26 ---
    print("\n5. Test OOS na 2025/26 — Baseline...")
    df_test_base = run_rolling(df_all_base, "2025/26", use_hybrid=False, label="TEST_BASE")

    print(f"\n6. Test OOS na 2025/26 — Hybrid xG...")
    df_test_hybrid = run_rolling(df_all_hybrid, "2025/26", use_hybrid=True, label="TEST_HYB")

    # Aplikuj parametry z kalibracji
    def apply_cal(df_raw, params):
        T, bH, bD, bA = params
        ll_list = []
        for row in df_raw.itertuples(index=False):
            p = calibrate(row.p_home_raw, row.p_draw_raw, row.p_away_raw, T, bH, bD, bA)
            ll_list.append(log_loss_1x2(p, row.wynik_1x2))
        return ll_list

    ll_test_base = np.mean(apply_cal(df_test_base, params_base))
    ll_test_hybrid = np.mean(apply_cal(df_test_hybrid, params_hybrid))

    df_test_base["ll"] = apply_cal(df_test_base, params_base)
    df_test_hybrid["ll"] = apply_cal(df_test_hybrid, params_hybrid)

    ll_early_base = df_test_base[df_test_base["kolejka"] <= 5]["ll"].mean()
    ll_early_hybrid = df_test_hybrid[df_test_hybrid["kolejka"] <= 5]["ll"].mean()
    ll_late_base = df_test_base[df_test_base["kolejka"] > 5]["ll"].mean()
    ll_late_hybrid = df_test_hybrid[df_test_hybrid["kolejka"] > 5]["ll"].mean()

    T_A, bH_A, bD_A, bA_A = params_base
    T_B, bH_B, bD_B, bA_B = params_hybrid

    # Zapis
    df_test_hybrid["ll_baseline"] = apply_cal(df_test_base, params_base)
    df_test_hybrid.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    delta_calib = ll_calib_base - ll_calib_hybrid
    delta_oos = ll_test_base - ll_test_hybrid
    delta_early = ll_early_base - ll_early_hybrid
    delta_late = ll_late_base - ll_late_hybrid

    lines = []
    lines.append("=" * 78)
    lines.append("HYBRID xG 2023/24 — OOS TEST")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. SETUP")
    lines.append("-" * 78)
    lines.append("  Baseline: 2023/24 = gole (fallback)")
    lines.append("  Hybrid:   2023/24 = observed Flash xG + constrained imputation")
    lines.append("  Kalibracja parametrów: 2024/25")
    lines.append("  Test OOS:              2025/26")
    lines.append(f"  Predykcje kalibracja:  {len(df_calib_base)}")
    lines.append(f"  Predykcje test:        {len(df_test_base)}")
    lines.append("")
    lines.append("2. KALIBRACJA NA 2024/25")
    lines.append("-" * 78)
    lines.append(f"  Baseline log-loss: {ll_calib_base:.4f}")
    lines.append(f"  Hybrid log-loss:   {ll_calib_hybrid:.4f}")
    lines.append(f"  Delta:             {delta_calib:+.4f}")
    lines.append("")
    lines.append("3. TEST OOS NA 2025/26")
    lines.append("-" * 78)
    lines.append(f"  Baseline log-loss: {ll_test_base:.4f}")
    lines.append(f"  Hybrid log-loss:   {ll_test_hybrid:.4f}")
    lines.append(f"  Delta (base-hyb):  {delta_oos:+.4f}")
    lines.append(f"  Benchmark losowy:  {np.log(3):.4f}")
    lines.append("")
    lines.append("4. ANALIZA WCZESNY vs PÓŹNY SEZON")
    lines.append("-" * 78)
    lines.append(f"  Kolejki 1-5  baseline: {ll_early_base:.4f}")
    lines.append(f"  Kolejki 1-5  hybrid:   {ll_early_hybrid:.4f}")
    lines.append(f"  Delta early:           {delta_early:+.4f}")
    lines.append(f"  Kolejki 6-34 baseline: {ll_late_base:.4f}")
    lines.append(f"  Kolejki 6-34 hybrid:   {ll_late_hybrid:.4f}")
    lines.append(f"  Delta late:            {delta_late:+.4f}")
    lines.append("")
    lines.append("5. PARAMETRY KALIBRACJI")
    lines.append("-" * 78)
    lines.append(f"  Baseline: T={T_A:.4f}, bH={bH_A:.4f}, bD={bD_A:.4f}, bA={bA_A:.4f}")
    lines.append(f"  Hybrid:   T={T_B:.4f}, bH={bH_B:.4f}, bD={bD_B:.4f}, bA={bA_B:.4f}")
    lines.append("")
    lines.append("6. WNIOSEK")
    lines.append("-" * 78)

    if delta_oos > 0.005:
        lines.append(f"  WYRAŹNA POPRAWA OOS: {delta_oos:+.4f}")
        lines.append("  Hybrid xG 2023/24 realnie poprawia model.")
        lines.append("  Warto zaktualizować pipeline i bazę danych.")
    elif delta_oos > 0.001:
        lines.append(f"  MAŁA POPRAWA OOS: {delta_oos:+.4f}")
        lines.append("  Sygnał jest. Warto wdrożyć jeśli zależy nam na każdej setnej log-loss.")
    elif delta_oos > 0:
        lines.append(f"  MARGINALNA POPRAWA OOS: {delta_oos:+.4f}")
        lines.append("  Technicznie lepiej, ale różnica poniżej progu istotności.")
    else:
        lines.append(f"  BRAK POPRAWY OOS: {delta_oos:+.4f}")
        lines.append("  Hybrid xG nie poprawia modelu OOS.")
        lines.append("  Nie aktualizujemy bazy danych.")

    if delta_early > 0.005:
        lines.append(f"  SZCZEGÓLNA POPRAWA W KOL. 1-5: {delta_early:+.4f}")

    lines.append("")
    lines.append("7. PLIKI")
    lines.append("-" * 78)
    lines.append(f"  Predykcje: {OUTPUT_PATH}")
    lines.append(f"  Raport:    {REPORT_PATH}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {OUTPUT_PATH}")
    print(f"Zapisano: {REPORT_PATH}")


if __name__ == "__main__":
    main()