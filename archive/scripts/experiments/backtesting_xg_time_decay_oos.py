"""
backtesting_xg_time_decay_oos.py
=================================
Uczciwy test Time Decay na xG.

Różnica vs poprzedni backtest_time_decay.py:
  - trenuje na xG (nie na golach)
  - zachowuje wagi sezonowe (0.4 / 0.7 / 1.0)
  - decay MNOŻY wagi sezonowe (nie zastępuje)
  - kalibracja OOS: uczona na 2024/25, testowana na 2025/26

Decay:
  waga_finalna = waga_sezonu * exp(-decay * age_in_matches)
  age_in_matches = pozycja wstecz od najnowszego meczu w zbiorze treningowym

Grid search decay: [0.0, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]

Porównanie z baseline (bez decay):
  Baseline best = 1.0571
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
REPORT_PATH = Path("data/reports/model/backtesting_xg_time_decay_oos_report.txt")
OUTPUT_PATH = Path("data/processed/backtesting_xg_time_decay_oos_best.csv")

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"
MAX_GOLE = 10
BASELINE_BEST = 1.0571

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

K_PROMOTED_PRIOR = 10

PROMOTED_BY_SEASON = {
    "2024/25": ["GKS Katowice", "Lechia Gdańsk", "Motor Lublin"],
    "2025/26": ["Arka Gdynia", "Bruk-Bet Termalica Nieciecza", "Wisła Płock"],
}

PRIORS_BY_SEASON = {
    "2024/25": {"prior_atak": 0.80, "prior_obrona": 1.10},
    "2025/26": {"prior_atak": 0.7744, "prior_obrona": 1.0648},
}

DECAY_GRID = [0.0, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]


# =============================================================================
# WAGI DECAY — MNOŻNIK DO WAG SEZONOWYCH
# =============================================================================

def compute_combined_weights(df_train, decay):
    """
    waga_finalna = waga_sezonu * exp(-decay * age_in_matches)

    age_in_matches = 0 dla najnowszego meczu, rośnie wstecz.
    decay=0 -> waga_finalna = waga_sezonu (identyczne z baseline).
    """
    df = df_train.sort_values(
        ["sezon", "kolejka"],
        ascending=[True, True]
    ).reset_index(drop=True)

    n = len(df)
    age = (n - 1) - np.arange(n)  # 0 = najnowszy
    decay_weights = np.exp(-decay * age)
    season_weights = df["waga_sezonu"].values.astype(float)
    combined = season_weights * decay_weights

    # normalizacja: średnia = 1.0 żeby nie skalować MLE
    mean_w = combined.mean()
    if mean_w > 0:
        combined = combined / mean_w

    return combined


# =============================================================================
# MODEL POISSON NA xG — identyczny z best modelem
# =============================================================================

def przygotuj_dane_xg(df_train, decay=0.0):
    df = df_train.copy()
    df["xg_final_home"] = df["xg_gosp"].fillna(df["gole_gosp"])
    df["xg_final_away"] = df["xg_gosc"].fillna(df["gole_gosc"])

    druzyny = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    n = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    if decay > 0:
        weights = compute_combined_weights(df, decay)
    else:
        weights = df["waga_sezonu"].values.astype(float)

    return {
        "n_druzyn": n,
        "idx_to_team": i2t,
        "home_idx": df["gospodarz"].map(t2i).values,
        "away_idx": df["gosc"].map(t2i).values,
        "goals_home": df["xg_final_home"].values.astype(float),
        "goals_away": df["xg_final_away"].values.astype(float),
        "weights": weights,
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


def trenuj_model(data):
    N = data["n_druzyn"]
    theta0 = np.zeros(2 + 2 * (N - 1))
    theta0[0] = np.log(np.mean(data["goals_home"]))
    theta0[1] = np.log(np.mean(data["goals_away"]))
    result = minimize(
        neg_log_likelihood, theta0, args=(data,),
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-12}
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


def zastosuj_prior_beniaminkow(params, season, df_prev):
    promoted = PROMOTED_BY_SEASON.get(season, [])
    prior = PRIORS_BY_SEASON.get(season, None)
    if not promoted or prior is None:
        return params

    for team in promoted:
        n = int(
            ((df_prev["gospodarz"] == team) |
             (df_prev["gosc"] == team)).sum()
        )
        a = params["alpha"].get(team, prior["prior_atak"])
        b = params["beta"].get(team, prior["prior_obrona"])
        params["alpha"][team] = (
            K_PROMOTED_PRIOR * prior["prior_atak"] + n * a
        ) / (K_PROMOTED_PRIOR + n)
        params["beta"][team] = (
            K_PROMOTED_PRIOR * prior["prior_obrona"] + n * b
        ) / (K_PROMOTED_PRIOR + n)

    return params


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
# HELPERS
# =============================================================================

def wynik_1x2(gh, ga):
    if gh > ga: return "H"
    elif gh == ga: return "D"
    return "A"


def log_loss_1x2(ph, pd_, pa, wynik):
    p = {"H": ph, "D": pd_, "A": pa}[wynik]
    return -np.log(max(float(p), 1e-12))


def softmax(x):
    e = np.exp(np.asarray(x, dtype=float) - np.max(x))
    return e / e.sum()


def calibrate(ph, pd_, pa, T, bH, bD, bA):
    logits = np.log(np.maximum([ph, pd_, pa], 1e-12))
    return softmax((logits + np.array([bH, bD, bA])) / T)


def fit_calibrator(df_raw):
    def objective(params):
        T, bH, bD, bA = params
        if T <= 0.1:
            return 999.0
        ll = []
        for row in df_raw.itertuples(index=False):
            p = calibrate(
                row.p_home_raw, row.p_draw_raw, row.p_away_raw,
                T, bH, bD, bA
            )
            ll.append(log_loss_1x2(p[0], p[1], p[2], row.wynik_1x2))
        return float(np.mean(ll))

    result = minimize(
        objective,
        x0=[1.5, 0.1, 0.0, -0.1],
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2)]
    )
    return result.x, result.fun


def apply_calibrator(df_raw, params):
    T, bH, bD, bA = params
    ll_list = []
    for row in df_raw.itertuples(index=False):
        p = calibrate(
            row.p_home_raw, row.p_draw_raw, row.p_away_raw,
            T, bH, bD, bA
        )
        ll_list.append(log_loss_1x2(p[0], p[1], p[2], row.wynik_1x2))
    return ll_list


# =============================================================================
# ROLLING BACKTESTING
# =============================================================================

def run_rolling(df_all, target_season, decay, verbose=False):
    seasons = sorted(df_all["sezon"].unique())
    idx = seasons.index(target_season)
    prev_seasons = seasons[:idx]

    df_hist = df_all[df_all["sezon"].isin(prev_seasons)].copy()
    df_test = df_all[df_all["sezon"] == target_season].copy()
    kolejki = sorted(df_test["kolejka"].unique())

    rows = []

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane_xg(df_train, decay)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)
        params = zastosuj_prior_beniaminkow(params, target_season, df_prev)

        for _, mecz in df_test[df_test["kolejka"] == kolejka].iterrows():
            gosp = mecz["gospodarz"]
            gosc = mecz["gosc"]
            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])

            pred = przewiduj_mecz(params, gosp, gosc)
            if pred is None:
                continue

            w = wynik_1x2(gh, ga)

            rows.append({
                "sezon": target_season,
                "match_id": mecz["match_id"],
                "kolejka": int(kolejka),
                "gospodarz": gosp,
                "gosc": gosc,
                "wynik_1x2": w,
                "p_home_raw": pred["p_home"],
                "p_draw_raw": pred["p_draw"],
                "p_away_raw": pred["p_away"],
                "lambda_home": pred["lambda_home"],
                "lambda_away": pred["lambda_away"],
                "decay": decay,
            })

        if verbose:
            print(f"  decay={decay:.3f} {target_season} K{kolejka:02d} OK")

    return pd.DataFrame(rows)


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

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    # ==========================================================================
    # GRID SEARCH NA 2024/25
    # ==========================================================================
    print(f"\n2. Grid search decay na {VAL_SEASON}...")
    print(f"   Grid: {DECAY_GRID}")
    print()

    grid_results = []

    for decay in DECAY_GRID:
        df_val_raw = run_rolling(df, VAL_SEASON, decay, verbose=False)
        params_cal, ll_cal = fit_calibrator(df_val_raw)
        grid_results.append({
            "decay": decay,
            "ll_val_raw": df_val_raw["p_home_raw"].apply(
                lambda x: 0
            ).mean(),  # placeholder
            "ll_val_cal": ll_cal,
            "params": params_cal,
            "df_raw": df_val_raw,
        })
        # policz raw ll ręcznie
        ll_raw = np.mean([
            log_loss_1x2(r.p_home_raw, r.p_draw_raw, r.p_away_raw, r.wynik_1x2)
            for r in df_val_raw.itertuples(index=False)
        ])
        grid_results[-1]["ll_val_raw"] = ll_raw

        print(f"   decay={decay:.3f} | ll_val_raw={ll_raw:.4f} | ll_val_cal={ll_cal:.4f}")

    # znajdź najlepszy decay po ll_val_cal
    best = min(grid_results, key=lambda x: x["ll_val_cal"])
    best_decay = best["decay"]
    best_params_cal = best["params"]
    best_ll_val_cal = best["ll_val_cal"]

    print(f"\n   >>> Najlepszy decay: {best_decay} (ll_val_cal={best_ll_val_cal:.4f})")

    # ==========================================================================
    # TEST OOS NA 2025/26
    # ==========================================================================
    print(f"\n3. Test OOS na {TEST_SEASON} z decay={best_decay}...")

    # baseline (decay=0)
    df_test_baseline = run_rolling(df, TEST_SEASON, 0.0, verbose=False)
    params_baseline = next(
        r["params"] for r in grid_results if r["decay"] == 0.0
    )

    # best decay
    df_test_best = run_rolling(df, TEST_SEASON, best_decay, verbose=True)

    ll_test_baseline = np.mean(
        apply_calibrator(df_test_baseline, params_baseline)
    )
    ll_test_best = np.mean(
        apply_calibrator(df_test_best, best_params_cal)
    )

    # analiza per kolejka
    df_test_best["ll_cal"] = apply_calibrator(df_test_best, best_params_cal)
    df_test_baseline["ll_cal"] = apply_calibrator(
        df_test_baseline, params_baseline
    )

    ll_early_base = df_test_baseline[
        df_test_baseline["kolejka"] <= 5
    ]["ll_cal"].mean()
    ll_early_best = df_test_best[
        df_test_best["kolejka"] <= 5
    ]["ll_cal"].mean()
    ll_late_base = df_test_baseline[
        df_test_baseline["kolejka"] > 5
    ]["ll_cal"].mean()
    ll_late_best = df_test_best[
        df_test_best["kolejka"] > 5
    ]["ll_cal"].mean()

    T_base, bH_base, bD_base, bA_base = params_baseline
    T_best, bH_best, bD_best, bA_best = best_params_cal

    # zapis CSV
    df_test_best.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # ==========================================================================
    # RAPORT
    # ==========================================================================
    lines = []
    lines.append("=" * 78)
    lines.append("TIME DECAY NA xG — OOS TEST")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. SETUP")
    lines.append("-" * 78)
    lines.append("  Model:       Poisson(xG) + wagi sezonowe * exp(-decay * age)")
    lines.append("  Kalibracja:  Softmax uczona na 2024/25")
    lines.append("  Test OOS:    2025/26")
    lines.append(f"  Grid decay:  {DECAY_GRID}")
    lines.append("")
    lines.append("2. GRID SEARCH NA 2024/25")
    lines.append("-" * 78)
    for r in grid_results:
        marker = " <- BEST" if r["decay"] == best_decay else ""
        lines.append(
            f"  decay={r['decay']:.3f} | "
            f"ll_val_raw={r['ll_val_raw']:.4f} | "
            f"ll_val_cal={r['ll_val_cal']:.4f}{marker}"
        )
    lines.append("")
    lines.append("3. TEST OOS NA 2025/26")
    lines.append("-" * 78)
    lines.append(f"  Baseline (decay=0.0):       {ll_test_baseline:.4f}")
    lines.append(f"  Best decay ({best_decay:.3f}):        {ll_test_best:.4f}")
    lines.append(f"  Delta (base - best):        {ll_test_baseline - ll_test_best:+.4f}")
    lines.append(f"  Poprzedni best (baseline):  {BASELINE_BEST}")
    lines.append(f"  Delta vs poprzedni best:    {BASELINE_BEST - ll_test_best:+.4f}")
    lines.append(f"  Benchmark losowy:           {np.log(3):.4f}")
    lines.append("")
    lines.append("4. WCZESNY vs PÓŹNY SEZON")
    lines.append("-" * 78)
    lines.append(f"  Kolejki 1-5  baseline:      {ll_early_base:.4f}")
    lines.append(f"  Kolejki 1-5  best decay:    {ll_early_best:.4f}")
    lines.append(f"  Delta early:                {ll_early_base - ll_early_best:+.4f}")
    lines.append(f"  Kolejki 6-34 baseline:      {ll_late_base:.4f}")
    lines.append(f"  Kolejki 6-34 best decay:    {ll_late_best:.4f}")
    lines.append(f"  Delta late:                 {ll_late_base - ll_late_best:+.4f}")
    lines.append("")
    lines.append("5. PARAMETRY KALIBRACJI")
    lines.append("-" * 78)
    lines.append(
        f"  Baseline: T={T_base:.4f}, bH={bH_base:.4f}, "
        f"bD={bD_base:.4f}, bA={bA_base:.4f}"
    )
    lines.append(
        f"  Best:     T={T_best:.4f}, bH={bH_best:.4f}, "
        f"bD={bD_best:.4f}, bA={bA_best:.4f}"
    )
    lines.append("")
    lines.append("6. WNIOSEK")
    lines.append("-" * 78)

    delta = BASELINE_BEST - ll_test_best
    delta_vs_internal = ll_test_baseline - ll_test_best

    if delta > 0.005:
        lines.append(f"  WYRAŹNA POPRAWA vs poprzedni best: {delta:+.4f}")
        lines.append(f"  Time Decay na xG poprawia model.")
        lines.append(f"  Rekomendacja: WDROŻYĆ do pipeline produkcyjnego.")
    elif delta > 0.001:
        lines.append(f"  MAŁA POPRAWA vs poprzedni best: {delta:+.4f}")
        lines.append(f"  Sygnał jest, ale niewielki.")
        lines.append(f"  Rekomendacja: WDROŻYĆ jeśli zależy nam na każdej setnej.")
    elif delta > 0:
        lines.append(f"  MARGINALNA POPRAWA: {delta:+.4f}")
        lines.append(f"  Poniżej progu istotności.")
        lines.append(f"  Rekomendacja: nie wdrażać.")
    else:
        lines.append(f"  BRAK POPRAWY vs poprzedni best: {delta:+.4f}")
        lines.append(f"  Time Decay na xG nie poprawia modelu OOS.")
        lines.append(f"  Rekomendacja: zamknąć temat.")

    if delta_vs_internal > 0.002:
        lines.append(f"  UWAGA: decay poprawia vs baseline wewnętrzny: {delta_vs_internal:+.4f}")
        lines.append(f"  Różnica vs poprzedni best może wynikać z różnicy setupów.")

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