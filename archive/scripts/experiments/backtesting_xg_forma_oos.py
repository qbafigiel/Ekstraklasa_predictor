"""
backtesting_xg_forma_oos.py - PELNY GRID
=========================================
Uczciwy test formy xG.

Uruchamiamy WSZYSTKIE 96 kombinacji hiperparametrow.
Raportujemy:
  - mediane delty na tescie (czy forma systematycznie pomaga?)
  - procent konfiguracji lepszych na tescie
  - top 5 konfiguracji
Jesli >60% konfiguracji lepszych na tescie -> forma ma realny sygnal.
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
REPORT_PATH = Path("data/reports/model/backtesting_xg_forma_oos_report.txt")
OUTPUT_PATH = Path("data/processed/backtesting_xg_forma_oos_2025_26.csv")

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"
MAX_GOLE = 10
OFFICIAL_BASELINE = 1.0653

WAGI_SEZONOW = {"2023/24": 0.4, "2024/25": 0.7, "2025/26": 1.0}
K_PROMOTED_PRIOR = 10

PROMOTED_BY_SEASON = {
    "2024/25": ["GKS Katowice", "Lechia Gdansk", "Motor Lublin"],
    "2025/26": ["Arka Gdynia", "Bruk-Bet Termalica Nieciecza", "Wisla Plock"],
}
PRIORS_BY_SEASON = {
    "2024/25": {"prior_atak": 0.80, "prior_obrona": 1.10},
    "2025/26": {"prior_atak": 0.7744, "prior_obrona": 1.0648},
}

# Pelny grid - 4 x 4 x 3 x 3 = 144 kombinacji
WINDOWS = [3, 5, 8, 10]
ETAS = [0.25, 0.50, 0.75, 1.00]
K_FORMS = [3, 5, 8]
CAPS = [1.25, 1.35, 1.50]

CONFIGS = [
    (w, e, k, c, f"w{w}_e{e}_k{k}_c{c}")
    for w in WINDOWS
    for e in ETAS
    for k in K_FORMS
    for c in CAPS
]


# =============================================================================
# MODEL POISSON
# =============================================================================

def przygotuj_dane_xg(df_train):
    df = df_train.copy()
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
    lh = np.maximum(mu_home * alpha[data["home_idx"]] * beta[data["away_idx"]], 1e-10)
    la = np.maximum(mu_away * alpha[data["away_idx"]] * beta[data["home_idx"]], 1e-10)
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
        "alpha": {data["idx_to_team"][i]: np.exp(log_alpha[i]) for i in range(N)},
        "beta": {data["idx_to_team"][i]: np.exp(log_beta[i]) for i in range(N)},
    }


def zastosuj_prior_beniaminkow(params, season, df_prev):
    promoted = PROMOTED_BY_SEASON.get(season, [])
    prior = PRIORS_BY_SEASON.get(season, None)
    if not promoted or prior is None:
        return params
    for team in promoted:
        n = int(((df_prev["gospodarz"] == team) | (df_prev["gosc"] == team)).sum())
        a = params["alpha"].get(team, prior["prior_atak"])
        b = params["beta"].get(team, prior["prior_obrona"])
        params["alpha"][team] = (K_PROMOTED_PRIOR * prior["prior_atak"] + n * a) / (K_PROMOTED_PRIOR + n)
        params["beta"][team] = (K_PROMOTED_PRIOR * prior["prior_obrona"] + n * b) / (K_PROMOTED_PRIOR + n)
    return params


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


# =============================================================================
# FORMA
# =============================================================================

def form_multiplier(history, window, eta, k_form, cap):
    if eta <= 0 or len(history) == 0:
        return 1.0
    recent = history[-window:]
    n = len(recent)
    actual = sum(x["actual"] for x in recent)
    expected = sum(x["expected"] for x in recent)
    if expected <= 1e-12:
        return 1.0
    ratio = float(np.clip(actual / expected, 1.0 / cap, cap))
    weight = n / (k_form + n)
    return float(np.exp(eta * weight * np.log(ratio)))


# =============================================================================
# ROLLING BACKTESTING - BAZOWE LAMBDY
# =============================================================================

def build_base_context(df_all, target_season):
    seasons = sorted(df_all["sezon"].unique())
    idx = seasons.index(target_season)
    prev_seasons = seasons[:idx]
    df_hist = df_all[df_all["sezon"].isin(prev_seasons)].copy()
    df_test = df_all[df_all["sezon"] == target_season].copy()
    kolejki = sorted(df_test["kolejka"].unique())
    rows = []

    print(f"  Buduje kontekst bazowy dla {target_season}...")
    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka].copy()
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)
        data = przygotuj_dane_xg(df_train)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)
        params = zastosuj_prior_beniaminkow(params, target_season, df_prev)

        for _, mecz in df_test[df_test["kolejka"] == kolejka].iterrows():
            home = mecz["gospodarz"]
            away = mecz["gosc"]
            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])
            xgh = float(mecz["xg_gosp"]) if pd.notna(mecz["xg_gosp"]) else float(gh)
            xga = float(mecz["xg_gosc"]) if pd.notna(mecz["xg_gosc"]) else float(ga)

            if home not in params["alpha"] or away not in params["alpha"]:
                continue

            lh = params["mu_home"] * params["alpha"][home] * params["beta"][away]
            la = params["mu_away"] * params["alpha"][away] * params["beta"][home]

            macierz = np.zeros((MAX_GOLE, MAX_GOLE))
            for i in range(MAX_GOLE):
                for j in range(MAX_GOLE):
                    macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
            total = macierz.sum()
            if total <= 0:
                continue
            macierz /= total

            rows.append({
                "sezon": target_season,
                "kolejka": int(kolejka),
                "gospodarz": home,
                "gosc": away,
                "gole_gosp": gh,
                "gole_gosc": ga,
                "wynik_1x2": wynik_1x2(gh, ga),
                "xg_gosp_actual": xgh,
                "xg_gosc_actual": xga,
                "lambda_home_base": float(lh),
                "lambda_away_base": float(la),
                "p_home_base": float(np.sum(np.tril(macierz, -1))),
                "p_draw_base": float(np.sum(np.diag(macierz))),
                "p_away_base": float(np.sum(np.triu(macierz, 1))),
            })

    return pd.DataFrame(rows)


# =============================================================================
# WARSTWA FORMY
# =============================================================================

def apply_form_layer(df_base, window, eta, k_form, cap):
    df_base = df_base.sort_values(["kolejka", "gospodarz"]).reset_index(drop=True)
    teams = sorted(set(df_base["gospodarz"]) | set(df_base["gosc"]))
    history = {t: {"atk": [], "def": []} for t in teams}
    rows = []

    for kolejka in sorted(df_base["kolejka"].unique()):
        df_round = df_base[df_base["kolejka"] == kolejka]
        pending = []

        for _, row in df_round.iterrows():
            home = row["gospodarz"]
            away = row["gosc"]

            home_atk = form_multiplier(history[home]["atk"], window, eta, k_form, cap)
            away_atk = form_multiplier(history[away]["atk"], window, eta, k_form, cap)
            home_def = form_multiplier(history[home]["def"], window, eta, k_form, cap)
            away_def = form_multiplier(history[away]["def"], window, eta, k_form, cap)

            lh = row["lambda_home_base"] * home_atk * away_def
            la = row["lambda_away_base"] * away_atk * home_def

            macierz = np.zeros((MAX_GOLE, MAX_GOLE))
            for i in range(MAX_GOLE):
                for j in range(MAX_GOLE):
                    macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
            total = macierz.sum()
            if total <= 0:
                continue
            macierz /= total

            p_home = float(np.sum(np.tril(macierz, -1)))
            p_draw = float(np.sum(np.diag(macierz)))
            p_away = float(np.sum(np.triu(macierz, 1)))

            rows.append({
                **row.to_dict(),
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
            })

            pending.append({
                "team": home,
                "atk_actual": row["xg_gosp_actual"],
                "atk_expected": row["lambda_home_base"],
                "def_actual": row["xg_gosc_actual"],
                "def_expected": row["lambda_away_base"],
            })
            pending.append({
                "team": away,
                "atk_actual": row["xg_gosc_actual"],
                "atk_expected": row["lambda_away_base"],
                "def_actual": row["xg_gosp_actual"],
                "def_expected": row["lambda_home_base"],
            })

        for upd in pending:
            history[upd["team"]]["atk"].append({"actual": upd["atk_actual"], "expected": upd["atk_expected"]})
            history[upd["team"]]["def"].append({"actual": upd["def_actual"], "expected": upd["def_expected"]})

    return pd.DataFrame(rows)


# =============================================================================
# KALIBRACJA
# =============================================================================

def fit_calibrator(df, p_col_h="p_home", p_col_d="p_draw", p_col_a="p_away"):
    def objective(params):
        T, bH, bD, bA = params
        if T <= 0.1:
            return 999.0
        ll = []
        for row in df.itertuples(index=False):
            ph = getattr(row, p_col_h)
            pd_ = getattr(row, p_col_d)
            pa = getattr(row, p_col_a)
            p = calibrate(ph, pd_, pa, T, bH, bD, bA)
            ll.append(log_loss_1x2(p[0], p[1], p[2], row.wynik_1x2))
        return float(np.mean(ll))

    result = minimize(
        objective, x0=[1.5, 0.1, 0.0, -0.1],
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-2, 2), (-2, 2), (-2, 2)]
    )
    return result.x, result.fun


def apply_calibrator(df, params, p_col_h="p_home", p_col_d="p_draw", p_col_a="p_away"):
    T, bH, bD, bA = params
    ll_list = []
    for row in df.itertuples(index=False):
        ph = getattr(row, p_col_h)
        pd_ = getattr(row, p_col_d)
        pa = getattr(row, p_col_a)
        p = calibrate(ph, pd_, pa, T, bH, bD, bA)
        ll_list.append(log_loss_1x2(p[0], p[1], p[2], row.wynik_1x2))
    return ll_list


# =============================================================================
# MAIN
# =============================================================================

def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuje matches...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    print("\n2. Buduje bazowe konteksty (Poisson MLE rolling)...")
    df_base_val = build_base_context(df, VAL_SEASON)
    df_base_test = build_base_context(df, TEST_SEASON)

    print("\n3. Kalibruje baseline (bez formy)...")
    params_base_cal, ll_base_val = fit_calibrator(
        df_base_val, "p_home_base", "p_draw_base", "p_away_base"
    )
    ll_base_test = np.mean(apply_calibrator(
        df_base_test, params_base_cal, "p_home_base", "p_draw_base", "p_away_base"
    ))
    print(f"   Baseline val:  {ll_base_val:.4f}")
    print(f"   Baseline test: {ll_base_test:.4f}")

    print(f"\n4. Testuje {len(CONFIGS)} konfiguracji formy...")
    results = []

    for i, (window, eta, k_form, cap, label) in enumerate(CONFIGS):
        df_val_forma = apply_form_layer(df_base_val, window, eta, k_form, cap)
        params_forma_cal, ll_forma_val = fit_calibrator(df_val_forma)

        df_test_forma = apply_form_layer(df_base_test, window, eta, k_form, cap)
        ll_forma_test = np.mean(apply_calibrator(df_test_forma, params_forma_cal))

        delta_val = ll_base_val - ll_forma_val
        delta_test = ll_base_test - ll_forma_test

        results.append({
            "label": label,
            "window": window,
            "eta": eta,
            "k_form": k_form,
            "cap": cap,
            "ll_val_baseline": ll_base_val,
            "ll_val_forma": ll_forma_val,
            "delta_val": delta_val,
            "ll_test_baseline": ll_base_test,
            "ll_test_forma": ll_forma_test,
            "delta_test": delta_test,
        })

        print(f"  [{i+1:03d}/{len(CONFIGS)}] {label:<22} val={delta_val:+.4f}  test={delta_test:+.4f}")

    df_results = pd.DataFrame(results).sort_values("delta_test", ascending=False)
    df_results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Statystyki
    n_total = len(results)
    n_better = sum(1 for r in results if r["delta_test"] > 0)
    n_better_001 = sum(1 for r in results if r["delta_test"] > 0.001)
    pct_better = 100 * n_better / n_total
    median_delta = float(np.median([r["delta_test"] for r in results]))
    mean_delta = float(np.mean([r["delta_test"] for r in results]))

    top5 = df_results.head(5)

    lines = []
    lines.append("=" * 78)
    lines.append("FORMA xG - PELNY GRID OOS TEST")
    lines.append("=" * 78)
    lines.append("")
    lines.append("METODOLOGIA")
    lines.append("-" * 78)
    lines.append(f"  Kombinacji grid:        {n_total}")
    lines.append("  Kalibracja:             2024/25")
    lines.append("  Test OOS:               2025/26")
    lines.append(f"  Oficjalny OOS baseline: {OFFICIAL_BASELINE}")
    lines.append("")
    lines.append("STATYSTYKI ZBIORCZE")
    lines.append("-" * 78)
    lines.append(f"  Lepsza od baseline:     {n_better}/{n_total} ({pct_better:.1f}%)")
    lines.append(f"  Lepsza o >0.001:        {n_better_001}/{n_total}")
    lines.append(f"  Mediana delta test:     {median_delta:+.4f}")
    lines.append(f"  Srednia delta test:     {mean_delta:+.4f}")
    lines.append("")
    lines.append("TOP 5 KONFIGURACJI (test delta)")
    lines.append("-" * 78)
    lines.append(f"  {'Konfiguracja':<25} {'Val delta':>10} {'Test delta':>10} {'Test LL':>10}")
    lines.append("  " + "-" * 60)
    for _, r in top5.iterrows():
        lines.append(
            f"  {r['label']:<25} {r['delta_val']:>+10.4f} {r['delta_test']:>+10.4f} {r['ll_test_forma']:>10.4f}"
        )
    lines.append("")
    lines.append("BOTTOM 5 KONFIGURACJI (test delta)")
    lines.append("-" * 78)
    bottom5 = df_results.tail(5)
    lines.append(f"  {'Konfiguracja':<25} {'Val delta':>10} {'Test delta':>10} {'Test LL':>10}")
    lines.append("  " + "-" * 60)
    for _, r in bottom5.iterrows():
        lines.append(
            f"  {r['label']:<25} {r['delta_val']:>+10.4f} {r['delta_test']:>+10.4f} {r['ll_test_forma']:>10.4f}"
        )
    lines.append("")
    lines.append(f"  Baseline test:          {ll_base_test:.4f}")
    lines.append(f"  Oficjalny OOS baseline: {OFFICIAL_BASELINE:.4f}")
    lines.append(f"  Benchmark losowy:       {np.log(3):.4f}")
    lines.append("")
    lines.append("WNIOSEK")
    lines.append("-" * 78)
    if pct_better >= 60 and median_delta > 0:
        lines.append(f"  FORMA MA SYGNAL: {pct_better:.1f}% konfiguracji lepszych, mediana={median_delta:+.4f}")
        lines.append("  Rekomendacja: wdrozyc najlepsza konfiguracje.")
    elif pct_better >= 50:
        lines.append(f"  SYGNAL SLABY/NIEJEDNOZNACZNY: {pct_better:.1f}% lepszych, mediana={median_delta:+.4f}")
        lines.append("  Forma moze nieznacznie pomagac ale ryzyko overfitu wysokie.")
        lines.append("  Rekomendacja: nie wdrazac.")
    else:
        lines.append(f"  BRAK SYGNALU: tylko {pct_better:.1f}% konfiguracji lepszych, mediana={median_delta:+.4f}")
        lines.append("  Forma nie pomaga systematycznie.")
        lines.append("  Rekomendacja: zamknac temat.")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {REPORT_PATH}")
    print(f"Zapisano: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()