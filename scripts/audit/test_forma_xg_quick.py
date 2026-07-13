"""
test_forma_xg_quick.py
======================
Szybki test najlepszej konfiguracji formy z kalibratora_xg_form_best.json.
Bez grid search — tylko jeden run z best_cfg.

Kalibracja: 2024/25
Test OOS:   2025/26
"""

import sqlite3
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

DB_PATH = Path("db/ekstraklasa.db")
CALIB_JSON = Path("data/processed/kalibrator_xg_form_best.json")
REPORT_PATH = Path("data/reports/model/test_forma_xg_quick_report.txt")

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"
MAX_GOLE = 10

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


# =============================================================================
# MODEL BAZOWY — identyczny z best modelem
# =============================================================================

def przygotuj_dane_xg(df_trening):
    df = df_trening.copy()
    df["xg_gosp_final"] = df["xg_gosp"].fillna(df["gole_gosp"])
    df["xg_gosc_final"] = df["xg_gosc"].fillna(df["gole_gosc"])

    druzyny = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    n = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    return {
        "n_druzyn": n,
        "idx_to_team": i2t,
        "home_idx": df["gospodarz"].map(t2i).values,
        "away_idx": df["gosc"].map(t2i).values,
        "goals_home": df["xg_gosp_final"].values.astype(float),
        "goals_away": df["xg_gosc_final"].values.astype(float),
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
    result = minimize(neg_log_likelihood, theta0, args=(data,), method="L-BFGS-B",
                      options={"maxiter": 10000})
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


def lambda_base(params, home, away):
    lh = params["mu_home"] * params["alpha"][home] * params["beta"][away]
    la = params["mu_away"] * params["alpha"][away] * params["beta"][home]
    return float(lh), float(la)


def probs_from_lambdas(lh, la):
    macierz = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    total = macierz.sum()
    if total <= 0:
        return None
    macierz /= total
    return (
        float(np.sum(np.tril(macierz, -1))),
        float(np.sum(np.diag(macierz))),
        float(np.sum(np.triu(macierz, 1))),
    )


def wynik_1x2(gh, ga):
    if gh > ga: return "H"
    elif gh == ga: return "D"
    return "A"


def log_loss_val(ph, pd_, pa, wynik):
    p = {"H": ph, "D": pd_, "A": pa}[wynik]
    return -np.log(max(p, 1e-12))


# =============================================================================
# FORMA
# =============================================================================

def form_multiplier(history, window, eta, k_form, cap):
    if eta <= 0 or len(history) == 0:
        return 1.0

    recent = history[-window:] if window is not None else history
    n = len(recent)
    actual = sum(x["actual"] for x in recent)
    expected = sum(x["expected"] for x in recent)

    if expected <= 1e-12:
        return 1.0

    ratio = float(np.clip(actual / expected, 1.0 / cap, cap))
    weight = n / (k_form + n)
    return float(np.exp(eta * weight * np.log(ratio)))


# =============================================================================
# SOFTMAX CALIBRATION
# =============================================================================

def softmax(x):
    e = np.exp(np.asarray(x, dtype=float) - np.max(x))
    return e / e.sum()


def calibrate(ph, pd_, pa, T, bH, bD, bA):
    logits = np.log(np.maximum([ph, pd_, pa], 1e-12))
    return softmax((logits + np.array([bH, bD, bA])) / T)


def calibration_loss(params, raw_probs, outcomes):
    T, bH, bD, bA = params
    if T <= 0:
        return 1e9
    total = 0.0
    for (ph, pd_, pa), w in zip(raw_probs, outcomes):
        p_cal = calibrate(ph, pd_, pa, T, bH, bD, bA)
        total += -np.log(max(p_cal[w], 1e-12))
    return total / len(outcomes)


def fit_calibrator(df):
    raw = df[["p_home", "p_draw", "p_away"]].values.astype(float)
    y = df["wynik_1x2"].map({"H": 0, "D": 1, "A": 2}).values
    result = minimize(
        calibration_loss,
        [1.0, 0.0, 0.0, 0.0],
        args=(raw, y),
        method="L-BFGS-B",
        bounds=[(0.25, 6.0), (None, None), (None, None), (None, None)],
        options={"maxiter": 10000}
    )
    T, bH, bD, bA = result.x
    return {"T": float(T), "bH": float(bH), "bD": float(bD), "bA": float(bA)}


# =============================================================================
# ROLLING BACKTESTING Z FORMĄ
# =============================================================================

def run_season(df_all, target_season, cfg):
    window = cfg["window"]
    eta = cfg["eta_form"]
    k_form = cfg["k_form"]
    cap = cfg["cap_form"]

    seasons = sorted(df_all["sezon"].unique())
    idx = seasons.index(target_season)
    prev_seasons = seasons[:idx]

    df_hist = df_all[df_all["sezon"].isin(prev_seasons)].copy()
    df_test = df_all[df_all["sezon"] == target_season].copy()
    kolejki = sorted(df_test["kolejka"].unique())

    teams = sorted(set(df_test["gospodarz"]) | set(df_test["gosc"]))
    history = {t: {"atk": [], "def_": []} for t in teams}

    rows = []

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane_xg(df_train)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)
        params = zastosuj_prior_beniaminkow(params, target_season, df_prev)

        df_round = df_test[df_test["kolejka"] == kolejka]
        pending = []

        for _, mecz in df_round.iterrows():
            home = mecz["gospodarz"]
            away = mecz["gosc"]

            if home not in params["alpha"] or away not in params["alpha"]:
                continue

            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])
            xgh = float(mecz["xg_gosp"]) if pd.notna(mecz.get("xg_gosp")) else float(gh)
            xga = float(mecz["xg_gosc"]) if pd.notna(mecz.get("xg_gosc")) else float(ga)

            lh_base, la_base = lambda_base(params, home, away)

            # Forma
            home_atk = form_multiplier(history[home]["atk"], window, eta, k_form, cap)
            away_atk = form_multiplier(history[away]["atk"], window, eta, k_form, cap)
            home_def = form_multiplier(history[home]["def_"], window, eta, k_form, cap)
            away_def = form_multiplier(history[away]["def_"], window, eta, k_form, cap)

            lh = lh_base * home_atk * away_def
            la = la_base * away_atk * home_def

            probs = probs_from_lambdas(lh, la)
            if probs is None:
                continue

            ph, pd_, pa = probs
            w = wynik_1x2(gh, ga)
            ll = log_loss_val(ph, pd_, pa, w)

            rows.append({
                "sezon": target_season,
                "kolejka": int(kolejka),
                "gospodarz": home,
                "gosc": away,
                "wynik_1x2": w,
                "p_home": ph,
                "p_draw": pd_,
                "p_away": pa,
                "log_loss": ll,
                "lambda_home_base": lh_base,
                "lambda_away_base": la_base,
                "xg_gosp_actual": xgh,
                "xg_gosc_actual": xga,
            })

            pending.append({
                "home": home, "away": away,
                "xgh": xgh, "xga": xga,
                "lh_base": lh_base, "la_base": la_base,
            })

        for p in pending:
            history[p["home"]]["atk"].append({"actual": p["xgh"], "expected": p["lh_base"]})
            history[p["home"]]["def_"].append({"actual": p["xga"], "expected": p["la_base"]})
            history[p["away"]]["atk"].append({"actual": p["xga"], "expected": p["la_base"]})
            history[p["away"]]["def_"].append({"actual": p["xgh"], "expected": p["lh_base"]})

        print(f"  {target_season} K{kolejka:02d} OK")

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CALIB_JSON.exists():
        raise FileNotFoundError(f"Brak pliku: {CALIB_JSON}")

    with open(CALIB_JSON) as f:
        saved = json.load(f)

    best_cfg = saved["best_form_cfg"]
    print(f"Best cfg z JSON: {best_cfg}")

    print("\n1. Wczytuję matches...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    print(f"\n2. Rolling backtesting kalibracja ({VAL_SEASON})...")
    df_val = run_season(df, VAL_SEASON, best_cfg)

    print(f"\n3. Rolling backtesting test ({TEST_SEASON})...")
    df_test = run_season(df, TEST_SEASON, best_cfg)

    print(f"\n4. Fit kalibratora na {VAL_SEASON}...")
    calib = fit_calibrator(df_val)

    def apply_cal(df_in, cal):
        ll_list = []
        for row in df_in.itertuples(index=False):
            p = calibrate(row.p_home, row.p_draw, row.p_away,
                          cal["T"], cal["bH"], cal["bD"], cal["bA"])
            ll_list.append(log_loss_val(p[0], p[1], p[2], row.wynik_1x2))
        return ll_list

    ll_val_raw = df_val["log_loss"].mean()
    ll_val_cal = np.mean(apply_cal(df_val, calib))
    ll_test_raw = df_test["log_loss"].mean()
    ll_test_cal = np.mean(apply_cal(df_test, calib))

    BEST_BASELINE = 1.0571

    ll_val_early = df_val[df_val["kolejka"] <= 5]["log_loss"].mean()
    ll_test_early = df_test[df_test["kolejka"] <= 5]["log_loss"].mean()
    ll_val_late = df_val[df_val["kolejka"] > 5]["log_loss"].mean()
    ll_test_late = df_test[df_test["kolejka"] > 5]["log_loss"].mean()

    lines = []
    lines.append("=" * 78)
    lines.append("FORMA xG — QUICK TEST")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. KONFIGURACJA")
    lines.append("-" * 78)
    lines.append(f"  window:   {best_cfg['window']}")
    lines.append(f"  eta_form: {best_cfg['eta_form']}")
    lines.append(f"  k_form:   {best_cfg['k_form']}")
    lines.append(f"  cap_form: {best_cfg['cap_form']}")
    lines.append(f"  Kalibrator: T={calib['T']:.4f}, bH={calib['bH']:.4f}, bD={calib['bD']:.4f}, bA={calib['bA']:.4f}")
    lines.append("")
    lines.append("2. WYNIKI")
    lines.append("-" * 78)
    lines.append(f"  VAL  {VAL_SEASON}  RAW:  {ll_val_raw:.4f}")
    lines.append(f"  VAL  {VAL_SEASON}  CAL:  {ll_val_cal:.4f}")
    lines.append(f"  TEST {TEST_SEASON} RAW:  {ll_test_raw:.4f}")
    lines.append(f"  TEST {TEST_SEASON} CAL:  {ll_test_cal:.4f}")
    lines.append(f"  Baseline best:         {BEST_BASELINE}")
    lines.append(f"  Delta (base - forma):  {BEST_BASELINE - ll_test_cal:+.4f}")
    lines.append(f"  Benchmark losowy:      {np.log(3):.4f}")
    lines.append("")
    lines.append("3. WCZESNY vs PÓŹNY SEZON")
    lines.append("-" * 78)
    lines.append(f"  VAL  kolejki 1-5:  {ll_val_early:.4f}")
    lines.append(f"  TEST kolejki 1-5:  {ll_test_early:.4f}")
    lines.append(f"  VAL  kolejki 6-34: {ll_val_late:.4f}")
    lines.append(f"  TEST kolejki 6-34: {ll_test_late:.4f}")
    lines.append("")
    lines.append("4. WNIOSEK")
    lines.append("-" * 78)

    delta = BEST_BASELINE - ll_test_cal
    if delta > 0.005:
        lines.append(f"  WYRAŹNA POPRAWA OOS: {delta:+.4f}")
        lines.append("  Forma xG poprawia model. Warto wdrożyć.")
    elif delta > 0.001:
        lines.append(f"  MAŁA POPRAWA OOS: {delta:+.4f}")
        lines.append("  Sygnał jest, ale niewielki.")
    elif delta > 0:
        lines.append(f"  MARGINALNA POPRAWA: {delta:+.4f}")
    else:
        lines.append(f"  BRAK POPRAWY OOS: {delta:+.4f}")
        lines.append("  Forma xG nie poprawia modelu poza próbką.")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {REPORT_PATH}")


if __name__ == "__main__":
    main()