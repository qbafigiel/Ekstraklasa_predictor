"""
eksperyment_forma_xg_1x2.py
===========================

Cel:
- poprawić model 1X2 przez dodanie krótkoterminowej formy
- baza: Poisson trenowany na xG
- forma: korekta lambd na podstawie ostatnich N meczów
- tuning hyperparametrów na sezonie 2024/25
- test finalny na sezonie 2025/26
- kalibracja softmax uczona na 2024/25, testowana na 2025/26

Architektura:
  lambda_home_final = lambda_home_base * form_attack(home) * form_defense(away)
  lambda_away_final = lambda_away_base * form_attack(away) * form_defense(home)

Forma liczona jako:
  ratio = recent_actual_xg / recent_expected_xg
  shrink do 1.0 przez K_FORM
  dodatkowo siła wpływu = ETA_FORM
  clipping ekstremów = CAP_FORM

Output:
- data/processed/forma_grid_2024_25.csv
- data/processed/backtesting_xg_form_best_2024_25.csv
- data/processed/backtesting_xg_form_best_2025_26.csv
- data/processed/kalibrator_xg_form_best.json
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

# =============================================================================
# KONFIGURACJA
# =============================================================================

DB_PATH = Path("db/ekstraklasa.db")

OUTPUT_GRID_PATH = Path("data/processed/forma_grid_2024_25.csv")
OUTPUT_VAL_PATH = Path("data/processed/backtesting_xg_form_best_2024_25.csv")
OUTPUT_TEST_PATH = Path("data/processed/backtesting_xg_form_best_2025_26.csv")
OUTPUT_CALIB_PATH = Path("data/processed/kalibrator_xg_form_best.json")

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"
MAX_GOLE = 10

# Wagi sezonów
WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

# Prior beniaminków
K_PROMOTED_PRIOR = 10

PROMOTED_BY_SEASON = {
    "2024/25": [
        "GKS Katowice",
        "Lechia Gdańsk",
        "Motor Lublin",
    ],
    "2025/26": [
        "Arka Gdynia",
        "Bruk-Bet Termalica Nieciecza",
        "Wisła Płock",
    ],
}

PRIORS_BY_SEASON = {
    "2024/25": {
        "prior_atak": 0.80,
        "prior_obrona": 1.10,
    },
    "2025/26": {
        "prior_atak": 0.7744,
        "prior_obrona": 1.0648,
    },
}

# Grid hyperparametrów formy
WINDOWS = [3, 5, 8, 10]
ETAS = [0.25, 0.50, 0.75, 1.00]
K_FORMS = [3, 5, 8]
CAPS = [1.25, 1.35, 1.50]

# =============================================================================
# HELPERS
# =============================================================================

def season_key(season_str):
    return int(str(season_str).split("/")[0])

def wynik_1x2(gh, ga):
    if gh > ga:
        return "H"
    elif gh == ga:
        return "D"
    return "A"

def log_loss_1x2(p_home, p_draw, p_away, wynik):
    eps = 1e-12
    p = {"H": p_home, "D": p_draw, "A": p_away}[wynik]
    return -np.log(max(p, eps))

def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / e.sum()

# =============================================================================
# MODEL BAZOWY xG-Poisson
# =============================================================================

def przygotuj_dane_xg(df_trening):
    df_trening = df_trening.copy()

    # fallback: jeśli brak xG, użyj goli
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

def trenuj_model_xg(data):
    N = data["n_druzyn"]

    theta0 = np.zeros(2 + 2 * (N - 1))
    theta0[0] = np.log(np.mean(data["goals_home"]))
    theta0[1] = np.log(np.mean(data["goals_away"]))

    result = minimize(
        neg_log_likelihood,
        theta0,
        args=(data,),
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
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
            data["idx_to_team"][i]: np.exp(log_alpha[i]) for i in range(N)
        },
        "beta": {
            data["idx_to_team"][i]: np.exp(log_beta[i]) for i in range(N)
        },
    }

def policz_mecze_beniaminkow(df_prev_current_season, promoted):
    wynik = {}
    for team in promoted:
        n = int(
            ((df_prev_current_season["gospodarz"] == team) |
             (df_prev_current_season["gosc"] == team)).sum()
        )
        wynik[team] = n
    return wynik

def zastosuj_prior_beniaminkow(params, season_test, df_prev_current_season):
    promoted = PROMOTED_BY_SEASON.get(season_test, [])
    prior = PRIORS_BY_SEASON.get(season_test, None)

    if not promoted or prior is None:
        return params

    prior_atak = prior["prior_atak"]
    prior_obrona = prior["prior_obrona"]

    counts = policz_mecze_beniaminkow(df_prev_current_season, promoted)

    for team in promoted:
        n = counts.get(team, 0)

        if team not in params["alpha"]:
            alpha_mle = None
            beta_mle = None
            alpha_final = prior_atak
            beta_final = prior_obrona
        else:
            alpha_mle = params["alpha"][team]
            beta_mle = params["beta"][team]

            alpha_final = (K_PROMOTED_PRIOR * prior_atak + n * alpha_mle) / (K_PROMOTED_PRIOR + n)
            beta_final = (K_PROMOTED_PRIOR * prior_obrona + n * beta_mle) / (K_PROMOTED_PRIOR + n)

        params["alpha"][team] = alpha_final
        params["beta"][team] = beta_final

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

    p_home = float(np.sum(np.tril(macierz, -1)))
    p_draw = float(np.sum(np.diag(macierz)))
    p_away = float(np.sum(np.triu(macierz, 1)))

    return p_home, p_draw, p_away

# =============================================================================
# BAZOWY BACKTEST xG DLA SEZONU
# =============================================================================

def get_previous_seasons(df, season_test):
    seasons = sorted(df["sezon"].unique(), key=season_key)
    idx = seasons.index(season_test)
    return seasons[:idx]

def build_base_context_for_season(df_all, season_test):
    df_all = df_all.copy()

    prev_seasons = get_previous_seasons(df_all, season_test)
    df_hist = df_all[df_all["sezon"].isin(prev_seasons)].copy()
    df_test = df_all[df_all["sezon"] == season_test].copy()

    kolejki = sorted(df_test["kolejka"].unique())

    rows = []

    print(f"\n[BASE] Buduję kontekst dla sezonu {season_test}...")

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka].copy()
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane_xg(df_train)
        theta = trenuj_model_xg(data)
        params = ekstrahuj_parametry(theta, data)

        params = zastosuj_prior_beniaminkow(
            params=params,
            season_test=season_test,
            df_prev_current_season=df_prev
        )

        df_round = df_test[df_test["kolejka"] == kolejka].copy()

        for _, match in df_round.iterrows():
            home = match["gospodarz"]
            away = match["gosc"]

            gh = int(match["gole_gosp"])
            ga = int(match["gole_gosc"])

            xgh = float(match["xg_gosp"]) if pd.notna(match["xg_gosp"]) else float(gh)
            xga = float(match["xg_gosc"]) if pd.notna(match["xg_gosc"]) else float(ga)

            if home not in params["alpha"] or away not in params["alpha"]:
                continue

            lh_base, la_base = lambda_base(params, home, away)
            probs = probs_from_lambdas(lh_base, la_base)
            if probs is None:
                continue

            p_home, p_draw, p_away = probs
            wynik = wynik_1x2(gh, ga)
            ll = log_loss_1x2(p_home, p_draw, p_away, wynik)

            rows.append({
                "sezon": season_test,
                "kolejka": int(kolejka),
                "gospodarz": home,
                "gosc": away,
                "gole_gosp": gh,
                "gole_gosc": ga,
                "wynik_1x2": wynik,
                "xg_gosp_actual": xgh,
                "xg_gosc_actual": xga,
                "lambda_home_base": lh_base,
                "lambda_away_base": la_base,
                "p_home_base": p_home,
                "p_draw_base": p_draw,
                "p_away_base": p_away,
                "log_loss_base": ll,
            })

        print(f"[BASE] {season_test} K{kolejka:02d} OK")

    return pd.DataFrame(rows)

# =============================================================================
# WARSTWA FORMY
# =============================================================================

def form_multiplier(history, window, eta, k_form, cap):
    """
    history: lista słowników {"actual": ..., "expected": ...}
    window : ile ostatnich meczów brać
    eta    : siła wpływu formy
    k_form : shrinkage do 1.0
    cap    : maks. odchylenie ratio, np 1.35 => clip do [1/1.35, 1.35]
    """
    if eta <= 0 or len(history) == 0:
        return 1.0

    recent = history[-window:] if window is not None else history
    n = len(recent)

    actual = sum(x["actual"] for x in recent)
    expected = sum(x["expected"] for x in recent)

    if expected <= 1e-12:
        return 1.0

    ratio = actual / expected

    low = 1.0 / cap
    high = cap
    ratio = float(np.clip(ratio, low, high))

    # shrink do 1.0 w log-space
    # waga danych rośnie wraz z liczbą meczów
    weight = n / (k_form + n)

    # final = exp(eta * weight * log(ratio))
    # eta=0   -> 1.0
    # weight=0 -> 1.0
    # ratio=1 -> 1.0
    return float(np.exp(eta * weight * np.log(ratio)))

def apply_form_layer(df_base, window, eta, k_form, cap):
    df_base = df_base.sort_values(["kolejka", "gospodarz", "gosc"]).reset_index(drop=True)

    teams = sorted(set(df_base["gospodarz"]) | set(df_base["gosc"]))
    history = {
        team: {
            "atk": [],
            "def": [],
        }
        for team in teams
    }

    rows = []

    for kolejka in sorted(df_base["kolejka"].unique()):
        df_round = df_base[df_base["kolejka"] == kolejka].copy()

        pending_updates = []

        for _, row in df_round.iterrows():
            home = row["gospodarz"]
            away = row["gosc"]

            # form multipliers
            home_atk_mult = form_multiplier(history[home]["atk"], window, eta, k_form, cap)
            away_atk_mult = form_multiplier(history[away]["atk"], window, eta, k_form, cap)

            # "def" > 1 = gorsza ostatnia obrona; <1 = lepsza
            home_def_mult = form_multiplier(history[home]["def"], window, eta, k_form, cap)
            away_def_mult = form_multiplier(history[away]["def"], window, eta, k_form, cap)

            lh = row["lambda_home_base"] * home_atk_mult * away_def_mult
            la = row["lambda_away_base"] * away_atk_mult * home_def_mult

            probs = probs_from_lambdas(lh, la)
            if probs is None:
                continue

            p_home, p_draw, p_away = probs
            ll = log_loss_1x2(p_home, p_draw, p_away, row["wynik_1x2"])

            pred = ["H", "D", "A"][int(np.argmax([p_home, p_draw, p_away]))]

            rows.append({
                **row.to_dict(),
                "window": window,
                "eta_form": eta,
                "k_form": k_form,
                "cap_form": cap,
                "home_atk_mult": home_atk_mult,
                "away_atk_mult": away_atk_mult,
                "home_def_mult": home_def_mult,
                "away_def_mult": away_def_mult,
                "lambda_home": lh,
                "lambda_away": la,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "log_loss": ll,
                "wynik_pred": pred,
                "czy_trafil": int(pred == row["wynik_1x2"]),
            })

            # update historii po kolejce:
            # atak: actual xG for vs expected xG for
            # obrona: actual xG against vs expected xG against
            pending_updates.append({
                "team": home,
                "atk_actual": row["xg_gosp_actual"],
                "atk_expected": row["lambda_home_base"],
                "def_actual": row["xg_gosc_actual"],
                "def_expected": row["lambda_away_base"],
            })
            pending_updates.append({
                "team": away,
                "atk_actual": row["xg_gosc_actual"],
                "atk_expected": row["lambda_away_base"],
                "def_actual": row["xg_gosp_actual"],
                "def_expected": row["lambda_home_base"],
            })

        # aktualizacja dopiero po całej kolejce
        for upd in pending_updates:
            history[upd["team"]]["atk"].append({
                "actual": float(upd["atk_actual"]),
                "expected": float(upd["atk_expected"]),
            })
            history[upd["team"]]["def"].append({
                "actual": float(upd["def_actual"]),
                "expected": float(upd["def_expected"]),
            })

    return pd.DataFrame(rows)

# =============================================================================
# KALIBRACJA
# =============================================================================

def calibration_loss(params, p_raw, y_true):
    T, bH, bD, bA = params

    # T musi być dodatnie
    if T <= 0:
        return 1e9

    total = 0.0

    for i in range(len(p_raw)):
        logits = np.log(np.maximum(p_raw[i], 1e-12))
        logits_cal = (logits + np.array([bH, bD, bA])) / T
        probs = softmax(logits_cal)
        total += -np.log(max(probs[y_true[i]], 1e-12))

    return total / len(p_raw)

def fit_calibrator(df_val):
    p_raw = df_val[["p_home", "p_draw", "p_away"]].values.astype(float)
    y_true = df_val["wynik_1x2"].map({"H": 0, "D": 1, "A": 2}).values

    x0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    result = minimize(
        calibration_loss,
        x0,
        args=(p_raw, y_true),
        method="L-BFGS-B",
        bounds=[(0.25, 6.0), (None, None), (None, None), (None, None)],
        options={"maxiter": 10000}
    )

    T, bH, bD, bA = result.x

    return {
        "T": float(T),
        "bH": float(bH),
        "bD": float(bD),
        "bA": float(bA),
    }

def apply_calibrator(df, calib):
    T = calib["T"]
    bH = calib["bH"]
    bD = calib["bD"]
    bA = calib["bA"]

    rows = []

    for _, row in df.iterrows():
        logits = np.log(np.maximum(
            np.array([row["p_home"], row["p_draw"], row["p_away"]], dtype=float),
            1e-12
        ))
        logits_cal = (logits + np.array([bH, bD, bA])) / T
        p_cal = softmax(logits_cal)

        wynik = row["wynik_1x2"]
        ll_cal = log_loss_1x2(p_cal[0], p_cal[1], p_cal[2], wynik)
        pred_cal = ["H", "D", "A"][int(np.argmax(p_cal))]

        new_row = row.to_dict()
        new_row["p_home_cal"] = float(p_cal[0])
        new_row["p_draw_cal"] = float(p_cal[1])
        new_row["p_away_cal"] = float(p_cal[2])
        new_row["log_loss_cal"] = float(ll_cal)
        new_row["wynik_pred_cal"] = pred_cal
        new_row["czy_trafil_cal"] = int(pred_cal == wynik)

        rows.append(new_row)

    return pd.DataFrame(rows)

# =============================================================================
# RAPORTY
# =============================================================================

def summarize_df(df, label, use_cal=False):
    if use_cal:
        ll_col = "log_loss_cal"
        pred_col = "wynik_pred_cal"
        pH = "p_home_cal"
        pD = "p_draw_cal"
        pA = "p_away_cal"
    else:
        ll_col = "log_loss"
        pred_col = "wynik_pred"
        pH = "p_home"
        pD = "p_draw"
        pA = "p_away"

    print(f"\n--- {label} ---")
    print(f"log-loss : {df[ll_col].mean():.4f}")
    print(f"accuracy : {(df[pred_col] == df['wynik_1x2']).mean():.4%}")
    print(
        f"avg probs: H={df[pH].mean():.3f} | "
        f"D={df[pD].mean():.3f} | "
        f"A={df[pA].mean():.3f}"
    )
    print(
        f"argmax   : H={(df[pred_col] == 'H').sum()} | "
        f"D={(df[pred_col] == 'D').sum()} | "
        f"A={(df[pred_col] == 'A').sum()}"
    )

# =============================================================================
# MAIN
# =============================================================================

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    # sanity
    required = ["sezon", "kolejka", "gospodarz", "gosc", "gole_gosp", "gole_gosc"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny: {col}")

    print("============================================================")
    print("EKSPERYMENT: xG + FORMA + KALIBRACJA 1X2")
    print("============================================================")
    print(f"Validation season: {VAL_SEASON}")
    print(f"Test season      : {TEST_SEASON}")
    print("Tuning formy na 2024/25, test finalny na 2025/26")
    print("============================================================")

    # 1) Bazowe konteksty xG
    df_base_val = build_base_context_for_season(df, VAL_SEASON)
    df_base_test = build_base_context_for_season(df, TEST_SEASON)

    print(f"\nBASE xG {VAL_SEASON} log-loss : {df_base_val['log_loss_base'].mean():.4f}")
    print(f"BASE xG {TEST_SEASON} log-loss: {df_base_test['log_loss_base'].mean():.4f}")

    # 2) Grid search formy na validation season
    print("\n[GRID] Start grid search formy...")
    grid_rows = []
    best_ll = 999.0
    best_cfg = None
    best_val_df = None

    total_cfg = len(WINDOWS) * len(ETAS) * len(K_FORMS) * len(CAPS)
    cfg_idx = 0

    for window in WINDOWS:
        for eta in ETAS:
            for k_form in K_FORMS:
                for cap in CAPS:
                    cfg_idx += 1
                    df_val_variant = apply_form_layer(
                        df_base=df_base_val,
                        window=window,
                        eta=eta,
                        k_form=k_form,
                        cap=cap,
                    )

                    ll = df_val_variant["log_loss"].mean()
                    acc = df_val_variant["czy_trafil"].mean()

                    grid_rows.append({
                        "window": window,
                        "eta_form": eta,
                        "k_form": k_form,
                        "cap_form": cap,
                        "log_loss_val": ll,
                        "accuracy_val": acc,
                    })

                    print(
                        f"[GRID {cfg_idx:03d}/{total_cfg}] "
                        f"w={window} eta={eta:.2f} k={k_form} cap={cap:.2f} "
                        f"-> ll={ll:.4f}"
                    )

                    if ll < best_ll:
                        best_ll = ll
                        best_cfg = {
                            "window": window,
                            "eta_form": eta,
                            "k_form": k_form,
                            "cap_form": cap,
                        }
                        best_val_df = df_val_variant.copy()

    df_grid = pd.DataFrame(grid_rows).sort_values("log_loss_val").reset_index(drop=True)

    OUTPUT_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(OUTPUT_GRID_PATH, index=False, encoding="utf-8-sig")

    print("\n============================================================")
    print("NAJLEPSZA KONFIGURACJA FORMY (na 2024/25)")
    print("============================================================")
    print(best_cfg)
    print(f"best validation log-loss: {best_ll:.4f}")
    print(f"base validation log-loss: {df_base_val['log_loss_base'].mean():.4f}")

    # 3) Test best cfg na 2025/26
    df_test_best = apply_form_layer(
        df_base=df_base_test,
        window=best_cfg["window"],
        eta=best_cfg["eta_form"],
        k_form=best_cfg["k_form"],
        cap=best_cfg["cap_form"],
    )

    # baseline no-form w wersji spójnej kolumnami
    df_val_base = df_base_val.copy()
    df_val_base["p_home"] = df_val_base["p_home_base"]
    df_val_base["p_draw"] = df_val_base["p_draw_base"]
    df_val_base["p_away"] = df_val_base["p_away_base"]
    df_val_base["log_loss"] = df_val_base["log_loss_base"]
    df_val_base["wynik_pred"] = df_val_base[["p_home", "p_draw", "p_away"]].idxmax(axis=1)
    df_val_base["wynik_pred"] = df_val_base["wynik_pred"].map({
        "p_home": "H", "p_draw": "D", "p_away": "A"
    })

    df_test_base = df_base_test.copy()
    df_test_base["p_home"] = df_test_base["p_home_base"]
    df_test_base["p_draw"] = df_test_base["p_draw_base"]
    df_test_base["p_away"] = df_test_base["p_away_base"]
    df_test_base["log_loss"] = df_test_base["log_loss_base"]
    df_test_base["wynik_pred"] = df_test_base[["p_home", "p_draw", "p_away"]].idxmax(axis=1)
    df_test_base["wynik_pred"] = df_test_base["wynik_pred"].map({
        "p_home": "H", "p_draw": "D", "p_away": "A"
    })

    # 4) Kalibracja: uczymy na validation best, testujemy na test best
    calib = fit_calibrator(best_val_df)
    df_val_best_cal = apply_calibrator(best_val_df, calib)
    df_test_best_cal = apply_calibrator(df_test_best, calib)

    # 5) Zapisy
    df_val_best_cal.to_csv(OUTPUT_VAL_PATH, index=False, encoding="utf-8-sig")
    df_test_best_cal.to_csv(OUTPUT_TEST_PATH, index=False, encoding="utf-8-sig")

    with open(OUTPUT_CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation_season": VAL_SEASON,
            "test_season": TEST_SEASON,
            "best_form_cfg": best_cfg,
            "calibrator": calib,
        }, f, ensure_ascii=False, indent=2)

    # 6) Raport końcowy
    print("\n============================================================")
    print("PODSUMOWANIE")
    print("============================================================")

    summarize_df(df_val_base, f"VAL {VAL_SEASON} | BASE xG", use_cal=False)
    summarize_df(best_val_df, f"VAL {VAL_SEASON} | xG + FORMA", use_cal=False)
    summarize_df(df_val_best_cal, f"VAL {VAL_SEASON} | xG + FORMA + CAL", use_cal=True)

    summarize_df(df_test_base, f"TEST {TEST_SEASON} | BASE xG", use_cal=False)
    summarize_df(df_test_best, f"TEST {TEST_SEASON} | xG + FORMA", use_cal=False)
    summarize_df(df_test_best_cal, f"TEST {TEST_SEASON} | xG + FORMA + CAL", use_cal=True)

    print("\n============================================================")
    print("PLIKI")
    print("============================================================")
    print(OUTPUT_GRID_PATH)
    print(OUTPUT_VAL_PATH)
    print(OUTPUT_TEST_PATH)
    print(OUTPUT_CALIB_PATH)

    print("\nGOTOWE.")


if __name__ == "__main__":
    main()