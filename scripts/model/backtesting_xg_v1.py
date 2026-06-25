"""
backtesting_xg_v1.py
====================
Backtesting modelu Poissona trenowanego na xG zamiast goli.

Cel:
- sprawdzić czy xG jako sygnał siły poprawia 1X2
- porównać log-loss vs model gole (v2)

Output:
data/processed/backtesting_wyniki_xg_v1.csv
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURACJA
# =============================================================================

DB_PATH = Path("db/ekstraklasa.db")
OUTPUT_PATH = Path("data/processed/backtesting_wyniki_xg_v1.csv")
SEZON_TEST = "2025/26"
MAX_GOLE = 10

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

# =============================================================================
# MODEL MLE (identyczny jak w modelu goli)
# =============================================================================

def przygotuj_dane_xg(df_trening):
    # Fallback: jeśli brak xG, użyj goli (żeby nie było nan)
    df_trening['xg_gosp_final'] = df_trening['xg_gosp'].fillna(df_trening['gole_gosp'])
    df_trening['xg_gosc_final'] = df_trening['xg_gosc'].fillna(df_trening['gole_gosc'])
    
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


# =============================================================================
# PREDYKCJA
# =============================================================================

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

    macierz /= macierz.sum()

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
# BACKTESTING
# =============================================================================

def wynik_1x2(gh, ga):
    if gh > ga:
        return "H"
    elif gh == ga:
        return "D"
    else:
        return "A"


def log_loss_meczu(p_home, p_draw, p_away, wynik):
    eps = 1e-10
    p = {"H": p_home, "D": p_draw, "A": p_away}[wynik]
    return -np.log(max(p, eps))


def run_backtesting(df):

    df_test = df[df["sezon"] == SEZON_TEST].copy()
    df_hist = df[df["sezon"] != SEZON_TEST].copy()

    kolejki = sorted(df_test["kolejka"].unique())

    wyniki = []

    for kolejka in kolejki:

        df_prev = df_test[df_test["kolejka"] < kolejka]

        df_trening = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane_xg(df_trening)

        theta = trenuj_model(data)

        params = ekstrahuj_parametry(theta, data)

        df_kolejka = df_test[df_test["kolejka"] == kolejka]

        for _, mecz in df_kolejka.iterrows():

            gosp = mecz["gospodarz"]
            gosc = mecz["gosc"]

            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])

            pred = przewiduj_mecz(params, gosp, gosc)

            if pred is None:
                continue

            wynik_rzecz = wynik_1x2(gh, ga)

            ll = log_loss_meczu(
                pred["p_home"],
                pred["p_draw"],
                pred["p_away"],
                wynik_rzecz
            )

            wyniki.append({
                "kolejka": kolejka,
                "gospodarz": gosp,
                "gosc": gosc,
                "wynik_1x2": wynik_rzecz,
                "p_home": pred["p_home"],
                "p_draw": pred["p_draw"],
                "p_away": pred["p_away"],
                "log_loss": ll,
            })

        print(f"K{kolejka:02d} OK")

    return pd.DataFrame(wyniki)


# =============================================================================
# MAIN
# =============================================================================

def main():

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    df_wyniki = run_backtesting(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_wyniki.to_csv(OUTPUT_PATH, index=False)

    print("\n==============================")
    print("WYNIKI xG MODEL")
    print("==============================")
    print(f"Log-loss xG model: {df_wyniki['log_loss'].mean():.4f}")


if __name__ == "__main__":
    main()