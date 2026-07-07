"""
model_1x2_logistic_backtest.py
==============================
Pełny backtesting: Poisson(xG) → lambdy → Logistic Regression → 1X2

Każda kolejka:
1. Trenuje Poisson(xG) na historii
2. Wyciąga lambdy dla meczów kolejki
3. Trenuje LogisticRegression na historycznych lambdach + wynikach
4. Przewiduje 1X2 na nowych lambdach

To unika data leakage.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

DB_PATH = Path("db/ekstraklasa.db")
SEZON_TEST = "2025/26"
MAX_GOLE = 10

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

BENIAMINKOWIE = {
    "2024/25": ["GKS Katowice", "Lechia Gdańsk", "Motor Lublin"],
    "2025/26": ["Arka Gdynia", "Bruk-Bet Termalica Nieciecza", "Wisła Płock"],
}
PRIORY = {
    "2024/25": {"atak": 0.80, "obrona": 1.10},
    "2025/26": {"atak": 0.7744, "obrona": 1.0648},
}
K_PROM = 10

# =============================================================================
# POISSON xG MODEL (skrócony — tylko kluczowe funkcje)
# =============================================================================

def prepare_data_xg(df):
    df = df.copy()
    df["xgh"] = df["xg_gosp"].fillna(df["gole_gosp"])
    df["xga"] = df["xg_gosc"].fillna(df["gole_gosc"])

    teams = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    t2i = {t: i for i, t in enumerate(teams)}
    i2t = {i: t for t, i in t2i.items()}
    n = len(teams)

    return {
        "n": n, "t2i": t2i, "i2t": i2t,
        "teams": teams,
        "home_idx": df["gospodarz"].map(t2i).values,
        "away_idx": df["gosc"].map(t2i).values,
        "xgh": df["xgh"].values.astype(float),
        "xga": df["xga"].values.astype(float),
        "w": df["waga_sezonu"].values.astype(float),
    }


def nll_poisson(theta, d):
    N = d["n"]
    la = np.zeros(N)
    lb = np.zeros(N)
    la[:N-1] = theta[2:2+N-1]
    la[N-1] = -np.sum(la[:N-1])
    lb[:N-1] = theta[2+N-1:2+2*(N-1)]
    lb[N-1] = -np.sum(lb[:N-1])

    mu_h = np.exp(theta[0])
    mu_a = np.exp(theta[1])
    a = np.exp(la)
    b = np.exp(lb)

    lh = np.maximum(mu_h * a[d["home_idx"]] * b[d["away_idx"]], 1e-10)
    ll = np.maximum(mu_a * a[d["away_idx"]] * b[d["home_idx"]], 1e-10)

    return -np.sum(d["w"] * (
        d["xgh"] * np.log(lh) - lh + d["xga"] * np.log(ll) - ll
    ))


def fit_poisson_xg(df_train):
    d = prepare_data_xg(df_train)
    N = d["n"]
    th0 = np.zeros(2 + 2*(N-1))
    th0[0] = np.log(max(np.mean(d["xgh"]), 0.01))
    th0[1] = np.log(max(np.mean(d["xga"]), 0.01))

    res = minimize(nll_poisson, th0, args=(d,), method="L-BFGS-B",
                   options={"maxiter": 10000, "ftol": 1e-12})

    la = np.zeros(N)
    lb = np.zeros(N)
    la[:N-1] = res.x[2:2+N-1]
    la[N-1] = -np.sum(la[:N-1])
    lb[:N-1] = res.x[2+N-1:2+2*(N-1)]
    lb[N-1] = -np.sum(lb[:N-1])

    return {
        "mu_h": np.exp(res.x[0]),
        "mu_a": np.exp(res.x[1]),
        "alpha": {d["i2t"][i]: np.exp(la[i]) for i in range(N)},
        "beta": {d["i2t"][i]: np.exp(lb[i]) for i in range(N)},
    }


def apply_promoted(params, season, df_prev):
    promoted = BENIAMINKOWIE.get(season, [])
    prior = PRIORY.get(season, None)
    if not promoted or not prior:
        return params

    for team in promoted:
        n = int(((df_prev["gospodarz"] == team) | (df_prev["gosc"] == team)).sum())
        a_mle = params["alpha"].get(team, prior["atak"])
        b_mle = params["beta"].get(team, prior["obrona"])
        params["alpha"][team] = (K_PROM * prior["atak"] + n * a_mle) / (K_PROM + n)
        params["beta"][team] = (K_PROM * prior["obrona"] + n * b_mle) / (K_PROM + n)
    return params


def get_lambdas(params, home, away):
    if home not in params["alpha"] or away not in params["alpha"]:
        return None, None
    lh = params["mu_h"] * params["alpha"][home] * params["beta"][away]
    la = params["mu_a"] * params["alpha"][away] * params["beta"][home]
    return float(lh), float(la)


def lambdas_to_probs(lh, la):
    m = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            m[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    s = m.sum()
    if s <= 0:
        return None, None, None
    m /= s
    return (
        float(np.sum(np.tril(m, -1))),
        float(np.sum(np.diag(m))),
        float(np.sum(np.triu(m, 1))),
    )


# =============================================================================
# BACKTESTING: POISSON(xG) → LOGISTIC REGRESSION
# =============================================================================

def run_backtest(df_all):
    df_test = df_all[df_all["sezon"] == SEZON_TEST].copy()
    df_hist = df_all[df_all["sezon"] != SEZON_TEST].copy()
    kolejki = sorted(df_test["kolejka"].unique())

    # historia do treningu logistic regression
    hist_lambdas = []  # [{lh, la, ldiff, ltotal, wynik}, ...]

    results = []

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)

        # 1. Fit Poisson xG
        params = fit_poisson_xg(df_train)
        params = apply_promoted(params, SEZON_TEST, df_prev)

        # 2. Lambdy dla meczów tej kolejki
        df_round = df_test[df_test["kolejka"] == kolejka]

        round_features = []
        round_meta = []

        for _, mx in df_round.iterrows():
            home, away = mx["gospodarz"], mx["gosc"]
            gh, ga = int(mx["gole_gosp"]), int(mx["gole_gosc"])

            lh, la = get_lambdas(params, home, away)
            if lh is None:
                continue

            ldiff = lh - la
            ltotal = lh + la

            round_features.append([lh, la, ldiff, ltotal])
            round_meta.append({
                "kolejka": int(kolejka),
                "gospodarz": home,
                "gosc": away,
                "gole_gosp": gh,
                "gole_gosc": ga,
                "wynik_1x2": wynik(gh, ga),
                "lambda_home": lh,
                "lambda_away": la,
            })

        if len(round_features) == 0:
            continue

        X_round = np.array(round_features)

        # 3. Trenuj logistic regression na historii (jeśli mamy dość danych)
        if len(hist_lambdas) >= 30:
            hist_df = pd.DataFrame(hist_lambdas)
            X_hist = hist_df[["lh", "la", "ldiff", "ltotal"]].values
            y_hist = hist_df["wynik"].values

            scaler = StandardScaler()
            X_hist_s = scaler.fit_transform(X_hist)
            X_round_s = scaler.transform(X_round)

            lr = LogisticRegression(
    solver="lbfgs",
    C=1.0, max_iter=1000, random_state=42
)
            lr.fit(X_hist_s, y_hist)

            probs = lr.predict_proba(X_round_s)
            classes = lr.classes_

            for i, meta in enumerate(round_meta):
                p_dict = {classes[j]: probs[i, j] for j in range(len(classes))}
                p_h = p_dict.get("H", 0.0)
                p_d = p_dict.get("D", 0.0)
                p_a = p_dict.get("A", 0.0)

                ll = -np.log(max(
                    {"H": p_h, "D": p_d, "A": p_a}[meta["wynik_1x2"]], 1e-12
                ))

                meta["p_home_log"] = round(p_h, 4)
                meta["p_draw_log"] = round(p_d, 4)
                meta["p_away_log"] = round(p_a, 4)
                meta["log_loss_log"] = round(ll, 4)

                # porównanie z czystym Poissonem
                p_ph, p_pd, p_pa = lambdas_to_probs(
                    meta["lambda_home"], meta["lambda_away"]
                )
                ll_base = -np.log(max(
                    {"H": p_ph, "D": p_pd, "A": p_pa}[meta["wynik_1x2"]], 1e-12
                ))
                meta["log_loss_base"] = round(ll_base, 4)

                results.append(meta)
        else:
            # za mało danych na logistic — używamy czystego Poissona
            for i, meta in enumerate(round_meta):
                p_h, p_d, p_a = lambdas_to_probs(
                    meta["lambda_home"], meta["lambda_away"]
                )
                meta["p_home_log"] = round(p_h, 4)
                meta["p_draw_log"] = round(p_d, 4)
                meta["p_away_log"] = round(p_a, 4)
                meta["log_loss_log"] = round(-np.log(max(
                    {"H": p_h, "D": p_d, "A": p_a}[meta["wynik_1x2"]], 1e-12
                )), 4)
                meta["log_loss_base"] = meta["log_loss_log"]
                results.append(meta)

        # 4. Dodaj lambdy z tej kolejki do historii (do treningu logistic w następnej kolejce)
        for meta in round_meta:
            lh, la = meta["lambda_home"], meta["lambda_away"]
            hist_lambdas.append({
                "lh": lh,
                "la": la,
                "ldiff": lh - la,
                "ltotal": lh + la,
                "wynik": meta["wynik_1x2"],
            })

        print(f"K{kolejka:02d} OK ({len(round_meta)} meczów)")

    return pd.DataFrame(results)


def wynik(gh, ga):
    if gh > ga:
        return "H"
    elif gh == ga:
        return "D"
    return "A"


# =============================================================================
# MAIN
# =============================================================================

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    print("=" * 60)
    print("BACKTEST: Poisson(xG) → Logistic Regression → 1X2")
    print("=" * 60)

    df_res = run_backtest(df)

    # Save
    out = Path("data/processed/backtesting_1x2_logistic.csv")
    df_res.to_csv(out, index=False, encoding="utf-8-sig")

    # Report
    print("\n" + "=" * 60)
    print("WYNIKI")
    print("=" * 60)

    ll_log = df_res["log_loss_log"].mean()
    ll_base = df_res["log_loss_base"].mean()

    print(f"\nLog-loss BASE (Poisson xG)  : {ll_base:.4f}")
    print(f"Log-loss LOGISTIC           : {ll_log:.4f}")
    print(f"Poprawa                     : {ll_base - ll_log:+.4f}")

    # argmax
    pred_log = df_res[["p_home_log", "p_draw_log", "p_away_log"]].idxmax(axis=1)
    pred_log = pred_log.map({"p_home_log": "H", "p_draw_log": "D", "p_away_log": "A"})

    pred_base = df_res[["p_home_log", "p_draw_log", "p_away_log"]].idxmax(axis=1)
    # base to ten sam Poisson — porównajmy argmax
    # Musimy odtworzyć argmax z Poissona
    print("\n--- ARGLAX LOGISTIC ---")
    for r in ["H", "D", "A"]:
        print(f"  {r}: {(pred_log == r).sum()}")

    print(f"\nAccuracy LOGISTIC: {(pred_log == df_res['wynik_1x2']).mean():.1%}")

    # Średnie prawdopodobieństwa
    print(f"\n--- ŚREDNIE PRAWDOPODOBIENSTWA LOGISTIC ---")
    print(f"H = {df_res['p_home_log'].mean():.3f}")
    print(f"D = {df_res['p_draw_log'].mean():.3f}")
    print(f"A = {df_res['p_away_log'].mean():.3f}")

    print("\n" + "=" * 60)
    print("GOTOWE")
    print("=" * 60)


if __name__ == "__main__":
    main()