"""
backtest_time_decay.py
======================
Test różnych wartości parametru decay w MLE.

Zamiast wag schodkowych (0.4 / 0.7 / 1.0)
używamy ciągłego exponential decay:

  waga(mecz) = exp(-decay * dni_wstecz)

lub wersja prostsza:

  waga(mecz) = exp(-decay * pozycja_wstecz)

Grid search decay na sezonie 2024/25 (validation).
Test finalny na 2025/26.
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.optimize import minimize as scipy_minimize
import warnings

warnings.filterwarnings("ignore")

DB_PATH = Path("db/ekstraklasa.db")
SEZON_VAL = "2024/25"
SEZON_TEST = "2025/26"
MAX_GOLE = 10

BENIAMINKOWIE = {
    "2024/25": ["GKS Katowice", "Lechia Gdańsk", "Motor Lublin"],
    "2025/26": ["Arka Gdynia", "Bruk-Bet Termalica Nieciecza", "Wisła Płock"],
}
PRIORY = {
    "2024/25": {"atak": 0.80, "obrona": 1.10},
    "2025/26": {"atak": 0.7744, "obrona": 1.0648},
}
K_PROM = 10

# Grid decay do przeszukania
DECAY_GRID = [0.0, 0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]


# =============================================================================
# WAGI DECAY
# =============================================================================

def compute_decay_weights(df_train, decay):
    """
    Każdy mecz dostaje wagę exp(-decay * pozycja_wstecz).
    pozycja_wstecz=0 to najnowszy mecz, rośnie wstecz.
    decay=0 → wszystkie wagi = 1.0 (brak decay)
    """
    df = df_train.copy().reset_index(drop=True)
    df = df.sort_values(["sezon", "kolejka"], ascending=[True, True])
    df = df.reset_index(drop=True)
    n = len(df)
    # najnowszy mecz = n-1, najstarszy = 0
    pozycja_wstecz = (n - 1) - df.index.values
    wagi = np.exp(-decay * pozycja_wstecz)
    return wagi


# =============================================================================
# POISSON xG
# =============================================================================

def prepare_data(df_train, decay=0.0):
    df = df_train.copy()
    df["xgh"] = df["xg_gosp"].fillna(df["gole_gosp"])
    df["xga"] = df["xg_gosc"].fillna(df["gole_gosc"])

    teams = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    t2i = {t: i for i, t in enumerate(teams)}
    i2t = {i: t for t, i in t2i.items()}
    n = len(teams)

    # decay weights zamiast wag sezonowych
    if decay > 0:
        wagi = compute_decay_weights(df, decay)
    else:
        wagi = np.ones(len(df))

    return {
        "n": n, "t2i": t2i, "i2t": i2t,
        "home_idx": df["gospodarz"].map(t2i).values,
        "away_idx": df["gosc"].map(t2i).values,
        "xgh": df["xgh"].values.astype(float),
        "xga": df["xga"].values.astype(float),
        "w": wagi.astype(float),
    }


def nll(theta, d):
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


def fit_model(df_train, decay=0.0):
    d = prepare_data(df_train, decay)
    N = d["n"]
    th0 = np.zeros(2 + 2*(N-1))
    th0[0] = np.log(max(np.mean(d["xgh"]), 0.01))
    th0[1] = np.log(max(np.mean(d["xga"]), 0.01))

    res = minimize(nll, th0, args=(d,), method="L-BFGS-B",
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
        a = params["alpha"].get(team, prior["atak"])
        b = params["beta"].get(team, prior["obrona"])
        params["alpha"][team] = (K_PROM * prior["atak"] + n * a) / (K_PROM + n)
        params["beta"][team] = (K_PROM * prior["obrona"] + n * b) / (K_PROM + n)
    return params


def get_probs(params, home, away):
    if home not in params["alpha"] or away not in params["alpha"]:
        return None

    lh = params["mu_h"] * params["alpha"][home] * params["beta"][away]
    la = params["mu_a"] * params["alpha"][away] * params["beta"][home]

    m = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            m[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    s = m.sum()
    if s <= 0:
        return None
    m /= s

    return {
        "lh": float(lh), "la": float(la),
        "p_home": float(np.sum(np.tril(m, -1))),
        "p_draw": float(np.sum(np.diag(m))),
        "p_away": float(np.sum(np.triu(m, 1))),
    }


def wynik(gh, ga):
    if gh > ga: return "H"
    if gh == ga: return "D"
    return "A"


# =============================================================================
# KALIBRACJA SOFTMAX
# =============================================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cal_loss(params, p_raw, y):
    T, bH, bD, bA = params
    if T <= 0:
        return 1e9
    total = 0.0
    for i in range(len(p_raw)):
        logits = np.log(np.maximum(p_raw[i], 1e-12))
        p_cal = softmax((logits + np.array([bH, bD, bA])) / T)
        total += -np.log(max(p_cal[y[i]], 1e-12))
    return total / len(p_raw)


def fit_calibrator(df_val):
    p_raw = df_val[["p_home", "p_draw", "p_away"]].values.astype(float)
    y = df_val["wynik_1x2"].map({"H": 0, "D": 1, "A": 2}).values
    res = scipy_minimize(
        cal_loss, [1.0, 0.0, 0.0, 0.0], args=(p_raw, y),
        method="L-BFGS-B",
        bounds=[(0.2, 8.0), (None, None), (None, None), (None, None)]
    )
    T, bH, bD, bA = res.x
    return {"T": T, "bH": bH, "bD": bD, "bA": bA}


def apply_calibrator(df, cal):
    T, bH, bD, bA = cal["T"], cal["bH"], cal["bD"], cal["bA"]
    rows = []
    for _, row in df.iterrows():
        logits = np.log(np.maximum(
            [row["p_home"], row["p_draw"], row["p_away"]], 1e-12
        ))
        p_cal = softmax((logits + np.array([bH, bD, bA])) / T)
        ll = -np.log(max(
            {"H": p_cal[0], "D": p_cal[1], "A": p_cal[2]}[row["wynik_1x2"]],
            1e-12
        ))
        r = row.to_dict()
        r["p_home_cal"] = round(p_cal[0], 4)
        r["p_draw_cal"] = round(p_cal[1], 4)
        r["p_away_cal"] = round(p_cal[2], 4)
        r["ll_cal"] = round(ll, 4)
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# BACKTESTING DLA JEDNEGO SEZONU
# =============================================================================

def run_season_backtest(df_all, season_test, decay):
    seasons = sorted(df_all["sezon"].unique())
    idx = seasons.index(season_test)
    prev_seasons = seasons[:idx]

    df_hist = df_all[df_all["sezon"].isin(prev_seasons)].copy()
    df_test = df_all[df_all["sezon"] == season_test].copy()
    kolejki = sorted(df_test["kolejka"].unique())

    rows = []
    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)

        params = fit_model(df_train, decay)
        params = apply_promoted(params, season_test, df_prev)

        for _, mx in df_test[df_test["kolejka"] == kolejka].iterrows():
            home, away = mx["gospodarz"], mx["gosc"]
            gh, ga = int(mx["gole_gosp"]), int(mx["gole_gosc"])

            pred = get_probs(params, home, away)
            if pred is None:
                continue

            w = wynik(gh, ga)
            ll = -np.log(max(
                {"H": pred["p_home"], "D": pred["p_draw"],
                 "A": pred["p_away"]}[w], 1e-12
            ))

            rows.append({
                "kolejka": kolejka,
                "gospodarz": home,
                "gosc": away,
                "wynik_1x2": w,
                "p_home": round(pred["p_home"], 4),
                "p_draw": round(pred["p_draw"], 4),
                "p_away": round(pred["p_away"], 4),
                "log_loss": round(ll, 4),
                "decay": decay,
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    print("=" * 60)
    print("TIME DECAY GRID SEARCH")
    print(f"Validation: {SEZON_VAL} | Test: {SEZON_TEST}")
    print("=" * 60)

    # GRID SEARCH na validation season
    print("\n[1/3] Grid search na sezonie validation...")
    grid_results = []

    for decay in DECAY_GRID:
        df_val = run_season_backtest(df, SEZON_VAL, decay)
        ll = df_val["log_loss"].mean()
        grid_results.append({"decay": decay, "ll_val": ll})
        print(f"  decay={decay:.3f} → ll={ll:.4f}")

    df_grid = pd.DataFrame(grid_results).sort_values("ll_val")
    best_decay = df_grid.iloc[0]["decay"]
    best_ll_val = df_grid.iloc[0]["ll_val"]

    print(f"\n✅ Najlepszy decay: {best_decay} (ll_val={best_ll_val:.4f})")

    # KALIBRACJA uczona na validation
    print("\n[2/3] Kalibracja na validation...")
    df_val_best = run_season_backtest(df, SEZON_VAL, best_decay)
    cal = fit_calibrator(df_val_best)
    df_val_cal = apply_calibrator(df_val_best, cal)
    print(f"  T={cal['T']:.4f} | bH={cal['bH']:.4f} | bD={cal['bD']:.4f} | bA={cal['bA']:.4f}")
    print(f"  ll_val RAW={df_val_best['log_loss'].mean():.4f} | CAL={df_val_cal['ll_cal'].mean():.4f}")

    # TEST FINALNY na 2025/26
    print("\n[3/3] Test finalny na sezonie 2025/26...")
    df_test_best = run_season_backtest(df, SEZON_TEST, best_decay)
    df_test_cal = apply_calibrator(df_test_best, cal)

    ll_test_raw = df_test_best["log_loss"].mean()
    ll_test_cal = df_test_cal["ll_cal"].mean()

    # Porównaj z poprzednim najlepszym (xG + cal bez decay) = 1.0571
    POPRZEDNI_BEST = 1.0571

    print("\n" + "=" * 60)
    print("WYNIKI FINALNE")
    print("=" * 60)
    print(f"\nNajlepszy decay     : {best_decay}")
    print(f"\nTEST {SEZON_TEST}:")
    print(f"  RAW  log-loss     : {ll_test_raw:.4f}")
    print(f"  CAL  log-loss     : {ll_test_cal:.4f}")
    print(f"\nPoprzedni best      : {POPRZEDNI_BEST}")
    print(f"Różnica             : {POPRZEDNI_BEST - ll_test_cal:+.4f}")

    if ll_test_cal < POPRZEDNI_BEST:
        print("\n✅ TIME DECAY POPRAWIA MODEL")
    else:
        print("\n❌ TIME DECAY NIE POPRAWIA — ZOSTAJEMY PRZY POPRZEDNIM")

    # Szczegółowy raport
    print(f"\n--- ROZKŁAD TYPOWAŃ (argmax CAL) ---")
    pred_cal = df_test_cal[["p_home_cal","p_draw_cal","p_away_cal"]].idxmax(axis=1)
    pred_cal = pred_cal.map({
        "p_home_cal": "H", "p_draw_cal": "D", "p_away_cal": "A"
    })
    for r in ["H", "D", "A"]:
        print(f"  {r}: {(pred_cal == r).sum()}")

    print(f"\n--- ŚREDNIE PRAWDOPODOBIEŃSTWA (CAL) ---")
    print(f"  H = {df_test_cal['p_home_cal'].mean():.3f}")
    print(f"  D = {df_test_cal['p_draw_cal'].mean():.3f}")
    print(f"  A = {df_test_cal['p_away_cal'].mean():.3f}")

    # Zapis
    out = Path("data/processed")
    out.mkdir(exist_ok=True)
    df_grid.to_csv(out / "decay_grid_results.csv", index=False)
    df_test_cal.to_csv(out / "backtesting_final_decay_test.csv", index=False)

    print(f"\nZapisano: decay_grid_results.csv + backtesting_final_decay_test.csv")

    print("\n" + "=" * 60)
    print("GOTOWE")
    print("=" * 60)


if __name__ == "__main__":
    main()