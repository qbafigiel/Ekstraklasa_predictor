"""
baseline_xg_oos.py
===================
Oficjalny Benchmark OOS — Ekstraklasa Predictor.

Rynki:
  - 1X2 (Softmax Calibration)
  - Over/Under: 0.5, 1.5, 2.5, 3.5
  - BTTS (z korekcją biasu)

Schemat OOS:
  - Kalibracja: 2024/25
  - Test:       2025/26
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.stats import poisson
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DB_PATH = Path("db/ekstraklasa.db")
OUTPUT_CSV = Path("data/processed/baseline_xg_oos_predictions.csv")
REPORT_PATH = Path("data/reports/model/model_baseline_oos_report.txt")

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"
MAX_GOLE = 10

WAGI_SEZONOW = {"2023/24": 0.4, "2024/25": 0.7, "2025/26": 1.0}
K_PRIOR = 10
PROMOTED = {
    "2024/25": ["GKS Katowice", "Lechia Gdańsk", "Motor Lublin"],
    "2025/26": ["Arka Gdynia", "Bruk-Bet Termalica Nieciecza", "Wisła Płock"],
}
PRIORS = {
    "2024/25": {"atak": 0.80, "obrona": 1.10},
    "2025/26": {"atak": 0.7744, "obrona": 1.0648},
}


# =============================================================================
# POISSON MLE
# =============================================================================

def przygotuj(df):
    df = df.copy()
    df["xgh"] = df["xg_gosp"].fillna(df["gole_gosp"])
    df["xga"] = df["xg_gosc"].fillna(df["gole_gosc"])
    teams = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    t2i = {t: i for i, t in enumerate(teams)}
    i2t = {i: t for t, i in t2i.items()}
    return {
        "n": len(teams), "i2t": i2t,
        "hi": df["gospodarz"].map(t2i).values,
        "ai": df["gosc"].map(t2i).values,
        "gh": df["xgh"].values.astype(float),
        "ga": df["xga"].values.astype(float),
        "w": df["waga_sezonu"].values.astype(float),
    }


def nll(theta, d):
    N = d["n"]
    la = np.zeros(N)
    lb = np.zeros(N)
    la[:N - 1] = theta[2:2 + N - 1]
    la[N - 1] = -la[:N - 1].sum()
    lb[:N - 1] = theta[2 + N - 1:2 + 2 * (N - 1)]
    lb[N - 1] = -lb[:N - 1].sum()
    mh = np.exp(theta[0])
    ma = np.exp(theta[1])
    a = np.exp(la)
    b = np.exp(lb)
    lh = np.maximum(mh * a[d["hi"]] * b[d["ai"]], 1e-10)
    ll = np.maximum(ma * a[d["ai"]] * b[d["hi"]], 1e-10)
    return -np.sum(d["w"] * (d["gh"] * np.log(lh) - lh + d["ga"] * np.log(ll) - ll))


def fit(df):
    d = przygotuj(df)
    N = d["n"]
    t0 = np.zeros(2 + 2 * (N - 1))
    t0[0] = np.log(max(d["gh"].mean(), 0.01))
    t0[1] = np.log(max(d["ga"].mean(), 0.01))
    res = minimize(nll, t0, args=(d,), method="L-BFGS-B", options={"maxiter": 10000})
    la = np.zeros(N)
    lb = np.zeros(N)
    la[:N - 1] = res.x[2:2 + N - 1]
    la[N - 1] = -la[:N - 1].sum()
    lb[:N - 1] = res.x[2 + N - 1:2 + 2 * (N - 1)]
    lb[N - 1] = -lb[:N - 1].sum()
    return {
        "mh": np.exp(res.x[0]),
        "ma": np.exp(res.x[1]),
        "alpha": {d["i2t"][i]: np.exp(la[i]) for i in range(N)},
        "beta": {d["i2t"][i]: np.exp(lb[i]) for i in range(N)},
    }


def apply_priors(params, season, df_prev):
    for team in PROMOTED.get(season, []):
        n = int(((df_prev["gospodarz"] == team) | (df_prev["gosc"] == team)).sum())
        pr = PRIORS[season]
        a = params["alpha"].get(team, pr["atak"])
        b = params["beta"].get(team, pr["obrona"])
        params["alpha"][team] = (K_PRIOR * pr["atak"] + n * a) / (K_PRIOR + n)
        params["beta"][team] = (K_PRIOR * pr["obrona"] + n * b) / (K_PRIOR + n)
    return params


# =============================================================================
# PREDYKCJA — pełna macierz
# =============================================================================

def predict_full(params, home, away):
    if home not in params["alpha"] or away not in params["alpha"]:
        return None

    lh = params["mh"] * params["alpha"][home] * params["beta"][away]
    la = params["ma"] * params["alpha"][away] * params["beta"][home]

    m = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            m[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)

    total = m.sum()
    if total <= 0:
        return None
    m /= total

    p_home = float(np.sum(np.tril(m, -1)))
    p_draw = float(np.sum(np.diag(m)))
    p_away = float(np.sum(np.triu(m, 1)))

    p_under = {}
    for prog in [0, 1, 2, 3]:
        s = 0.0
        for i in range(MAX_GOLE):
            for j in range(MAX_GOLE):
                if i + j <= prog:
                    s += m[i, j]
        p_under[prog] = s

    p_btts_no = 0.0
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            if i == 0 or j == 0:
                p_btts_no += m[i, j]
    p_btts_yes = 1.0 - p_btts_no

    return {
        "lambda_home": float(lh),
        "lambda_away": float(la),
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over_05": 1.0 - p_under[0],
        "p_over_15": 1.0 - p_under[1],
        "p_over_25": 1.0 - p_under[2],
        "p_over_35": 1.0 - p_under[3],
        "p_under_05": p_under[0],
        "p_under_15": p_under[1],
        "p_under_25": p_under[2],
        "p_under_35": p_under[3],
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
    }


# =============================================================================
# HELPERS
# =============================================================================

def wynik(gh, ga):
    return "H" if gh > ga else ("D" if gh == ga else "A")


def softmax(x):
    e = np.exp(np.array(x) - np.max(x))
    return e / e.sum()


def calibrate_1x2(ph, pd_, pa, T, bH, bD, bA):
    logits = np.log(np.maximum([ph, pd_, pa], 1e-12))
    return softmax((logits + [bH, bD, bA]) / T)


def ll_binary(p_pred, actual):
    if actual == 1:
        return -np.log(max(p_pred, 1e-12))
    else:
        return -np.log(max(1.0 - p_pred, 1e-12))


# =============================================================================
# KALIBRATORY
# =============================================================================

def fit_cal_1x2(df):
    L2_PENALTY = 0.05

    def obj(p):
        T, bH, bD, bA = p
        if T <= 0.1:
            return 999.0
        ll = []
        for r in df.itertuples():
            p_cal = calibrate_1x2(r.p_home_raw, r.p_draw_raw, r.p_away_raw, T, bH, bD, bA)
            idx = {"H": 0, "D": 1, "A": 2}[r.wynik_1x2]
            ll.append(-np.log(max(p_cal[idx], 1e-12)))
        penalty = L2_PENALTY * (bH ** 2 + bD ** 2 + bA ** 2)
        return np.mean(ll) + penalty

    return minimize(
        obj, [1.5, 0.0, 0.0, 0.0],
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-1, 1), (-1, 1), (-1, 1)]
    ).x


def fit_btts_correction(df):
    """
    Szuka optymalnego additive shift dla p_btts_yes
    minimalizując log-loss BTTS na zbiorze kalibracyjnym.

    p_btts_corrected = clip(p_btts_yes + shift, 0.01, 0.99)
    """
    p_raw = df["p_btts_yes"].values
    y = df["btts_rzecz"].values

    def obj(shift):
        total = 0.0
        for p, actual in zip(p_raw, y):
            p_corr = np.clip(p + shift, 0.01, 0.99)
            total += ll_binary(p_corr, actual)
        return total / len(y)

    result = minimize_scalar(obj, bounds=(-0.15, 0.15), method="bounded")
    return float(result.x)


# =============================================================================
# ROLLING BACKTESTING
# =============================================================================

def run_season_raw(df_all, target):
    seasons = sorted(df_all["sezon"].unique())
    idx = seasons.index(target)
    df_hist = df_all[df_all["sezon"].isin(seasons[:idx])].copy()
    df_test = df_all[df_all["sezon"] == target].copy()
    rows = []

    for k in sorted(df_test["kolejka"].unique()):
        df_prev = df_test[df_test["kolejka"] < k]
        df_train = pd.concat([df_hist, df_prev], ignore_index=True)
        params = apply_priors(fit(df_train), target, df_prev)

        for _, mecz in df_test[df_test["kolejka"] == k].iterrows():
            pred = predict_full(params, mecz["gospodarz"], mecz["gosc"])
            if pred is None:
                continue

            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])
            suma = gh + ga
            w = wynik(gh, ga)

            rows.append({
                "sezon": target,
                "kolejka": k,
                "match_id": mecz["match_id"],
                "gospodarz": mecz["gospodarz"],
                "gosc": mecz["gosc"],
                "gole_gosp": gh,
                "gole_gosc": ga,
                "suma_goli": suma,
                "wynik_1x2": w,
                "btts_rzecz": int(gh >= 1 and ga >= 1),
                "over05_rzecz": int(suma > 0),
                "over15_rzecz": int(suma > 1),
                "over25_rzecz": int(suma > 2),
                "over35_rzecz": int(suma > 3),
                "lambda_home": pred["lambda_home"],
                "lambda_away": pred["lambda_away"],
                "p_home_raw": pred["p_home"],
                "p_draw_raw": pred["p_draw"],
                "p_away_raw": pred["p_away"],
                "p_over_05": pred["p_over_05"],
                "p_over_15": pred["p_over_15"],
                "p_over_25": pred["p_over_25"],
                "p_over_35": pred["p_over_35"],
                "p_under_05": pred["p_under_05"],
                "p_under_15": pred["p_under_15"],
                "p_under_25": pred["p_under_25"],
                "p_under_35": pred["p_under_35"],
                "p_btts_yes": pred["p_btts_yes"],
                "p_btts_no": pred["p_btts_no"],
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytywanie meczów z bazy...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    print(f"\n2. Generowanie surowych predykcji dla {VAL_SEASON}...")
    df_val = run_season_raw(df, VAL_SEASON)
    print(f"   Predykcji: {len(df_val)}")

    print(f"\n3. Generowanie surowych predykcji dla {TEST_SEASON}...")
    df_test = run_season_raw(df, TEST_SEASON)
    print(f"   Predykcji: {len(df_test)}")

    # --- 1X2 kalibracja ---
    print("\n4. Trenowanie kalibratora 1X2 na 2024/25...")
    cal_1x2 = fit_cal_1x2(df_val)
    T, bH, bD, bA = cal_1x2

    # --- BTTS korekcja ---
    print("5. Trenowanie korekcji BTTS na 2024/25...")
    btts_shift = fit_btts_correction(df_val)
    print(f"   BTTS shift: {btts_shift:+.4f}")

    # --- Aplikacja kalibracji ---
    def apply_cal(df_raw):
        df_out = df_raw.copy()

        # 1X2
        cal_results = df_out.apply(
            lambda r: calibrate_1x2(r.p_home_raw, r.p_draw_raw, r.p_away_raw, T, bH, bD, bA),
            axis=1
        )
        df_out["p_home_cal"] = [p[0] for p in cal_results]
        df_out["p_draw_cal"] = [p[1] for p in cal_results]
        df_out["p_away_cal"] = [p[2] for p in cal_results]

        df_out["ll_1x2"] = df_out.apply(
            lambda r: -np.log(max(
                {"H": r.p_home_cal, "D": r.p_draw_cal, "A": r.p_away_cal}[r.wynik_1x2],
                1e-12
            )),
            axis=1
        )

        # BTTS — z korekcją
        df_out["p_btts_yes_cal"] = np.clip(df_out["p_btts_yes"] + btts_shift, 0.01, 0.99)
        df_out["p_btts_no_cal"] = 1.0 - df_out["p_btts_yes_cal"]

        df_out["ll_btts"] = df_out.apply(
            lambda r: ll_binary(r.p_btts_yes_cal, r.btts_rzecz),
            axis=1
        )

        df_out["ll_btts_raw"] = df_out.apply(
            lambda r: ll_binary(r.p_btts_yes, r.btts_rzecz),
            axis=1
        )

        # O/U
        for line, col_pred, col_actual in [
            ("05", "p_over_05", "over05_rzecz"),
            ("15", "p_over_15", "over15_rzecz"),
            ("25", "p_over_25", "over25_rzecz"),
            ("35", "p_over_35", "over35_rzecz"),
        ]:
            df_out[f"ll_over_{line}"] = df_out.apply(
                lambda r: ll_binary(r[col_pred], r[col_actual]),
                axis=1
            )

        return df_out

    df_val_final = apply_cal(df_val)
    df_test_final = apply_cal(df_test)

    df_test_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ==========================================================================
    # RAPORT
    # ==========================================================================
    def fmt_market(df_, col_pred, col_actual, name):
        p_avg = df_[col_pred].mean()
        r_avg = df_[col_actual].mean()
        delta = r_avg - p_avg
        ok = "OK" if abs(delta) < 0.05 else "BIAS"
        return f"  {name:12s}: model={p_avg:.3f} | rzecz={r_avg:.3f} | delta={delta:+.3f} {ok}"

    ll_1x2_val = df_val_final["ll_1x2"].mean()
    ll_1x2_test = df_test_final["ll_1x2"].mean()

    ll_btts_val_raw = df_val_final["ll_btts_raw"].mean()
    ll_btts_val_cal = df_val_final["ll_btts"].mean()
    ll_btts_test_raw = df_test_final["ll_btts_raw"].mean()
    ll_btts_test_cal = df_test_final["ll_btts"].mean()

    lines = []
    lines.append("=" * 78)
    lines.append("OFICJALNY BENCHMARK OOS — EKSTRAKLASA PREDICTOR")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. ARCHITEKTURA")
    lines.append("-" * 78)
    lines.append("  Poisson MLE na xG (fallback gole 23/24)")
    lines.append("  Wagi sezonowe: 0.4 / 0.7 / 1.0")
    lines.append("  Priory beniaminków (K=10)")
    lines.append("  Softmax Calibration (L2=0.05) dla 1X2")
    lines.append(f"  BTTS additive shift: {btts_shift:+.4f}")
    lines.append("")
    lines.append("2. KALIBRACJA 1X2 (2024/25)")
    lines.append("-" * 78)
    lines.append(f"  Log-loss in-sample:  {ll_1x2_val:.4f}")
    lines.append(f"  T={T:.4f}, bH={bH:.4f}, bD={bD:.4f}, bA={bA:.4f}")
    lines.append("")
    lines.append("3. TEST OOS 1X2 (2025/26)")
    lines.append("-" * 78)
    lines.append(f"  Meczów:              {len(df_test_final)}")
    lines.append(f"  LOG-LOSS OOS 1X2:    {ll_1x2_test:.4f}")
    lines.append(f"  Benchmark losowy:    {np.log(3):.4f}")

    early = df_test_final[df_test_final["kolejka"] <= 5]
    late = df_test_final[df_test_final["kolejka"] > 5]
    lines.append(f"  Kolejki 1-5:         {early['ll_1x2'].mean():.4f}")
    lines.append(f"  Kolejki 6-34:        {late['ll_1x2'].mean():.4f}")
    lines.append("")

    lines.append("4. OVER/UNDER — KALIBRACJA")
    lines.append("-" * 78)
    lines.append("  [2024/25 — in-sample]")
    for col_pred, col_actual, nice in [
        ("p_over_05", "over05_rzecz", "Over 0.5"),
        ("p_over_15", "over15_rzecz", "Over 1.5"),
        ("p_over_25", "over25_rzecz", "Over 2.5"),
        ("p_over_35", "over35_rzecz", "Over 3.5"),
    ]:
        lines.append(fmt_market(df_val_final, col_pred, col_actual, nice))

    lines.append("")
    lines.append("  [2025/26 — OOS]")
    for col_pred, col_actual, nice in [
        ("p_over_05", "over05_rzecz", "Over 0.5"),
        ("p_over_15", "over15_rzecz", "Over 1.5"),
        ("p_over_25", "over25_rzecz", "Over 2.5"),
        ("p_over_35", "over35_rzecz", "Over 3.5"),
    ]:
        lines.append(fmt_market(df_test_final, col_pred, col_actual, nice))
    lines.append("")

    lines.append("5. OVER/UNDER — LOG-LOSS OOS")
    lines.append("-" * 78)
    for line_name, nice in [("05", "Over 0.5"), ("15", "Over 1.5"), ("25", "Over 2.5"), ("35", "Over 3.5")]:
        ll_val_ou = df_val_final[f"ll_over_{line_name}"].mean()
        ll_test_ou = df_test_final[f"ll_over_{line_name}"].mean()
        bench = -np.log(0.5)
        lines.append(f"  {nice:12s}: VAL={ll_val_ou:.4f} | OOS={ll_test_ou:.4f} | bench_50/50={bench:.4f}")
    lines.append("")

    lines.append("6. BTTS — KALIBRACJA")
    lines.append("-" * 78)
    lines.append("  [2024/25 — in-sample]")
    lines.append(fmt_market(df_val_final, "p_btts_yes", "btts_rzecz", "BTTS raw"))
    lines.append(fmt_market(df_val_final, "p_btts_yes_cal", "btts_rzecz", "BTTS cal"))
    lines.append("")
    lines.append("  [2025/26 — OOS]")
    lines.append(fmt_market(df_test_final, "p_btts_yes", "btts_rzecz", "BTTS raw"))
    lines.append(fmt_market(df_test_final, "p_btts_yes_cal", "btts_rzecz", "BTTS cal"))
    lines.append("")

    lines.append("7. BTTS — LOG-LOSS")
    lines.append("-" * 78)
    lines.append(f"  BTTS raw:     VAL={ll_btts_val_raw:.4f} | OOS={ll_btts_test_raw:.4f}")
    lines.append(f"  BTTS cal:     VAL={ll_btts_val_cal:.4f} | OOS={ll_btts_test_cal:.4f}")
    lines.append(f"  Poprawa OOS:  {ll_btts_test_raw - ll_btts_test_cal:+.4f}")
    lines.append(f"  bench 50/50:  {-np.log(0.5):.4f}")
    lines.append("")

    lines.append("8. PODSUMOWANIE WSZYSTKICH RYNKÓW OOS")
    lines.append("-" * 78)
    lines.append(f"  1X2:          {ll_1x2_test:.4f}")
    lines.append(f"  Over 2.5:     {df_test_final['ll_over_25'].mean():.4f}")
    lines.append(f"  BTTS raw:     {ll_btts_test_raw:.4f}")
    lines.append(f"  BTTS cal:     {ll_btts_test_cal:.4f}")
    lines.append(f"  Benchmark:")
    lines.append(f"    1X2 losowy:    {np.log(3):.4f}")
    lines.append(f"    binarny 50/50: {-np.log(0.5):.4f}")
    lines.append("")
    lines.append("9. PLIKI")
    lines.append("-" * 78)
    lines.append(f"  Predykcje: {OUTPUT_CSV}")
    lines.append(f"  Raport:    {REPORT_PATH}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {OUTPUT_CSV}")
    print(f"Zapisano: {REPORT_PATH}")


if __name__ == "__main__":
    main()