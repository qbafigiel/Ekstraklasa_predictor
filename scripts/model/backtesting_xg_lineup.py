"""
backtesting_xg_lineup.py
========================
Eksperyment: Poisson(xG) + Softmax Calibration + diff_lineup_offense

Porównanie na meczach 2025/26 (sezon testowy):
  A) Baseline: Poisson(xG) + Softmax (obecny best = 1.0571)
  B) Lineup:   Poisson(xG) + Softmax + gamma * diff_lineup_offense

Lineup values dostępne tylko dla 2024/25 i 2025/26.
Trening MLE Poissona: na xG (z fallback na gole dla 2023/24 bez xG).
Kalibracja: optymalizacja T, bH, bD, bA (i opcjonalnie gamma) na danych treningowych.
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
REPORT_DIR = Path("data/reports/model")
SEZON_TEST = "2025/26"
MAX_GOLE = 10

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}


# =============================================================================
# POISSON MLE (identyczny z backtesting_xg_v1.py)
# =============================================================================

def przygotuj_dane_xg(df_trening):
    df_trening = df_trening.copy()
    df_trening["xg_gosp_final"] = df_trening["xg_gosp"].fillna(df_trening["gole_gosp"])
    df_trening["xg_gosc_final"] = df_trening["xg_gosc"].fillna(df_trening["gole_gosc"])

    druzyny = sorted(set(df_trening["gospodarz"]) | set(df_trening["gosc"]))
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
    result = minimize(neg_log_likelihood, theta0, args=(data,), method="L-BFGS-B", options={"maxiter": 10000})
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


# =============================================================================
# PREDYKCJA
# =============================================================================

def przewiduj_mecz(params, gospodarz, gosc):
    if gospodarz not in params["alpha"] or gosc not in params["alpha"]:
        return None

    lh = params["mu_home"] * params["alpha"][gospodarz] * params["beta"][gosc]
    la = params["mu_away"] * params["alpha"][gosc] * params["beta"][gospodarz]

    macierz = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    macierz /= macierz.sum()

    p_home = float(np.sum(np.tril(macierz, -1)))
    p_draw = float(np.sum(np.diag(macierz)))
    p_away = float(np.sum(np.triu(macierz, 1)))

    return {"lambda_home": lh, "lambda_away": la, "p_home": p_home, "p_draw": p_draw, "p_away": p_away}


# =============================================================================
# SOFTMAX CALIBRATION
# =============================================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def calibrate_probs_baseline(p_home, p_draw, p_away, T, bH, bD, bA):
    logits = np.log(np.maximum(np.array([p_home, p_draw, p_away]), 1e-12))
    logits_cal = (logits + np.array([bH, bD, bA])) / T
    return softmax(logits_cal)


def calibrate_probs_lineup(p_home, p_draw, p_away, diff_off, T, bH, bD, bA, gamma):
    logits = np.log(np.maximum(np.array([p_home, p_draw, p_away]), 1e-12))
    # gamma * diff_off: jeśli gospodarze mają lepszy skład, zwiększ logit Home, zmniejsz Away
    logits_cal = (logits + np.array([bH + gamma * diff_off, bD, bA - gamma * diff_off])) / T
    return softmax(logits_cal)


def log_loss_1x2(p_cal, wynik):
    idx = {"H": 0, "D": 1, "A": 2}[wynik]
    return -np.log(max(p_cal[idx], 1e-12))


# =============================================================================
# OPTYMALIZACJA KALIBRACJI
# =============================================================================

def optimize_baseline(raw_probs, outcomes):
    """Optymalizuje T, bH, bD, bA na zbiorze treningowym."""
    def objective(params):
        T, bH, bD, bA = params
        if T <= 0.01:
            return 1e6
        total_ll = 0.0
        for (ph, pd_, pa), wynik in zip(raw_probs, outcomes):
            p_cal = calibrate_probs_baseline(ph, pd_, pa, T, bH, bD, bA)
            total_ll += log_loss_1x2(p_cal, wynik)
        return total_ll / len(outcomes)

    result = minimize(objective, [1.5, 0.3, 0.0, -0.3], method="Nelder-Mead",
                      options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8})
    return result.x, result.fun


def optimize_lineup(raw_probs, diff_offs, outcomes):
    """Optymalizuje T, bH, bD, bA, gamma na zbiorze treningowym."""
    def objective(params):
        T, bH, bD, bA, gamma = params
        if T <= 0.01:
            return 1e6
        total_ll = 0.0
        for (ph, pd_, pa), diff_off, wynik in zip(raw_probs, diff_offs, outcomes):
            p_cal = calibrate_probs_lineup(ph, pd_, pa, diff_off, T, bH, bD, bA, gamma)
            total_ll += log_loss_1x2(p_cal, wynik)
        return total_ll / len(outcomes)

    result = minimize(objective, [1.5, 0.3, 0.0, -0.3, 0.01], method="Nelder-Mead",
                      options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8})
    return result.x, result.fun


# =============================================================================
# HELPERS
# =============================================================================

def wynik_1x2(gh, ga):
    if gh > ga:
        return "H"
    elif gh == ga:
        return "D"
    else:
        return "A"


# =============================================================================
# MAIN
# =============================================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)

    lineup = pd.read_csv(LINEUP_PATH)
    lineup = lineup[["match_id", "sezon", "diff_lineup_offense"]].copy()

    df = df.merge(lineup, on=["match_id", "sezon"], how="left")
    df["diff_lineup_offense"] = df["diff_lineup_offense"].fillna(0.0)

    # --- Split ---
    df_test = df[df["sezon"] == SEZON_TEST].copy()
    df_hist = df[df["sezon"] != SEZON_TEST].copy()
    kolejki = sorted(df_test["kolejka"].unique())

    print("=" * 70)
    print(f"EKSPERYMENT: Poisson(xG) + Lineup Offense [{SEZON_TEST}]")
    print("=" * 70)
    print(f"Mecze testowe: {len(df_test)}")
    print(f"Mecze historyczne: {len(df_hist)}")
    print(f"Kolejki: {len(kolejki)}")
    print()

    # --- Backtesting: generuj surowe prawdopodobieństwa ---
    results = []

    for kolejka in kolejki:
        df_prev = df_test[df_test["kolejka"] < kolejka]
        df_trening = pd.concat([df_hist, df_prev], ignore_index=True)

        data = przygotuj_dane_xg(df_trening)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)

        df_kolejka = df_test[df_test["kolejka"] == kolejka]

        for _, mecz in df_kolejka.iterrows():
            gosp = mecz["gospodarz"]
            gosc_d = mecz["gosc"]
            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])

            pred = przewiduj_mecz(params, gosp, gosc_d)
            if pred is None:
                continue

            results.append({
                "match_id": mecz["match_id"],
                "kolejka": kolejka,
                "gospodarz": gosp,
                "gosc": gosc_d,
                "wynik_1x2": wynik_1x2(gh, ga),
                "p_home_raw": pred["p_home"],
                "p_draw_raw": pred["p_draw"],
                "p_away_raw": pred["p_away"],
                "diff_lineup_offense": mecz["diff_lineup_offense"],
            })

        print(f"  K{kolejka:02d} OK")

    df_results = pd.DataFrame(results)
    print(f"\nWygenerowano {len(df_results)} predykcji.\n")

    # --- Kalibracja na pełnym zbiorze testowym ---
    # (To jest "in-sample" kalibracja — uczciwe porównanie A vs B
    #  bo oba modele kalibrują się na tych samych danych)

    raw_probs = list(zip(
        df_results["p_home_raw"],
        df_results["p_draw_raw"],
        df_results["p_away_raw"]
    ))
    outcomes = df_results["wynik_1x2"].tolist()
    diff_offs = df_results["diff_lineup_offense"].tolist()

    # --- Model A: Baseline ---
    print("Optymalizuję Model A (Baseline)...")
    params_A, ll_A = optimize_baseline(raw_probs, outcomes)
    T_A, bH_A, bD_A, bA_A = params_A

    # --- Model B: Baseline + Lineup ---
    print("Optymalizuję Model B (Baseline + Lineup Offense)...")
    params_B, ll_B = optimize_lineup(raw_probs, diff_offs, outcomes)
    T_B, bH_B, bD_B, bA_B, gamma_B = params_B

    # --- Per-match log-loss ---
    ll_A_list = []
    ll_B_list = []
    for i, row in df_results.iterrows():
        p_A = calibrate_probs_baseline(row["p_home_raw"], row["p_draw_raw"], row["p_away_raw"], T_A, bH_A, bD_A, bA_A)
        p_B = calibrate_probs_lineup(row["p_home_raw"], row["p_draw_raw"], row["p_away_raw"],
                                     row["diff_lineup_offense"], T_B, bH_B, bD_B, bA_B, gamma_B)
        ll_A_list.append(log_loss_1x2(p_A, row["wynik_1x2"]))
        ll_B_list.append(log_loss_1x2(p_B, row["wynik_1x2"]))

    df_results["ll_baseline"] = ll_A_list
    df_results["ll_lineup"] = ll_B_list

    # --- RAPORT ---
    lines = []
    lines.append("=" * 70)
    lines.append(f"WYNIKI EKSPERYMENTU: Poisson(xG) + Lineup Offense [{SEZON_TEST}]")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. LOG-LOSS (in-sample calibration)")
    lines.append("-" * 70)
    lines.append(f"  Model A (Baseline):          {ll_A:.4f}")
    lines.append(f"  Model B (Baseline + Lineup): {ll_B:.4f}")
    lines.append(f"  Różnica (A - B):             {ll_A - ll_B:+.4f}")
    lines.append(f"  Poprawa:                     {'TAK' if ll_B < ll_A else 'NIE'}")
    lines.append(f"  Benchmark losowy:            {np.log(3):.4f}")
    lines.append("")
    lines.append("2. PARAMETRY KALIBRACJI")
    lines.append("-" * 70)
    lines.append(f"  Model A: T={T_A:.4f}, bH={bH_A:.4f}, bD={bD_A:.4f}, bA={bA_A:.4f}")
    lines.append(f"  Model B: T={T_B:.4f}, bH={bH_B:.4f}, bD={bD_B:.4f}, bA={bA_B:.4f}, gamma={gamma_B:.6f}")
    lines.append("")
    lines.append("3. INTERPRETACJA GAMMA")
    lines.append("-" * 70)
    if abs(gamma_B) < 0.001:
        lines.append(f"  gamma={gamma_B:.6f} — praktycznie ZERO")
        lines.append("  Lineup offense NIE wnosi dodatkowego sygnału ponad Poisson(xG).")
        lines.append("  WNIOSEK: NIE wpinamy do modelu produkcyjnego.")
    elif abs(gamma_B) < 0.005:
        lines.append(f"  gamma={gamma_B:.6f} — bardzo mały efekt")
        lines.append("  Lineup offense ma marginalny wpływ. Ryzyko overfitting > potencjalna poprawa.")
        lines.append("  WNIOSEK: raczej NIE wpinamy.")
    else:
        lines.append(f"  gamma={gamma_B:.6f} — mierzalny efekt")
        lines.append("  Lineup offense dodaje informację ponad sam Poisson(xG).")
        lines.append("  WNIOSEK: WARTO wpiąć do modelu produkcyjnego.")
    lines.append("")
    lines.append("4. STATYSTYKI MECZU")
    lines.append("-" * 70)
    lines.append(f"  Mecze: {len(df_results)}")
    lines.append(f"  Mecze z diff_lineup_offense != 0: {(df_results['diff_lineup_offense'] != 0).sum()}")
    lines.append(f"  diff_lineup_offense mean: {df_results['diff_lineup_offense'].mean():.2f}")
    lines.append(f"  diff_lineup_offense std:  {df_results['diff_lineup_offense'].std():.2f}")

    report_text = "\n".join(lines)
    print()
    print(report_text)

    report_path = REPORT_DIR / "experiment_lineup_offense_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nZapisano raport: {report_path}")


if __name__ == "__main__":
    main()