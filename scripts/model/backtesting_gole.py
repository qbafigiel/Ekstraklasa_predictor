"""
backtesting_gole.py
====================
Backtesting modelu Poissona dla goli.
Symulacja kolejka po kolejce na sezonie 2025/26.

Rynki:
  - 1X2
  - BTTS
  - Over/Under: 0.5 / 1.5 / 2.5 / 3.5

Źródło: db/ekstraklasa.db
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings

warnings.filterwarnings("ignore")

# ── ścieżki ───────────────────────────────────────────────────────────────────
DB_PATH = Path("db/ekstraklasa.db")
OUTPUT_PATH = Path("data/processed/backtesting_wyniki.csv")
SEZON_TEST = "2025/26"
MAX_GOLE = 10

# ── wagi sezonów ──────────────────────────────────────────────────────────────
WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

# =============================================================================
# MODEL
# =============================================================================

def przygotuj_dane(df_trening):
    druzyny = sorted(
        set(df_trening["gospodarz"].unique()) |
        set(df_trening["gosc"].unique())
    )
    n = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    return {
        "druzyny": druzyny,
        "n_druzyn": n,
        "team_to_idx": t2i,
        "idx_to_team": i2t,
        "home_idx": df_trening["gospodarz"].map(t2i).values,
        "away_idx": df_trening["gosc"].map(t2i).values,
        "goals_home": df_trening["gole_gosp"].values.astype(float),
        "goals_away": df_trening["gole_gosc"].values.astype(float),
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
        neg_log_likelihood, theta0, args=(data,),
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
        "alpha": {data["idx_to_team"][i]: np.exp(log_alpha[i])
                  for i in range(N)},
        "beta":  {data["idx_to_team"][i]: np.exp(log_beta[i])
                  for i in range(N)},
    }


def przewiduj_mecz(params, gospodarz, gosc):
    if gospodarz not in params["alpha"] or gosc not in params["alpha"]:
        return None

    lh = (params["mu_home"]
          * params["alpha"][gospodarz]
          * params["beta"][gosc])
    la = (params["mu_away"]
          * params["alpha"][gosc]
          * params["beta"][gospodarz])

    # macierz wyników
    macierz = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    macierz /= macierz.sum()

    # 1X2
    p_home = float(np.sum(np.tril(macierz, -1)))
    p_draw = float(np.sum(np.diag(macierz)))
    p_away = float(np.sum(np.triu(macierz, 1)))

    # BTTS
    p_btts = float(
        (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
    )

    # ── Over/Under — wszystkie linie ──────────────────────────────────────────
    # sumujemy macierz po łącznej liczbie goli
    p_under = {}
    for prog in [0, 1, 2, 3]:
        s = 0.0
        for i in range(MAX_GOLE):
            for j in range(MAX_GOLE):
                if i + j <= prog:
                    s += macierz[i, j]
        p_under[prog] = s

    # Under X.5 = P(suma <= X)
    # Over  X.5 = 1 - Under X.5
    return {
        "lambda_home":  round(lh, 4),
        "lambda_away":  round(la, 4),

        # 1X2
        "p_home":       p_home,
        "p_draw":       p_draw,
        "p_away":       p_away,

        # BTTS
        "p_btts":       p_btts,

        # Under X.5
        "p_under_05":   p_under[0],   # tylko 0:0
        "p_under_15":   p_under[1],   # suma <= 1
        "p_under_25":   p_under[2],   # suma <= 2
        "p_under_35":   p_under[3],   # suma <= 3

        # Over X.5
        "p_over_05":    1 - p_under[0],
        "p_over_15":    1 - p_under[1],
        "p_over_25":    1 - p_under[2],
        "p_over_35":    1 - p_under[3],
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
    df_historia = df[df["sezon"] != SEZON_TEST].copy()
    kolejki = sorted(df_test["kolejka"].unique())

    print(f"Sezon testowy:  {SEZON_TEST}")
    print(f"Kolejek:        {len(kolejki)}")
    print(f"Meczów testu:   {len(df_test)}")
    print(f"Danych hist.:   {len(df_historia)} meczów\n")

    wyniki = []
    bledy = []

    for kolejka in kolejki:
        df_poprzednie = df_test[df_test["kolejka"] < kolejka]
        df_trening = pd.concat(
            [df_historia, df_poprzednie], ignore_index=True
        )

        data = przygotuj_dane(df_trening)
        theta = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)

        df_kolejka = df_test[df_test["kolejka"] == kolejka]
        n_ok = 0
        n_brak = 0

        for _, mecz in df_kolejka.iterrows():
            gosp = mecz["gospodarz"]
            gosc_d = mecz["gosc"]
            gh = int(mecz["gole_gosp"])
            ga = int(mecz["gole_gosc"])
            suma = gh + ga

            pred = przewiduj_mecz(params, gosp, gosc_d)
            if pred is None:
                n_brak += 1
                bledy.append(
                    f"K{kolejka}: brak parametrów dla {gosp} / {gosc_d}"
                )
                continue

            n_ok += 1
            wynik_rzecz = wynik_1x2(gh, ga)

            # typowanie 1X2
            p_max = max(pred["p_home"], pred["p_draw"], pred["p_away"])
            wynik_pred = (
                "H" if p_max == pred["p_home"] else
                "D" if p_max == pred["p_draw"] else "A"
            )

            ll = log_loss_meczu(
                pred["p_home"], pred["p_draw"],
                pred["p_away"], wynik_rzecz
            )

            wyniki.append({
                # identyfikacja
                "kolejka":          int(kolejka),
                "gospodarz":        gosp,
                "gosc":             gosc_d,
                "mecze_treningowe": len(df_trening),

                # rzeczywistość
                "gole_gosp":        gh,
                "gole_gosc":        ga,
                "suma_goli":        suma,
                "wynik_1x2":        wynik_rzecz,

                # rzeczywiste wyniki rynków
                "btts_rzecz":       int(gh >= 1 and ga >= 1),
                "over05_rzecz":     int(suma > 0),
                "over15_rzecz":     int(suma > 1),
                "over25_rzecz":     int(suma > 2),
                "over35_rzecz":     int(suma > 3),

                # lambda
                "lambda_home":      pred["lambda_home"],
                "lambda_away":      pred["lambda_away"],
                "lambda_total":     round(
                    pred["lambda_home"] + pred["lambda_away"], 4
                ),

                # predykcje 1X2
                "p_home":           round(pred["p_home"], 4),
                "p_draw":           round(pred["p_draw"], 4),
                "p_away":           round(pred["p_away"], 4),
                "wynik_pred":       wynik_pred,
                "czy_trafil":       int(wynik_pred == wynik_rzecz),
                "log_loss":         round(ll, 4),

                # predykcje BTTS
                "p_btts":           round(pred["p_btts"], 4),

                # predykcje Under
                "p_under_05":       round(pred["p_under_05"], 4),
                "p_under_15":       round(pred["p_under_15"], 4),
                "p_under_25":       round(pred["p_under_25"], 4),
                "p_under_35":       round(pred["p_under_35"], 4),

                # predykcje Over
                "p_over_05":        round(pred["p_over_05"], 4),
                "p_over_15":        round(pred["p_over_15"], 4),
                "p_over_25":        round(pred["p_over_25"], 4),
                "p_over_35":        round(pred["p_over_35"], 4),

                # błąd goli
                "blad_lambda_home": round(abs(pred["lambda_home"] - gh), 4),
                "blad_lambda_away": round(abs(pred["lambda_away"] - ga), 4),
            })

        print(f"  Kolejka {kolejka:2d}: {n_ok} predykcji | "
              f"{n_brak} brak danych")

    return pd.DataFrame(wyniki), bledy


# =============================================================================
# ANALIZA
# =============================================================================

def analizuj_wyniki(df_wyniki):
    n = len(df_wyniki)
    log_loss_losowy = -np.log(1 / 3)
    log_loss_sredni = df_wyniki["log_loss"].mean()

    # ── 1. METRYKI GLOBALNE ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. METRYKI GLOBALNE")
    print("=" * 60)

    accuracy = df_wyniki["czy_trafil"].mean()
    p_gosp_wygrywa = (df_wyniki["wynik_1x2"] == "H").mean()

    print(f"\n  Predykcji łącznie:       {n}")
    print(f"  Log-loss modelu:         {log_loss_sredni:.4f}")
    print(f"  Log-loss losowy (1/3):   {log_loss_losowy:.4f}")
    print(f"  Poprawa vs losowy:       "
          f"{log_loss_losowy - log_loss_sredni:+.4f}")
    print(f"\n  Accuracy 1X2:            {accuracy:.1%}")
    print(f"  Benchmark 'zawsze H':    {p_gosp_wygrywa:.1%}")

    print(f"\n  Rozkład typowań modelu:")
    for w, label in [("H", "Gosp."), ("D", "Remis"), ("A", "Gość")]:
        n_pred = (df_wyniki["wynik_pred"] == w).sum()
        n_real = (df_wyniki["wynik_1x2"] == w).sum()
        print(f"    {label}: typował {n_pred}x | rzeczywiste {n_real}x")

    # ── 2. MAE GOLI ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. BŁĄD PREDYKCJI GOLI (MAE)")
    print("=" * 60)

    print(f"\n  MAE goli gospodarza:  {df_wyniki['blad_lambda_home'].mean():.4f}")
    print(f"  MAE goli gościa:      {df_wyniki['blad_lambda_away'].mean():.4f}")
    mae_total = (df_wyniki["lambda_total"] - df_wyniki["suma_goli"]).abs().mean()
    print(f"  MAE goli łącznie:     {mae_total:.4f}")

    print(f"\n  Avg rzeczywiste gole gosp: {df_wyniki['gole_gosp'].mean():.3f}")
    print(f"  Avg lambda_home modelu:    {df_wyniki['lambda_home'].mean():.3f}")
    print(f"  Avg rzeczywiste gole gosc: {df_wyniki['gole_gosc'].mean():.3f}")
    print(f"  Avg lambda_away modelu:    {df_wyniki['lambda_away'].mean():.3f}")

    # ── 3. KALIBRACJA 1X2 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. KALIBRACJA 1X2 (kubełki co 10%)")
    print("=" * 60)

    kalibracja_rows = []
    for _, row in df_wyniki.iterrows():
        for kol_p, wynik_k in [
            ("p_home", "H"),
            ("p_draw", "D"),
            ("p_away", "A"),
        ]:
            kalibracja_rows.append({
                "p": row[kol_p],
                "zaszlo": int(row["wynik_1x2"] == wynik_k),
            })

    df_kal = pd.DataFrame(kalibracja_rows)
    bins = np.arange(0, 1.1, 0.1)
    df_kal["kubelek"] = pd.cut(df_kal["p"], bins=bins)

    print(f"\n  {'Przedział':12} {'N':>6} "
          f"{'Avg p':>8} {'Rzeczyw.':>10} {'Różnica':>10}")
    print("  " + "-" * 52)

    for kubelek, grupa in df_kal.groupby("kubelek", observed=True):
        if len(grupa) == 0:
            continue
        avg_p = grupa["p"].mean()
        freq = grupa["zaszlo"].mean()
        diff = freq - avg_p
        znak = ("✅" if abs(diff) < 0.05
                else "⚠️ " if abs(diff) < 0.10
                else "❌")
        print(f"  {str(kubelek):12} {len(grupa):>6} "
              f"{avg_p:>8.3f} {freq:>10.3f} "
              f"{diff:>+10.3f} {znak}")

    # ── 4. KALIBRACJA RYNKÓW — ROZSZERZONA ───────────────────────────────────
    print("\n" + "=" * 60)
    print("4. KALIBRACJA RYNKÓW")
    print("=" * 60)

    rynki = [
        ("BTTS",      "p_btts",    "btts_rzecz"),
        ("Over 0.5",  "p_over_05", "over05_rzecz"),
        ("Over 1.5",  "p_over_15", "over15_rzecz"),
        ("Over 2.5",  "p_over_25", "over25_rzecz"),
        ("Over 3.5",  "p_over_35", "over35_rzecz"),
        ("Under 0.5", "p_under_05","over05_rzecz"),  # under 0.5 = NOT over 0.5
        ("Under 1.5", "p_under_15","over15_rzecz"),
        ("Under 2.5", "p_under_25","over25_rzecz"),
        ("Under 3.5", "p_under_35","over35_rzecz"),
    ]

    print(f"\n  {'Rynek':<12} {'Avg p modelu':>14} "
          f"{'Rzeczyw. %':>12} {'Różnica':>10} {'Ocena':>8}")
    print("  " + "-" * 62)

    for nazwa, kol_pred, kol_rzecz in rynki:
        avg_pred = df_wyniki[kol_pred].mean()

        # dla Under: rzeczywista częstość = 1 - over
        if nazwa.startswith("Under"):
            avg_rzecz = 1 - df_wyniki[kol_rzecz].mean()
        else:
            avg_rzecz = df_wyniki[kol_rzecz].mean()

        diff = avg_rzecz - avg_pred
        znak = ("✅" if abs(diff) < 0.03
                else "⚠️ " if abs(diff) < 0.07
                else "❌")

        print(f"  {nazwa:<12} {avg_pred:>14.3f} "
              f"{avg_rzecz:>12.3f} {diff:>+10.3f} {znak}")

    # ── 5. ROZKŁAD SUMY GOLI — model vs rzeczywistość ────────────────────────
    print("\n" + "=" * 60)
    print("5. ROZKŁAD SUMY GOLI — MODEL vs RZECZYWISTOŚĆ")
    print("=" * 60)

    print(f"\n  {'Suma goli':>10} {'Rzeczyw. %':>12} "
          f"{'Model % (avg)':>15}")
    print("  " + "-" * 42)

    for k in range(8):
        rzecz = (df_wyniki["suma_goli"] == k).mean()

        # model: prawdop. że suma = k
        # suma k = (0,k), (1,k-1), ..., (k,0) w macierzy
        # przybliżamy przez lambda_total
        # ale lepiej wziąć bezpośrednio z danych
        # (nie mamy zapisanej per-wynik macierzy)
        # więc używamy przybliżenia Poissona sumy
        from scipy.stats import poisson as _p
        avg_lt = df_wyniki["lambda_total"].mean()
        model_p = _p.pmf(k, avg_lt)

        print(f"  {k:>10} {rzecz:>12.1%} {model_p:>15.1%}")

    # ── 6. LOG-LOSS PER KOLEJKA ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. LOG-LOSS PER KOLEJKA")
    print("=" * 60)

    df_per_k = (df_wyniki
                .groupby("kolejka")
                .agg(n=("log_loss", "count"),
                     ll=("log_loss", "mean"),
                     acc=("czy_trafil", "mean"))
                .reset_index())

    print(f"\n  {'Kolejka':>8} {'Meczów':>8} "
          f"{'Log-loss':>10} {'Accuracy':>10}")
    print("  " + "-" * 44)

    for _, row in df_per_k.iterrows():
        mark = "✅" if row["ll"] < log_loss_sredni else "⚠️ "
        print(f"  {int(row['kolejka']):>8} {int(row['n']):>8} "
              f"{row['ll']:>10.4f} {row['acc']:>10.1%} {mark}")

    print(f"\n  Średni log-loss: {log_loss_sredni:.4f}")

    # ── 7. NAJTRUDNIEJSZE MECZE ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. MECZE Z NAJWIĘKSZYM BŁĘDEM (top 10)")
    print("=" * 60)

    for _, row in df_wyniki.nlargest(10, "log_loss").iterrows():
        print(f"\n  K{int(row['kolejka']):2d} | "
              f"{row['gospodarz']} vs {row['gosc']}  "
              f"{int(row['gole_gosp'])}:{int(row['gole_gosc'])} "
              f"({row['wynik_1x2']})")
        print(f"       Model: H={row['p_home']:.2f} "
              f"D={row['p_draw']:.2f} "
              f"A={row['p_away']:.2f} | "
              f"λ {row['lambda_home']:.2f}:{row['lambda_away']:.2f} | "
              f"ll={row['log_loss']:.4f}")


# =============================================================================
# GŁÓWNY PROGRAM
# =============================================================================

def main():
    print("=" * 60)
    print("BACKTESTING MODELU POISSONA — EKSTRAKLASA 2025/26")
    print("=" * 60)
    print()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    print(f"Wczytano {len(df)} meczów.\n")

    print("Trenowanie modelu kolejka po kolejce...\n")
    df_wyniki, bledy = run_backtesting(df)

    if bledy:
        print(f"\n⚠️  Błędy ({len(bledy)}):")
        for b in bledy:
            print(f"   {b}")

    analizuj_wyniki(df_wyniki)

    df_wyniki.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ Wyniki zapisane: {OUTPUT_PATH}")
    print("✅ Backtesting zakończony.")


if __name__ == "__main__":
    main()