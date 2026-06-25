"""
backtesting_gole_v3.py
======================
Backtesting modelu Poissona dla goli — wersja z Dixon-Coles.

Zmiany vs v2:
  - Dixon-Coles korekta tau dla wyników 0:0 / 1:0 / 0:1 / 1:1
  - parametr rho estymowany przez MLE razem z alpha i beta
  - macierz wyników korygowana przed liczeniem rynków
  - nowe kolumny: rho_kolejka (wartość rho w danej kolejce)

Rynki:
  - 1X2, BTTS, Over/Under: 0.5 / 1.5 / 2.5 / 3.5

Źródło: db/ekstraklasa.db
Output: data/processed/backtesting_wyniki_v3.csv
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

DB_PATH     = Path("db/ekstraklasa.db")
OUTPUT_PATH = Path("data/processed/backtesting_wyniki_v3.csv")
SEZON_TEST  = "2025/26"
MAX_GOLE    = 10
K_PRIOR     = 10

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

BENIAMINKOWIE_2025_26 = [
    "Arka Gdynia",
    "Bruk-Bet Termalica Nieciecza",
    "Wisła Płock",
]

PRIOR_2025_26 = {
    "prior_atak":   0.7744,
    "prior_obrona": 1.0648,
}


# =============================================================================
# DIXON-COLES — współczynnik korekcyjny tau
# =============================================================================

def tau(i, j, lh, la, rho):
    """
    Współczynnik korekcyjny Dixon-Coles dla wyniku i:j.

    Koryguję tylko cztery wyniki gdzie Poisson się myli:
      0:0  — zbyt rzadko w czystym Poissonie
      1:0  — zbyt rzadko
      0:1  — zbyt rzadko
      1:1  — zbyt rzadko (remis z golami)

    Dla wszystkich innych wyników tau = 1.0 (brak korekty).

    Matematyka:
      tau(0,0) = 1 - lh * la * rho
      tau(1,0) = 1 + la * rho
      tau(0,1) = 1 + lh * rho
      tau(1,1) = 1 - rho
      tau(i,j) = 1  dla i+j >= 3

    rho > 0 → zwiększa 0:0 i 1:1, zmniejsza 1:0 i 0:1
    rho = 0 → czysty Poisson (brak korekty)

    Ograniczenie: tau musi być > 0 dla wszystkich komórek.
    Dla 0:0: 1 - lh*la*rho > 0  →  rho < 1/(lh*la)
    W praktyce rho jest małe (~0.05-0.15) więc rzadko narusza warunek.
    """
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    elif i == 1 and j == 0:
        return 1.0 + la * rho
    elif i == 0 and j == 1:
        return 1.0 + lh * rho
    elif i == 1 and j == 1:
        return 1.0 - rho
    else:
        return 1.0


# =============================================================================
# MODEL — MLE z Dixon-Coles
# =============================================================================

def przygotuj_dane(df_trening):
    druzyny = sorted(
        set(df_trening["gospodarz"].unique()) |
        set(df_trening["gosc"].unique())
    )
    n   = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    return {
        "druzyny":     druzyny,
        "n_druzyn":    n,
        "team_to_idx": t2i,
        "idx_to_team": i2t,
        "home_idx":    df_trening["gospodarz"].map(t2i).values,
        "away_idx":    df_trening["gosc"].map(t2i).values,
        "goals_home":  df_trening["gole_gosp"].values.astype(float),
        "goals_away":  df_trening["gole_gosc"].values.astype(float),
        "weights":     df_trening["waga_sezonu"].values.astype(float),
    }


def neg_log_likelihood_dc(theta, data):
    """
    Ujemna log-likelihood z korektą Dixon-Coles.

    Struktura theta:
      theta[0]         = log(mu_home)
      theta[1]         = log(mu_away)
      theta[2]         = rho  (NIE w log — może być ujemne, ale
                               w praktyce rho > 0, więc bounded)
      theta[3..N+1]    = log_alpha[0..N-2]  (N-1 wolnych parametrów)
      theta[N+2..2N]   = log_beta[0..N-2]   (N-1 wolnych parametrów)

    Constraint identyfikowalności (taki sam jak v1/v2):
      log_alpha[N-1] = -sum(log_alpha[0..N-2])
      log_beta[N-1]  = -sum(log_beta[0..N-2])

    Dixon-Coles dodaje jeden parametr (rho) do MLE.
    Reszta struktury bez zmian — łatwo porównać z v2.
    """
    N = data["n_druzyn"]

    mu_home = np.exp(theta[0])
    mu_away = np.exp(theta[1])
    rho     = theta[2]  # bezpośrednio, bez exp — może być blisko zera

    log_alpha = np.zeros(N)
    log_beta  = np.zeros(N)
    log_alpha[:N - 1] = theta[3:3 + N - 1]
    log_alpha[N - 1]  = -np.sum(log_alpha[:N - 1])
    log_beta[:N - 1]  = theta[3 + N - 1:3 + 2 * (N - 1)]
    log_beta[N - 1]   = -np.sum(log_beta[:N - 1])

    alpha = np.exp(log_alpha)
    beta  = np.exp(log_beta)

    ll = 0.0
    for k in range(len(data["goals_home"])):
        hi  = data["home_idx"][k]
        ai  = data["away_idx"][k]
        gh  = int(data["goals_home"][k])
        ga  = int(data["goals_away"][k])
        w   = data["weights"][k]

        lh = max(mu_home * alpha[hi] * beta[ai], 1e-10)
        la = max(mu_away * alpha[ai] * beta[hi], 1e-10)

        # log P_poisson osobno dla każdej drużyny
        log_p_h = gh * np.log(lh) - lh  # + stała (log gh!)
        log_p_a = ga * np.log(la) - la

        # korekta Dixon-Coles — tylko dla 0:0, 1:0, 0:1, 1:1
        t = tau(gh, ga, lh, la, rho)

        # tau musi być > 0 — jeśli nie, kara numeryczna
        if t <= 0:
            ll -= w * 1e6
            continue

        log_t = np.log(t)

        ll += w * (log_p_h + log_p_a + log_t)

    return -ll


def trenuj_model_dc(data):
    """
    MLE z Dixon-Coles.

    Inicjalizacja:
      - mu_home, mu_away: jak w v2
      - rho: 0.1 (typowa wartość startowa — lekko dodatnia)
      - alpha, beta: 0 (jak w v2)

    Bounds dla rho:
      Musi być < 1/(lh*la) dla każdego meczu żeby tau(0,0) > 0.
      W praktyce ograniczamy rho do (-0.5, 0.5).
      Ujemne rho byłoby niestandardowe (zmniejszałoby 0:0),
      ale pozwalamy optymalizatorowi to odkryć.
    """
    N      = data["n_druzyn"]
    # theta: [log_mu_h, log_mu_a, rho, log_alpha x (N-1), log_beta x (N-1)]
    theta0 = np.zeros(3 + 2 * (N - 1))
    theta0[0] = np.log(np.mean(data["goals_home"]))
    theta0[1] = np.log(np.mean(data["goals_away"]))
    theta0[2] = 0.1  # rho startowe

    # bounds: mu_h i mu_a muszą być > 0 (log bez bounds),
    # rho ograniczamy, alpha i beta bez bounds (log-space)
    bounds = (
        [(None, None),   # log mu_home
         (None, None),   # log mu_away
         (-0.5, 0.5)]    # rho
        + [(None, None)] * (2 * (N - 1))  # log_alpha, log_beta
    )

    result = minimize(
        neg_log_likelihood_dc, theta0, args=(data,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
    )
    return result.x


def ekstrahuj_parametry_dc(theta, data):
    N = data["n_druzyn"]

    mu_home = np.exp(theta[0])
    mu_away = np.exp(theta[1])
    rho     = theta[2]

    log_alpha = np.zeros(N)
    log_beta  = np.zeros(N)
    log_alpha[:N - 1] = theta[3:3 + N - 1]
    log_alpha[N - 1]  = -np.sum(log_alpha[:N - 1])
    log_beta[:N - 1]  = theta[3 + N - 1:3 + 2 * (N - 1)]
    log_beta[N - 1]   = -np.sum(log_beta[:N - 1])

    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "rho":     rho,
        "alpha": {
            data["idx_to_team"][i]: np.exp(log_alpha[i]) for i in range(N)
        },
        "beta": {
            data["idx_to_team"][i]: np.exp(log_beta[i]) for i in range(N)
        },
    }


# =============================================================================
# PRIOR BAYESOWSKI — identyczny z v2
# =============================================================================

def policz_mecze_beniaminkow(df_trening, beniaminkowie):
    df_biezacy = df_trening[df_trening["sezon"] == SEZON_TEST]
    wynik = {}
    for druzyna in beniaminkowie:
        n = int(
            ((df_biezacy["gospodarz"] == druzyna) |
             (df_biezacy["gosc"] == druzyna)).sum()
        )
        wynik[druzyna] = n
    return wynik


def zastosuj_prior_beniaminkow(params, beniaminkowie, prior, mecze_ben, K):
    prior_atak   = prior["prior_atak"]
    prior_obrona = prior["prior_obrona"]
    info = {}

    for druzyna in beniaminkowie:
        n = mecze_ben.get(druzyna, 0)

        if druzyna not in params["alpha"]:
            alpha_przed = None
            beta_przed  = None
            alpha_po    = prior_atak
            beta_po     = prior_obrona
        else:
            alpha_przed = params["alpha"][druzyna]
            beta_przed  = params["beta"][druzyna]
            alpha_po    = (K * prior_atak   + n * alpha_przed) / (K + n)
            beta_po     = (K * prior_obrona + n * beta_przed)  / (K + n)

        params["alpha"][druzyna] = alpha_po
        params["beta"][druzyna]  = beta_po

        waga_priora = K / (K + n)

        info[druzyna] = {
            "n_mecze":     n,
            "alpha_przed": round(alpha_przed, 4) if alpha_przed else None,
            "beta_przed":  round(beta_przed, 4)  if beta_przed  else None,
            "alpha_po":    round(alpha_po, 4),
            "beta_po":     round(beta_po, 4),
            "waga_priora": round(waga_priora, 4),
        }

    return params, info


# =============================================================================
# PREDYKCJA z Dixon-Coles
# =============================================================================

def przewiduj_mecz_dc(params, gospodarz, gosc):
    """
    Buduje macierz wyników z korektą Dixon-Coles.

    Kroki:
      1. Oblicz lambda_home i lambda_away (identycznie jak v2)
      2. Wypełnij macierz P(i,j) = Poisson(i,lh) * Poisson(j,la) * tau(i,j)
      3. Znormalizuj macierz (suma = 1)
         UWAGA: normalizacja jest konieczna bo tau zaburza sumy
      4. Z macierzy oblicz rynki identycznie jak v2
    """
    if gospodarz not in params["alpha"] or gosc not in params["alpha"]:
        return None

    lh  = (params["mu_home"]
           * params["alpha"][gospodarz]
           * params["beta"][gosc])
    la  = (params["mu_away"]
           * params["alpha"][gosc]
           * params["beta"][gospodarz])
    rho = params["rho"]

    # macierz z korektą DC
    macierz = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            p_poisson = poisson.pmf(i, lh) * poisson.pmf(j, la)
            t         = tau(i, j, lh, la, rho)
            # tau może być minimalnie ujemne przy złych parametrach
            # clip do 0 — po normalizacji i tak nie zaszkodzi
            macierz[i, j] = max(p_poisson * t, 0.0)

    # normalizacja — konieczna po korekcie DC
    total = macierz.sum()
    if total <= 0:
        return None
    macierz /= total

    # rynki — identyczne obliczenia jak v2
    p_home = float(np.sum(np.tril(macierz, -1)))
    p_draw = float(np.sum(np.diag(macierz)))
    p_away = float(np.sum(np.triu(macierz, 1)))
    p_btts = float(
        (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
    )

    p_under = {}
    for prog in [0, 1, 2, 3]:
        s = 0.0
        for i in range(MAX_GOLE):
            for j in range(MAX_GOLE):
                if i + j <= prog:
                    s += macierz[i, j]
        p_under[prog] = s

    return {
        "lambda_home": round(lh, 4),
        "lambda_away": round(la, 4),
        "rho":         round(rho, 4),
        "p_home":      p_home,
        "p_draw":      p_draw,
        "p_away":      p_away,
        "p_btts":      p_btts,
        "p_under_05":  p_under[0],
        "p_under_15":  p_under[1],
        "p_under_25":  p_under[2],
        "p_under_35":  p_under[3],
        "p_over_05":   1 - p_under[0],
        "p_over_15":   1 - p_under[1],
        "p_over_25":   1 - p_under[2],
        "p_over_35":   1 - p_under[3],
    }


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


def log_loss_meczu(p_home, p_draw, p_away, wynik):
    eps = 1e-10
    p   = {"H": p_home, "D": p_draw, "A": p_away}[wynik]
    return -np.log(max(p, eps))


# =============================================================================
# BACKTESTING GŁÓWNA PĘTLA
# =============================================================================

def run_backtesting(df):
    df_test     = df[df["sezon"] == SEZON_TEST].copy()
    df_historia = df[df["sezon"] != SEZON_TEST].copy()
    kolejki     = sorted(df_test["kolejka"].unique())

    print("=" * 60)
    print("BACKTESTING V3 — Dixon-Coles + prior bayesowski")
    print("=" * 60)
    print(f"Sezon testowy : {SEZON_TEST}")
    print(f"Kolejek       : {len(kolejki)}")
    print(f"Meczów testu  : {len(df_test)}")
    print(f"Danych hist.  : {len(df_historia)} meczów")
    print(f"Beniaminkowie : {', '.join(BENIAMINKOWIE_2025_26)}")
    print(f"Prior atak    : {PRIOR_2025_26['prior_atak']}")
    print(f"Prior obrona  : {PRIOR_2025_26['prior_obrona']}")
    print(f"K (siła prior): {K_PRIOR}")
    print(f"Dixon-Coles   : ✅ rho estymowane przez MLE")
    print("=" * 60)
    print()

    wyniki    = []
    bledy     = []
    rho_lista = []  # śledzenie rho przez kolejki

    for kolejka in kolejki:

        # dane treningowe
        df_poprzednie = df_test[df_test["kolejka"] < kolejka]
        df_trening    = pd.concat(
            [df_historia, df_poprzednie], ignore_index=True
        )

        mecze_ben = policz_mecze_beniaminkow(
            df_trening, BENIAMINKOWIE_2025_26
        )

        # MLE z Dixon-Coles
        data   = przygotuj_dane(df_trening)
        theta  = trenuj_model_dc(data)
        params = ekstrahuj_parametry_dc(theta, data)

        rho_biezace = params["rho"]
        rho_lista.append({"kolejka": kolejka, "rho": round(rho_biezace, 4)})

        # prior bayesowski (identyczny jak v2)
        params, prior_info = zastosuj_prior_beniaminkow(
            params,
            beniaminkowie=BENIAMINKOWIE_2025_26,
            prior=PRIOR_2025_26,
            mecze_ben=mecze_ben,
            K=K_PRIOR,
        )

        # predykcje
        df_kolejka = df_test[df_test["kolejka"] == kolejka]
        n_ok       = 0
        n_brak     = 0

        for _, mecz in df_kolejka.iterrows():
            gosp   = mecz["gospodarz"]
            gosc_d = mecz["gosc"]
            gh     = int(mecz["gole_gosp"])
            ga     = int(mecz["gole_gosc"])
            suma   = gh + ga

            pred = przewiduj_mecz_dc(params, gosp, gosc_d)

            if pred is None:
                n_brak += 1
                bledy.append(
                    f"K{kolejka}: NIEOCZEKIWANY brak parametrów "
                    f"dla {gosp} / {gosc_d}"
                )
                continue

            n_ok += 1

            wynik_rzecz = wynik_1x2(gh, ga)
            p_max       = max(
                pred["p_home"], pred["p_draw"], pred["p_away"]
            )
            wynik_pred = (
                "H" if p_max == pred["p_home"] else
                "D" if p_max == pred["p_draw"] else "A"
            )
            ll = log_loss_meczu(
                pred["p_home"], pred["p_draw"],
                pred["p_away"], wynik_rzecz
            )

            jest_beniaminek = (
                gosp   in BENIAMINKOWIE_2025_26 or
                gosc_d in BENIAMINKOWIE_2025_26
            )

            ben_druzyna = None
            n_mecze_ben = None
            waga_priora = None
            if gosp in prior_info:
                ben_druzyna = gosp
                n_mecze_ben = prior_info[gosp]["n_mecze"]
                waga_priora = prior_info[gosp]["waga_priora"]
            elif gosc_d in prior_info:
                ben_druzyna = gosc_d
                n_mecze_ben = prior_info[gosc_d]["n_mecze"]
                waga_priora = prior_info[gosc_d]["waga_priora"]

            wyniki.append({
                # identyfikacja
                "kolejka":          int(kolejka),
                "gospodarz":        gosp,
                "gosc":             gosc_d,
                "mecze_treningowe": len(df_trening),
                # wynik
                "gole_gosp":        gh,
                "gole_gosc":        ga,
                "suma_goli":        suma,
                "wynik_1x2":        wynik_rzecz,
                "btts_rzecz":       int(gh >= 1 and ga >= 1),
                "over05_rzecz":     int(suma > 0),
                "over15_rzecz":     int(suma > 1),
                "over25_rzecz":     int(suma > 2),
                "over35_rzecz":     int(suma > 3),
                # lambdy i rho
                "lambda_home":      pred["lambda_home"],
                "lambda_away":      pred["lambda_away"],
                "lambda_total":     round(
                    pred["lambda_home"] + pred["lambda_away"], 4
                ),
                "rho":              pred["rho"],
                # predykcja 1X2
                "p_home":           round(pred["p_home"], 4),
                "p_draw":           round(pred["p_draw"], 4),
                "p_away":           round(pred["p_away"], 4),
                "wynik_pred":       wynik_pred,
                "czy_trafil":       int(wynik_pred == wynik_rzecz),
                "log_loss":         round(ll, 4),
                # pozostałe rynki
                "p_btts":           round(pred["p_btts"], 4),
                "p_under_05":       round(pred["p_under_05"], 4),
                "p_under_15":       round(pred["p_under_15"], 4),
                "p_under_25":       round(pred["p_under_25"], 4),
                "p_under_35":       round(pred["p_under_35"], 4),
                "p_over_05":        round(pred["p_over_05"], 4),
                "p_over_15":        round(pred["p_over_15"], 4),
                "p_over_25":        round(pred["p_over_25"], 4),
                "p_over_35":        round(pred["p_over_35"], 4),
                # błędy lambda
                "blad_lambda_home": round(abs(pred["lambda_home"] - gh), 4),
                "blad_lambda_away": round(abs(pred["lambda_away"] - ga), 4),
                # diagnostyka prior (z v2)
                "jest_beniaminek":  int(jest_beniaminek),
                "ben_druzyna":      ben_druzyna,
                "n_mecze_ben":      n_mecze_ben,
                "waga_priora":      waga_priora,
            })

        print(f"  K{kolejka:02d}: {n_ok} predykcji | "
              f"rho={rho_biezace:.4f} | "
              f"{n_brak} brak")

    return pd.DataFrame(wyniki), bledy, rho_lista


# =============================================================================
# ANALIZA WYNIKÓW
# =============================================================================

def analizuj_wyniki(df_wyniki, rho_lista):
    print()
    print("=" * 60)
    print("ANALIZA WYNIKÓW — BACKTESTING V3")
    print("=" * 60)

    n_total = len(df_wyniki)
    print(f"\nPredykcji łącznie: {n_total}")

    # --- Log-loss globalny ---
    ll_mean = df_wyniki["log_loss"].mean()
    ll_rand = np.log(3)
    poprawa = ll_rand - ll_mean

    print(f"\n--- LOG-LOSS 1X2 ---")
    print(f"Log-loss modelu    : {ll_mean:.4f}")
    print(f"Log-loss losowy    : {ll_rand:.4f}")
    print(f"Poprawa vs losowy  : {poprawa:+.4f}  "
          f"({'✅ MODEL LEPSZY' if poprawa > 0 else '❌ MODEL GORSZY'})")

    # --- Porównanie wszystkich wersji ---
    print(f"\n--- PORÓWNANIE V1 / V2 / V3 ---")
    print(f"  V1 (czysty Poisson)          : 1.1558  ❌ gorszy niż losowy")
    print(f"  V2 (+prior beniaminków)      : 1.0807  ✅")
    print(f"  V3 (+Dixon-Coles)            : {ll_mean:.4f}  "
          f"{'✅' if ll_mean < 1.0807 else '⚠️ brak poprawy vs v2'}")

    # --- Log-loss pierwsze 5 kolejek ---
    print(f"\n--- LOG-LOSS PIERWSZE 5 KOLEJEK ---")
    for k in range(1, 6):
        df_k = df_wyniki[df_wyniki["kolejka"] == k]
        if len(df_k) == 0:
            continue
        ll_k = df_k["log_loss"].mean()
        print(f"  Kolejka {k}: {ll_k:.4f}  ({len(df_k)} meczów)")

    # --- Rho przez sezon ---
    print(f"\n--- WARTOŚĆ RHO PRZEZ SEZON ---")
    df_rho = pd.DataFrame(rho_lista)
    print(f"  rho min : {df_rho['rho'].min():.4f}")
    print(f"  rho max : {df_rho['rho'].max():.4f}")
    print(f"  rho mean: {df_rho['rho'].mean():.4f}")
    print(f"  rho K01 : {df_rho[df_rho['kolejka']==1]['rho'].values[0]:.4f}")
    print(f"  rho K34 : {df_rho[df_rho['kolejka']==34]['rho'].values[0]:.4f}")
    print(f"\n  Interpretacja:")
    rho_srednie = df_rho["rho"].mean()
    if rho_srednie > 0.05:
        print(f"  ✅ rho={rho_srednie:.3f} > 0 — DC zwiększa prawdop. "
              f"0:0 i 1:1 (remisy)")
    elif rho_srednie > 0:
        print(f"  ⚠️ rho={rho_srednie:.3f} — mała korekta DC")
    else:
        print(f"  ❌ rho={rho_srednie:.3f} < 0 — nieoczekiwane, "
              f"sprawdź dane")

    # --- Accuracy i rozkład typowań ---
    acc = df_wyniki["czy_trafil"].mean()
    print(f"\n--- ACCURACY 1X2 ---")
    print(f"  Accuracy modelu     : {acc:.1%}")
    print(f"  Benchmark 'zawsze H': "
          f"{(df_wyniki['wynik_1x2'] == 'H').mean():.1%}")
    print(f"  V2 accuracy         : 44.4%")

    print(f"\n--- ROZKŁAD TYPOWAŃ vs RZECZYWISTOŚĆ ---")
    for etykieta in ["H", "D", "A"]:
        n_pred  = (df_wyniki["wynik_pred"] == etykieta).sum()
        n_rzecz = (df_wyniki["wynik_1x2"] == etykieta).sum()
        delta   = n_pred - n_rzecz
        print(f"  {etykieta}: model typuje {n_pred:3d}x | "
              f"rzeczywistość {n_rzecz:3d}x | "
              f"delta {delta:+d}")

    # --- Kalibracja Over/Under ---
    print(f"\n--- KALIBRACJA OVER/UNDER ---")
    rynki_ou = [
        ("Over 0.5",  "p_over_05",  "over05_rzecz"),
        ("Over 1.5",  "p_over_15",  "over15_rzecz"),
        ("Over 2.5",  "p_over_25",  "over25_rzecz"),
        ("Over 3.5",  "p_over_35",  "over35_rzecz"),
    ]
    for nazwa, p_col, r_col in rynki_ou:
        p_avg = df_wyniki[p_col].mean()
        r_avg = df_wyniki[r_col].mean()
        delta = r_avg - p_avg
        ocena = "✅" if abs(delta) < 0.05 else "⚠️"
        print(f"  {nazwa:10s}: model={p_avg:.3f} | "
              f"rzecz={r_avg:.3f} | delta={delta:+.3f} {ocena}")

    # --- BTTS ---
    p_btts_avg = df_wyniki["p_btts"].mean()
    r_btts_avg = df_wyniki["btts_rzecz"].mean()
    delta_btts = r_btts_avg - p_btts_avg
    ocena_btts = "✅" if abs(delta_btts) < 0.05 else "⚠️"
    print(f"\n--- BTTS ---")
    print(f"  V2: model=0.519 | rzecz=0.575 | delta=+0.056 ⚠️")
    print(f"  V3: model={p_btts_avg:.3f} | "
          f"rzecz={r_btts_avg:.3f} | delta={delta_btts:+.3f} {ocena_btts}")

    # --- Log-loss z beniaminkiem vs bez ---
    print(f"\n--- LOG-LOSS: Z BENIAMINKIEM vs BEZ ---")
    df_ben   = df_wyniki[df_wyniki["jest_beniaminek"] == 1]
    df_noben = df_wyniki[df_wyniki["jest_beniaminek"] == 0]
    print(f"  Z beniaminkiem  : {df_ben['log_loss'].mean():.4f}  "
          f"({len(df_ben)} meczów)")
    print(f"  Bez beniaminka  : {df_noben['log_loss'].mean():.4f}  "
          f"({len(df_noben)} meczów)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    print(f"Wczytano {len(df)} meczów z bazy.")

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)
    brak_wagi = df["waga_sezonu"].isna().sum()
    if brak_wagi > 0:
        print(f"UWAGA: {brak_wagi} meczów bez wagi sezonu!")
        df["waga_sezonu"] = df["waga_sezonu"].fillna(0.4)

    df_wyniki, bledy, rho_lista = run_backtesting(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_wyniki.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nZapisano: {OUTPUT_PATH}  ({len(df_wyniki)} wierszy)")

    if bledy:
        print(f"\n⚠️  BŁĘDY ({len(bledy)}):")
        for b in bledy:
            print(f"  {b}")
    else:
        print("✅ Brak błędów — wszystkie mecze przewidziane.")

    analizuj_wyniki(df_wyniki, rho_lista)

    print("\n" + "=" * 60)
    print("GOTOWE.")
    print("=" * 60)


if __name__ == "__main__":
    main()