"""
backtesting_gole_v2.py
======================
Backtesting modelu Poissona dla goli — wersja z priorem bayesowskim.

Zmiany vs v1:
  - beniaminkowie przed K1 dostają prior (0.7744 / 1.0648) zamiast None
  - shrinkage K=10: po n meczach miesza MLE z priorem
  - wszystkie 306 meczów przewidywane (poprzednio 303)
  - nowe kolumny: n_mecze_ben, prior_zastosowany, waga_priora

Rynki:
  - 1X2, BTTS, Over/Under: 0.5 / 1.5 / 2.5 / 3.5

Źródło: db/ekstraklasa.db
Output: data/processed/backtesting_wyniki_v2.csv
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

DB_PATH      = Path("db/ekstraklasa.db")
OUTPUT_PATH  = Path("data/processed/backtesting_wyniki_v2.csv")
SEZON_TEST   = "2025/26"
MAX_GOLE     = 10
K_PRIOR      = 10  # siła priora = ekwiwalent 10 pseudo-meczów

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
}

# Beniaminkowie sezonu testowego
BENIAMINKOWIE_2025_26 = [
    "Arka Gdynia",
    "Bruk-Bet Termalica Nieciecza",
    "Wisła Płock",
]

# Prior empiryczny dla 2025/26 (liczony z debiutów 2024/25 bez data leakage)
PRIOR_2025_26 = {
    "prior_atak":   0.7744,
    "prior_obrona": 1.0648,
}


# =============================================================================
# MODEL — MLE
# (identyczne z v1, bez zmian)
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
        "druzyny":      druzyny,
        "n_druzyn":     n,
        "team_to_idx":  t2i,
        "idx_to_team":  i2t,
        "home_idx":     df_trening["gospodarz"].map(t2i).values,
        "away_idx":     df_trening["gosc"].map(t2i).values,
        "goals_home":   df_trening["gole_gosp"].values.astype(float),
        "goals_away":   df_trening["gole_gosc"].values.astype(float),
        "weights":      df_trening["waga_sezonu"].values.astype(float),
    }


def neg_log_likelihood(theta, data):
    N = data["n_druzyn"]

    log_alpha = np.zeros(N)
    log_beta  = np.zeros(N)
    log_alpha[:N - 1] = theta[2:2 + N - 1]
    log_alpha[N - 1]  = -np.sum(log_alpha[:N - 1])
    log_beta[:N - 1]  = theta[2 + N - 1:2 + 2 * (N - 1)]
    log_beta[N - 1]   = -np.sum(log_beta[:N - 1])

    mu_home = np.exp(theta[0])
    mu_away = np.exp(theta[1])
    alpha   = np.exp(log_alpha)
    beta    = np.exp(log_beta)

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
    N      = data["n_druzyn"]
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
    log_beta  = np.zeros(N)
    log_alpha[:N - 1] = theta[2:2 + N - 1]
    log_alpha[N - 1]  = -np.sum(log_alpha[:N - 1])
    log_beta[:N - 1]  = theta[2 + N - 1:2 + 2 * (N - 1)]
    log_beta[N - 1]   = -np.sum(log_beta[:N - 1])

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


# =============================================================================
# PRIOR BAYESOWSKI — NOWE W V2
# =============================================================================

def policz_mecze_beniaminkow(df_trening, beniaminkowie):
    """
    Ile meczów rozegrał każdy beniaminek w bieżącym sezonie testowym.

    WAŻNE: liczymy tylko mecze z sezonu 2025/26 obecne w df_trening.
    Historia (2023/24, 2024/25) nie wlicza się do n w shrinkage —
    beniaminkowie nie grali wtedy w Ekstraklasie, więc te dane nie istnieją.
    df_trening zawiera historię + poprzednie kolejki 2025/26,
    więc filtrujemy po sezonie.
    """
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
    """
    Post-MLE shrinkage dla beniaminków.

    Formuła:
        alpha_final = (K * prior_atak   + n * alpha_mle) / (K + n)
        beta_final  = (K * prior_obrona + n * beta_mle)  / (K + n)

    Gdy n=0 (brak jakichkolwiek danych, kolejka 1):
        alpha_final = prior_atak
        beta_final  = prior_obrona

    Gdy n=10 (K=10):
        50% prior + 50% MLE

    Params jest modyfikowany in-place i zwracany.
    Zwraca też słownik info dla logowania.
    """
    prior_atak   = prior["prior_atak"]
    prior_obrona = prior["prior_obrona"]
    info = {}

    for druzyna in beniaminkowie:
        n = mecze_ben.get(druzyna, 0)

        if druzyna not in params["alpha"]:
            # Brak danych z MLE — drużyna nie pojawiła się w df_trening
            # Może się zdarzyć w kolejce 1 jeśli MLE w ogóle jej nie widzi.
            # Dajemy czysty prior.
            alpha_przed = None
            beta_przed  = None
            alpha_po    = prior_atak
            beta_po     = prior_obrona
        else:
            alpha_przed = params["alpha"][druzyna]
            beta_przed  = params["beta"][druzyna]

            # shrinkage
            alpha_po = (K * prior_atak   + n * alpha_przed) / (K + n)
            beta_po  = (K * prior_obrona + n * beta_przed)  / (K + n)

        params["alpha"][druzyna] = alpha_po
        params["beta"][druzyna]  = beta_po

        waga_priora = K / (K + n)  # udział priora po shrinkage

        info[druzyna] = {
            "n_mecze":      n,
            "alpha_przed":  round(alpha_przed, 4) if alpha_przed else None,
            "beta_przed":   round(beta_przed, 4)  if beta_przed  else None,
            "alpha_po":     round(alpha_po, 4),
            "beta_po":      round(beta_po, 4),
            "waga_priora":  round(waga_priora, 4),
        }

    return params, info


# =============================================================================
# PREDYKCJA — identyczna z v1
# =============================================================================

def przewiduj_mecz(params, gospodarz, gosc):
    """
    Teraz nigdy nie zwraca None dla beniaminków —
    prior gwarantuje że zawsze mają parametry.
    Pozostałe drużyny (spoza modelu) nadal mogą zwrócić None,
    ale w tym backtestingu nie ma takich przypadków.
    """
    if gospodarz not in params["alpha"] or gosc not in params["alpha"]:
        return None

    lh = (params["mu_home"]
          * params["alpha"][gospodarz]
          * params["beta"][gosc])
    la = (params["mu_away"]
          * params["alpha"][gosc]
          * params["beta"][gospodarz])

    macierz = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            macierz[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    macierz /= macierz.sum()

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
    print(f"BACKTESTING V2 — prior bayesowski dla beniaminków")
    print("=" * 60)
    print(f"Sezon testowy : {SEZON_TEST}")
    print(f"Kolejek       : {len(kolejki)}")
    print(f"Meczów testu  : {len(df_test)}")
    print(f"Danych hist.  : {len(df_historia)} meczów")
    print(f"Beniaminkowie : {', '.join(BENIAMINKOWIE_2025_26)}")
    print(f"Prior atak    : {PRIOR_2025_26['prior_atak']}")
    print(f"Prior obrona  : {PRIOR_2025_26['prior_obrona']}")
    print(f"K (siła prior): {K_PRIOR}")
    print("=" * 60)
    print()

    wyniki = []
    bledy  = []

    for kolejka in kolejki:

        # --- dane treningowe ---
        df_poprzednie = df_test[df_test["kolejka"] < kolejka]
        df_trening    = pd.concat(
            [df_historia, df_poprzednie], ignore_index=True
        )

        # --- liczba meczów beniaminków w bieżącym sezonie ---
        mecze_ben = policz_mecze_beniaminkow(df_trening, BENIAMINKOWIE_2025_26)

        # --- MLE ---
        # Uwaga: w kolejce 1 df_poprzednie jest puste,
        # więc beniaminkowie NIE istnieją w danych treningowych.
        # MLE ich nie uwzględnia → nie mają parametrów alpha/beta.
        # prior_info obsłuży ten przypadek: alpha_przed = None → czysty prior.
        data   = przygotuj_dane(df_trening)
        theta  = trenuj_model(data)
        params = ekstrahuj_parametry(theta, data)

        # --- PRIOR BAYESOWSKI (nowość v2) ---
        params, prior_info = zastosuj_prior_beniaminkow(
            params,
            beniaminkowie=BENIAMINKOWIE_2025_26,
            prior=PRIOR_2025_26,
            mecze_ben=mecze_ben,
            K=K_PRIOR,
        )

        # --- predykcje meczów kolejki ---
        df_kolejka = df_test[df_test["kolejka"] == kolejka]
        n_ok       = 0
        n_brak     = 0

        for _, mecz in df_kolejka.iterrows():
            gosp   = mecz["gospodarz"]
            gosc_d = mecz["gosc"]
            gh     = int(mecz["gole_gosp"])
            ga     = int(mecz["gole_gosc"])
            suma   = gh + ga

            pred = przewiduj_mecz(params, gosp, gosc_d)

            if pred is None:
                # Nie powinno się zdarzyć dla żadnej drużyny 2025/26,
                # ale zostawiamy obsługę błędu dla bezpieczeństwa.
                n_brak += 1
                bledy.append(
                    f"K{kolejka}: NIEOCZEKIWANY brak parametrów dla "
                    f"{gosp} / {gosc_d}"
                )
                continue

            n_ok += 1

            wynik_rzecz = wynik_1x2(gh, ga)
            p_max       = max(pred["p_home"], pred["p_draw"], pred["p_away"])
            wynik_pred  = (
                "H" if p_max == pred["p_home"] else
                "D" if p_max == pred["p_draw"] else "A"
            )
            ll = log_loss_meczu(
                pred["p_home"], pred["p_draw"],
                pred["p_away"], wynik_rzecz
            )

            # czy mecz dotyczył beniaminka?
            jest_beniaminek = (
                gosp in BENIAMINKOWIE_2025_26 or
                gosc_d in BENIAMINKOWIE_2025_26
            )

            # info o priory dla drużyny (beniaminka)
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
                "kolejka":           int(kolejka),
                "gospodarz":         gosp,
                "gosc":              gosc_d,
                "mecze_treningowe":  len(df_trening),
                # wynik
                "gole_gosp":         gh,
                "gole_gosc":         ga,
                "suma_goli":         suma,
                "wynik_1x2":         wynik_rzecz,
                "btts_rzecz":        int(gh >= 1 and ga >= 1),
                "over05_rzecz":      int(suma > 0),
                "over15_rzecz":      int(suma > 1),
                "over25_rzecz":      int(suma > 2),
                "over35_rzecz":      int(suma > 3),
                # lambdy
                "lambda_home":       pred["lambda_home"],
                "lambda_away":       pred["lambda_away"],
                "lambda_total":      round(
                    pred["lambda_home"] + pred["lambda_away"], 4
                ),
                # predykcja 1X2
                "p_home":            round(pred["p_home"], 4),
                "p_draw":            round(pred["p_draw"], 4),
                "p_away":            round(pred["p_away"], 4),
                "wynik_pred":        wynik_pred,
                "czy_trafil":        int(wynik_pred == wynik_rzecz),
                "log_loss":          round(ll, 4),
                # pozostałe rynki
                "p_btts":            round(pred["p_btts"], 4),
                "p_under_05":        round(pred["p_under_05"], 4),
                "p_under_15":        round(pred["p_under_15"], 4),
                "p_under_25":        round(pred["p_under_25"], 4),
                "p_under_35":        round(pred["p_under_35"], 4),
                "p_over_05":         round(pred["p_over_05"], 4),
                "p_over_15":         round(pred["p_over_15"], 4),
                "p_over_25":         round(pred["p_over_25"], 4),
                "p_over_35":         round(pred["p_over_35"], 4),
                # błędy lambda
                "blad_lambda_home":  round(abs(pred["lambda_home"] - gh), 4),
                "blad_lambda_away":  round(abs(pred["lambda_away"] - ga), 4),
                # kolumny diagnostyczne (NOWE W V2)
                "jest_beniaminek":   int(jest_beniaminek),
                "ben_druzyna":       ben_druzyna,
                "n_mecze_ben":       n_mecze_ben,
                "waga_priora":       waga_priora,
            })

        # log per kolejka z info o priory
        ben_status = "  ".join([
            f"{d}(n={mecze_ben.get(d, 0)})" for d in BENIAMINKOWIE_2025_26
        ])
        print(f"  K{kolejka:02d}: {n_ok} predykcji | "
              f"{n_brak} brak | {ben_status}")

    return pd.DataFrame(wyniki), bledy


# =============================================================================
# ANALIZA WYNIKÓW
# =============================================================================

def analizuj_wyniki(df_wyniki):
    print()
    print("=" * 60)
    print("ANALIZA WYNIKÓW — BACKTESTING V2")
    print("=" * 60)

    n_total = len(df_wyniki)
    print(f"\nPredykcji łącznie: {n_total}")

    # --- Log-loss globalny ---
    ll_mean = df_wyniki["log_loss"].mean()
    ll_rand = np.log(3)  # -log(1/3) = 1.0986
    poprawa = ll_rand - ll_mean

    print(f"\n--- LOG-LOSS 1X2 ---")
    print(f"Log-loss modelu    : {ll_mean:.4f}")
    print(f"Log-loss losowy    : {ll_rand:.4f}  (benchmark: zawsze 1/3)")
    print(f"Poprawa vs losowy  : {poprawa:+.4f}  "
          f"({'✅ MODEL LEPSZY' if poprawa > 0 else '❌ MODEL GORSZY'})")

    # --- Log-loss per kolejka (pierwsze 5 kolejek) ---
    print(f"\n--- LOG-LOSS PIERWSZE 5 KOLEJEK ---")
    for k in range(1, 6):
        df_k = df_wyniki[df_wyniki["kolejka"] == k]
        if len(df_k) == 0:
            continue
        ll_k = df_k["log_loss"].mean()
        flag = "  ← anomalia K2!" if k == 2 and ll_k > 2.0 else ""
        print(f"  Kolejka {k}: {ll_k:.4f}  ({len(df_k)} meczów){flag}")

    # --- Log-loss z beniaminkiem vs bez ---
    print(f"\n--- LOG-LOSS: Z BENIAMINKIEM vs BEZ ---")
    df_ben   = df_wyniki[df_wyniki["jest_beniaminek"] == 1]
    df_noben = df_wyniki[df_wyniki["jest_beniaminek"] == 0]
    print(f"  Z beniaminkiem  : {df_ben['log_loss'].mean():.4f}  "
          f"({len(df_ben)} meczów)")
    print(f"  Bez beniaminka  : {df_noben['log_loss'].mean():.4f}  "
          f"({len(df_noben)} meczów)")

    # --- Ewolucja wagi priora ---
    print(f"\n--- EWOLUCJA WAGI PRIORA (beniaminkowie) ---")
    df_b = df_wyniki[df_wyniki["ben_druzyna"].notna()].copy()
    if len(df_b) > 0:
        tab = (df_b.groupby(["ben_druzyna", "kolejka"])["waga_priora"]
               .first().reset_index())
        for druzyna in BENIAMINKOWIE_2025_26:
            wiersze = tab[tab["ben_druzyna"] == druzyna]
            if len(wiersze) == 0:
                continue
            print(f"\n  {druzyna}:")
            for _, r in wiersze.iterrows():
                bar = "█" * int(r["waga_priora"] * 20)
                print(f"    K{int(r['kolejka']):02d}: "
                      f"prior={r['waga_priora']:.2f}  {bar}")

    # --- Accuracy 1X2 ---
    acc = df_wyniki["czy_trafil"].mean()
    print(f"\n--- ACCURACY 1X2 ---")
    print(f"  Accuracy modelu     : {acc:.1%}")
    print(f"  Benchmark 'zawsze H': "
          f"{(df_wyniki['wynik_1x2'] == 'H').mean():.1%}")

    # --- Rozkład typowań vs rzeczywistość ---
    print(f"\n--- ROZKŁAD TYPOWAŃ vs RZECZYWISTOŚĆ ---")
    for etykieta in ["H", "D", "A"]:
        n_pred  = (df_wyniki["wynik_pred"] == etykieta).sum()
        n_rzecz = (df_wyniki["wynik_1x2"] == etykieta).sum()
        print(f"  {etykieta}: model typuje {n_pred:3d}x | "
              f"rzeczywistość {n_rzecz:3d}x")

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
    print(f"  BTTS: model={p_btts_avg:.3f} | "
          f"rzecz={r_btts_avg:.3f} | delta={delta_btts:+.3f} {ocena_btts}")

    # --- Porównanie z v1 ---
    print(f"\n--- PORÓWNANIE V1 vs V2 ---")
    print(f"  V1 log-loss : 1.1558  (model gorszy niż losowy)")
    print(f"  V2 log-loss : {ll_mean:.4f}  "
          f"({'✅ poprawa' if ll_mean < 1.1558 else '❌ brak poprawy'})")
    print(f"  V1 predykcji: 303  (3 pominięte)")
    print(f"  V2 predykcji: {n_total}  "
          f"({'✅ komplet' if n_total == 306 else f'⚠️ brak {306 - n_total}'})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # --- wczytaj dane ---
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()

    print(f"Wczytano {len(df)} meczów z bazy.")

    # --- dodaj wagę sezonu ---
    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW)
    brak_wagi = df["waga_sezonu"].isna().sum()
    if brak_wagi > 0:
        print(f"UWAGA: {brak_wagi} meczów bez przypisanej wagi sezonu!")
        df["waga_sezonu"] = df["waga_sezonu"].fillna(0.4)

    # --- backtesting ---
    df_wyniki, bledy = run_backtesting(df)

    # --- zapisz CSV ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_wyniki.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nZapisano: {OUTPUT_PATH}  ({len(df_wyniki)} wierszy)")

    # --- błędy (nie powinno być żadnych) ---
    if bledy:
        print(f"\n⚠️  BŁĘDY ({len(bledy)}):")
        for b in bledy:
            print(f"  {b}")
    else:
        print("✅ Brak błędów — wszystkie mecze przewidziane.")

    # --- analiza ---
    analizuj_wyniki(df_wyniki)

    print("\n" + "=" * 60)
    print("GOTOWE.")
    print("=" * 60)


if __name__ == "__main__":
    main()