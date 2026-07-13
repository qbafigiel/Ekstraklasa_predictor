"""
model_corners.py
================
Model predykcji rzutów rożnych (kornerów) w meczu Ekstraklasy.

Założenia:
- modelujemy osobno korniery gospodarzy i gości
- używamy Negative Binomial (overdispersion > Poisson)
- rolling features budowane kauzalnie dla wszystkich sezonów
- zapisujemy pełny rozkład sumy kornerów:
    p_sum_0 ... p_sum_20, p_sum_21_plus
- liczymy O/U dla linii:
    4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5

Schemat OOS:
- walidacja: 2024/25
- test:      2025/26
"""

import sqlite3
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

DB_PATH = Path("db/ekstraklasa.db")
OUTPUT_CSV = Path("data/processed/model_corners_oos_predictions.csv")
REPORT_PATH = Path("data/reports/model/model_corners_oos_report.txt")

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"

OU_LINES = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
MAX_CORNERS = 35
EXACT_SUM_MAX = 20  # zapisujemy p_sum_0 ... p_sum_20, reszta do p_sum_21_plus

SEASON_ORDER = {
    "2023/24": 1,
    "2024/25": 2,
    "2025/26": 3,
}


# =============================================================================
# HELPERS
# =============================================================================

def season_sort_key(season_str):
    return SEASON_ORDER[str(season_str)]


def ll_binary(p, actual):
    if actual == 1:
        return -np.log(max(float(p), 1e-12))
    return -np.log(max(1.0 - float(p), 1e-12))


def init_team_state():
    return {
        "home_for": [],
        "home_against": [],
        "away_for": [],
        "away_against": [],
        "all_for": [],
        "all_against": [],
    }


def safe_avg(values, fallback, window):
    vals = list(values)[-window:]
    if len(vals) == 0:
        return float(fallback)
    return float(np.mean(vals))


# =============================================================================
# DANE
# =============================================================================

def load_matches():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            match_id,
            sezon,
            kolejka,
            data_meczu,
            gospodarz,
            gosc,
            rozne_gosp,
            rozne_gosc
        FROM matches
        ORDER BY sezon, kolejka, match_id
    """, conn)
    conn.close()

    for col in ["rozne_gosp", "rozne_gosc", "kolejka"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["rozne_gosp", "rozne_gosc", "kolejka"]).copy()
    df["kolejka"] = df["kolejka"].astype(int)

    df = df.sort_values(
        by=["sezon", "kolejka", "match_id"],
        key=lambda s: s.map(season_sort_key) if s.name == "sezon" else s
    ).reset_index(drop=True)

    return df


# =============================================================================
# KAUZALNE ROLLING FEATURES DLA WSZYSTKICH SEZONÓW
# =============================================================================

def build_causal_corner_features(df_all, window=8):
    """
    Buduje rolling features mecz po meczu, bez leakage.
    Każdy mecz widzi tylko historię wcześniejszych meczów.

    Feature’y:
      - home_for_avg        : średnia kornerów gospodarza jako gospodarz
      - home_against_avg    : średnia kornerów oddawanych przez gospodarza jako gospodarz
      - away_for_avg        : średnia kornerów gościa jako gość
      - away_against_avg    : średnia kornerów oddawanych przez gościa jako gość
      - home_total_for_avg  : średnia wszystkich kornerów gospodarza (home+away)
      - away_total_for_avg  : średnia wszystkich kornerów gościa (home+away)
      - global_home_avg     : średnia kornerów gospodarzy w całej wcześniejszej próbie
      - global_away_avg     : średnia kornerów gości w całej wcześniejszej próbie
    """
    df = df_all.copy()

    team_state = {}
    global_home_hist = []
    global_away_hist = []

    rows = []

    fallback_home = 5.33
    fallback_away = 4.53

    for _, row in df.iterrows():
        home = row["gospodarz"]
        away = row["gosc"]

        if home not in team_state:
            team_state[home] = init_team_state()
        if away not in team_state:
            team_state[away] = init_team_state()

        hs = team_state[home]
        aw = team_state[away]

        g_home = float(np.mean(global_home_hist)) if len(global_home_hist) else fallback_home
        g_away = float(np.mean(global_away_hist)) if len(global_away_hist) else fallback_away

        feat_row = {
            "match_id": row["match_id"],
            "sezon": row["sezon"],
            "kolejka": int(row["kolejka"]),
            "gospodarz": home,
            "gosc": away,
            "rozne_gosp_rzecz": float(row["rozne_gosp"]),
            "rozne_gosc_rzecz": float(row["rozne_gosc"]),
            "suma_rzecz": float(row["rozne_gosp"] + row["rozne_gosc"]),
            "home_for_avg": safe_avg(hs["home_for"], g_home, window),
            "home_against_avg": safe_avg(hs["home_against"], g_away, window),
            "away_for_avg": safe_avg(aw["away_for"], g_away, window),
            "away_against_avg": safe_avg(aw["away_against"], g_home, window),
            "home_total_for_avg": safe_avg(hs["all_for"], g_home, window),
            "away_total_for_avg": safe_avg(aw["all_for"], g_away, window),
            "global_home_avg": g_home,
            "global_away_avg": g_away,
        }
        rows.append(feat_row)

        # update historii PO zbudowaniu feature'ów
        home_for = float(row["rozne_gosp"])
        home_against = float(row["rozne_gosc"])
        away_for = float(row["rozne_gosc"])
        away_against = float(row["rozne_gosp"])

        hs["home_for"].append(home_for)
        hs["home_against"].append(home_against)
        hs["all_for"].append(home_for)
        hs["all_against"].append(home_against)

        aw["away_for"].append(away_for)
        aw["away_against"].append(away_against)
        aw["all_for"].append(away_for)
        aw["all_against"].append(away_against)

        global_home_hist.append(home_for)
        global_away_hist.append(away_for)

    out = pd.DataFrame(rows)
    return out


# =============================================================================
# MODEL NEGATIVE BINOMIAL
# =============================================================================

def fit_negbin_model(df_train):
    """
    Model:
      log(mu_home) = b_home + w_for*log(home_for_avg+0.5)
                            + w_against*log(away_against_avg+0.5)
                            + w_total*log(home_total_for_avg+0.5)

      log(mu_away) = b_away + w_for*log(away_for_avg+0.5)
                            + w_against*log(home_against_avg+0.5)
                            + w_total*log(away_total_for_avg+0.5)

    alpha = overdispersion shared for home/away
    """
    df = df_train.copy().dropna(subset=[
        "home_for_avg",
        "home_against_avg",
        "away_for_avg",
        "away_against_avg",
        "home_total_for_avg",
        "away_total_for_avg",
        "rozne_gosp_rzecz",
        "rozne_gosc_rzecz",
    ])

    y_home = df["rozne_gosp_rzecz"].values.astype(int)
    y_away = df["rozne_gosc_rzecz"].values.astype(int)

    x_home_for = np.log(df["home_for_avg"].values + 0.5)
    x_home_total = np.log(df["home_total_for_avg"].values + 0.5)
    x_away_against = np.log(df["away_against_avg"].values + 0.5)

    x_away_for = np.log(df["away_for_avg"].values + 0.5)
    x_away_total = np.log(df["away_total_for_avg"].values + 0.5)
    x_home_against = np.log(df["home_against_avg"].values + 0.5)

    def nll(params):
        b_home, b_away, w_for, w_against, w_total, log_alpha = params

        alpha = np.exp(log_alpha)
        alpha = max(alpha, 1e-6)

        mu_home = np.exp(
            b_home
            + w_for * x_home_for
            + w_against * x_away_against
            + w_total * x_home_total
        )
        mu_away = np.exp(
            b_away
            + w_for * x_away_for
            + w_against * x_home_against
            + w_total * x_away_total
        )

        # parametrizacja scipy nbinom:
        # mean = r*(1-p)/p
        # alpha = 1/r
        # => r = 1/alpha, p = 1/(1+alpha*mu)
        r = 1.0 / alpha
        p_home = 1.0 / (1.0 + alpha * mu_home)
        p_away = 1.0 / (1.0 + alpha * mu_away)

        ll = np.sum(nbinom.logpmf(y_home, r, p_home))
        ll += np.sum(nbinom.logpmf(y_away, r, p_away))

        return -float(ll)

    x0 = np.array([
        1.45,           # b_home ~ exp(1.45)=4.26
        1.25,           # b_away ~ exp(1.25)=3.49
        0.25,           # w_for
        0.20,           # w_against
        0.15,           # w_total
        np.log(0.10),   # alpha
    ])

    bounds = [
        (0.0, 3.0),                 # b_home
        (0.0, 3.0),                 # b_away
        (-2.0, 2.0),                # w_for
        (-2.0, 2.0),                # w_against
        (-2.0, 2.0),                # w_total
        (np.log(1e-4), np.log(2.0)) # log_alpha
    ]

    result = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000}
    )

    b_home, b_away, w_for, w_against, w_total, log_alpha = result.x

    return {
        "b_home": float(b_home),
        "b_away": float(b_away),
        "w_for": float(w_for),
        "w_against": float(w_against),
        "w_total": float(w_total),
        "alpha": float(np.exp(log_alpha)),
        "success": bool(result.success),
        "fun": float(result.fun),
    }


def predict_corners(model, home_for, away_against, home_total_for, away_for, home_against, away_total_for):
    b_home = model["b_home"]
    b_away = model["b_away"]
    w_for = model["w_for"]
    w_against = model["w_against"]
    w_total = model["w_total"]
    alpha = model["alpha"]

    mu_home = np.exp(
        b_home
        + w_for * np.log(home_for + 0.5)
        + w_against * np.log(away_against + 0.5)
        + w_total * np.log(home_total_for + 0.5)
    )
    mu_away = np.exp(
        b_away
        + w_for * np.log(away_for + 0.5)
        + w_against * np.log(home_against + 0.5)
        + w_total * np.log(away_total_for + 0.5)
    )

    r = 1.0 / alpha
    p_h = 1.0 / (1.0 + alpha * mu_home)
    p_a = 1.0 / (1.0 + alpha * mu_away)

    p_home_vec = np.array([nbinom.pmf(k, r, p_h) for k in range(MAX_CORNERS + 1)])
    p_away_vec = np.array([nbinom.pmf(k, r, p_a) for k in range(MAX_CORNERS + 1)])

    p_sum = np.convolve(p_home_vec, p_away_vec)[:MAX_CORNERS + 1]
    total = p_sum.sum()
    if total <= 0:
        p_sum = np.zeros(MAX_CORNERS + 1)
        p_sum[0] = 1.0
    else:
        p_sum = p_sum / total

    out = {
        "lambda_home": float(mu_home),
        "lambda_away": float(mu_away),
        "lambda_sum": float(mu_home + mu_away),
        "alpha": float(alpha),
        "p_sum_vec": p_sum,
    }

    for line in OU_LINES:
        p_over = float(sum(p_sum[k] for k in range(len(p_sum)) if k > line))
        key = str(line).replace(".", "_")
        out[f"p_over_{key}"] = p_over
        out[f"p_under_{key}"] = 1.0 - p_over

    return out


# =============================================================================
# BACKTESTING OOS
# =============================================================================

def run_backtesting(df_features, target_season):
    season_order = SEASON_ORDER[target_season]
    df_target = df_features[df_features["sezon"] == target_season].copy()

    kolejki = sorted(df_target["kolejka"].unique())
    rows = []

    for k in kolejki:
        df_train = df_features[
            (df_features["sezon"].map(SEASON_ORDER) < season_order)
            | (
                (df_features["sezon"] == target_season)
                & (df_features["kolejka"] < k)
            )
        ].copy()

        if len(df_train) < 50:
            continue

        model = fit_negbin_model(df_train)

        df_round = df_target[df_target["kolejka"] == k]

        for _, mecz in df_round.iterrows():
            pred = predict_corners(
                model=model,
                home_for=mecz["home_for_avg"],
                away_against=mecz["away_against_avg"],
                home_total_for=mecz["home_total_for_avg"],
                away_for=mecz["away_for_avg"],
                home_against=mecz["home_against_avg"],
                away_total_for=mecz["away_total_for_avg"],
            )

            suma = int(mecz["suma_rzecz"])

            row = {
                "match_id": mecz["match_id"],
                "sezon": target_season,
                "kolejka": int(k),
                "gospodarz": mecz["gospodarz"],
                "gosc": mecz["gosc"],
                "rozne_gosp_rzecz": float(mecz["rozne_gosp_rzecz"]),
                "rozne_gosc_rzecz": float(mecz["rozne_gosc_rzecz"]),
                "suma_rzecz": float(mecz["suma_rzecz"]),
                "lambda_home": round(pred["lambda_home"], 4),
                "lambda_away": round(pred["lambda_away"], 4),
                "lambda_suma": round(pred["lambda_sum"], 4),
                "alpha": round(pred["alpha"], 6),
                "fit_success": int(model["success"]),
                "fit_fun": round(model["fun"], 4),
                "coef_b_home": round(model["b_home"], 4),
                "coef_b_away": round(model["b_away"], 4),
                "coef_w_for": round(model["w_for"], 4),
                "coef_w_against": round(model["w_against"], 4),
                "coef_w_total": round(model["w_total"], 4),
            }

            # exact distribution sumy
            p_sum = pred["p_sum_vec"]
            for exact_k in range(EXACT_SUM_MAX + 1):
                row[f"p_sum_{exact_k}"] = round(float(p_sum[exact_k]), 6)
            tail = float(sum(p_sum[EXACT_SUM_MAX + 1:]))
            row[f"p_sum_{EXACT_SUM_MAX + 1}_plus"] = round(tail, 6)

            # O/U
            for line in OU_LINES:
                key = str(line).replace(".", "_")
                p_over = pred[f"p_over_{key}"]
                actual = int(suma > line)

                row[f"p_over_{key}"] = round(p_over, 6)
                row[f"p_under_{key}"] = round(1.0 - p_over, 6)
                row[f"over_{key}_rzecz"] = actual
                row[f"ll_over_{key}"] = round(ll_binary(p_over, actual), 6)

            rows.append(row)

        print(
            f"  {target_season} K{k:02d} | "
            f"b_h={model['b_home']:.3f} b_a={model['b_away']:.3f} "
            f"w_for={model['w_for']:.3f} w_against={model['w_against']:.3f} "
            f"w_total={model['w_total']:.3f} alpha={model['alpha']:.3f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# REPORT
# =============================================================================

def format_ou_block(df, title):
    lines = [title]
    for line in OU_LINES:
        key = str(line).replace(".", "_")
        ll = df[f"ll_over_{key}"].mean()
        p_avg = df[f"p_over_{key}"].mean()
        r_avg = df[f"over_{key}_rzecz"].mean()
        delta = r_avg - p_avg
        status = "OK" if abs(delta) < 0.05 else "BIAS"
        lines.append(
            f"  Over {line:4.1f}: ll={ll:.4f} | model={p_avg:.3f} | rzecz={r_avg:.3f} | delta={delta:+.3f} {status}"
        )
    return lines


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuję matches...")
    df_all = load_matches()

    print("\n2. Buduję kauzalne rolling features dla wszystkich sezonów...")
    df_features = build_causal_corner_features(df_all, window=8)
    print(f"   Meczów z feature'ami: {len(df_features)}")

    print(f"\n3. Rolling backtesting {VAL_SEASON}...")
    df_val = run_backtesting(df_features, VAL_SEASON)

    print(f"\n4. Rolling backtesting {TEST_SEASON}...")
    df_test = run_backtesting(df_features, TEST_SEASON)

    if len(df_val) == 0 or len(df_test) == 0:
        raise RuntimeError("Brak wyników walidacji lub testu.")

    df_test.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    bench = -np.log(0.5)

    lines = []
    lines.append("=" * 78)
    lines.append("MODEL KORNERÓW — OOS RAPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. ARCHITEKTURA")
    lines.append("-" * 78)
    lines.append("  Negative Binomial")
    lines.append("  log(mu_home) = b_home + w_for*log(home_for_avg) + w_against*log(away_against_avg) + w_total*log(home_total_for_avg)")
    lines.append("  log(mu_away) = b_away + w_for*log(away_for_avg) + w_against*log(home_against_avg) + w_total*log(away_total_for_avg)")
    lines.append("  Pełny rozkład sumy zapisany jako p_sum_0 ... p_sum_20, p_sum_21_plus")
    lines.append(f"  Linie OU: {OU_LINES}")
    lines.append("  Rolling window: 8 meczów")
    lines.append("")
    lines.append("2. WALIDACJA 2024/25")
    lines.append("-" * 78)
    lines.append(f"  Meczów: {len(df_val)}")
    lines.extend(format_ou_block(df_val, "  Kalibracja O/U:"))
    lines.append("")
    lines.append("3. TEST OOS 2025/26")
    lines.append("-" * 78)
    lines.append(f"  Meczów: {len(df_test)}")
    lines.append(f"  Benchmark 50/50: {bench:.4f}")
    lines.extend(format_ou_block(df_test, "  OOS O/U:"))
    lines.append("")
    lines.append("4. ŚREDNIE PARAMETRY OOS")
    lines.append("-" * 78)
    lines.append(f"  lambda_home mean: {df_test['lambda_home'].mean():.2f} | rzecz: {df_test['rozne_gosp_rzecz'].mean():.2f}")
    lines.append(f"  lambda_away mean: {df_test['lambda_away'].mean():.2f} | rzecz: {df_test['rozne_gosc_rzecz'].mean():.2f}")
    lines.append(f"  lambda_suma mean: {df_test['lambda_suma'].mean():.2f} | rzecz: {df_test['suma_rzecz'].mean():.2f}")
    lines.append(f"  alpha mean:       {df_test['alpha'].mean():.4f}")
    lines.append(f"  coef b_home mean: {df_test['coef_b_home'].mean():.4f}")
    lines.append(f"  coef b_away mean: {df_test['coef_b_away'].mean():.4f}")
    lines.append(f"  coef w_for mean:  {df_test['coef_w_for'].mean():.4f}")
    lines.append(f"  coef w_against mean: {df_test['coef_w_against'].mean():.4f}")
    lines.append(f"  coef w_total mean:   {df_test['coef_w_total'].mean():.4f}")
    lines.append("")
    lines.append("5. PRZYKŁADOWE ROZKŁADY SUMY (pierwsze 5 meczów OOS)")
    lines.append("-" * 78)
    sample = df_test.head(5)
    for row in sample.itertuples(index=False):
        probs = []
        for k in range(5, 13):
            probs.append(f"{k}:{getattr(row, f'p_sum_{k}'):.3f}")
        lines.append(
            f"  K{row.kolejka:02d} | {row.gospodarz} vs {row.gosc} | suma_rzecz={row.suma_rzecz:.0f} | "
            + " ".join(probs)
        )
    lines.append("")
    lines.append("6. PLIKI")
    lines.append("-" * 78)
    lines.append(f"  {OUTPUT_CSV}")
    lines.append(f"  {REPORT_PATH}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {OUTPUT_CSV}")
    print(f"Zapisano: {REPORT_PATH}")


if __name__ == "__main__":
    main()