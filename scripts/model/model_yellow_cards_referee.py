"""
model_yellow_cards_referee.py
=============================
Challenger dla modelu ZOLTYCH KARTEK w meczu Ekstraklasy.

Cel:
- zachowac ta sama metodologie co oficjalny baseline model_yellow_cards.py
- dolozyc tylko aspekt sedziego
- porownac baseline vs referee w identycznym rolling OOS

Rynki:
- pelny rozklad sumy zoltych kartek w meczu
- Over/Under dla linii:
    0.5, 1.5, 2.5, ..., 7.5

Architektura:
- Poisson
- osobno modelujemy zolte kartki gospodarzy i gosci
- rolling features kauzalne dla wszystkich sezonow
- pelny rozklad sumy:
    p_sum_0 ... p_sum_12, p_sum_13_plus

Schemat OOS:
- walidacja: 2024/25
- test:      2025/26

Jedyna zmiana wzgledem baseline:
- dodatkowy wspolczynnik w_ref
- dodatkowy feature:
    ref_yc_log_ratio = log(ref_yc_shrunk_before + 0.5) - log(league_yc_avg_before + 0.5)

Feature refereego dziala symetrycznie na obie strony:
- log(mu_home) ... + w_ref * ref_yc_log_ratio
- log(mu_away) ... + w_ref * ref_yc_log_ratio
"""

import sqlite3
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "ekstraklasa.db"
REFEREE_FEATURES_PATH = ROOT / "data" / "processed" / "referee_causal_features.csv"

OUTPUT_CSV = ROOT / "data" / "processed" / "model_yellow_cards_referee_oos_predictions.csv"
REPORT_PATH = ROOT / "data" / "reports" / "model" / "model_yellow_cards_referee_oos_report.txt"

VAL_SEASON = "2024/25"
TEST_SEASON = "2025/26"

OU_LINES = [x + 0.5 for x in range(0, 8)]   # 0.5 ... 7.5
MAX_YC = 18
EXACT_SUM_MAX = 12

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


def mean_ou_logloss(df_pred):
    vals = []
    for line in OU_LINES:
        key = str(line).replace(".", "_")
        vals.append(float(df_pred[f"ll_over_{key}"].mean()))
    return float(np.mean(vals))


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
            f"  Over {line:3.1f}: ll={ll:.4f} | model={p_avg:.3f} | rzecz={r_avg:.3f} | delta={delta:+.3f} {status}"
        )
    return lines


def format_compare_block(df_base, df_ref, title):
    lines = [title]
    lines.append("  line  | baseline_ll | referee_ll | diff(ref-base) | base_bias | ref_bias")
    lines.append("  " + "-" * 74)
    for line in OU_LINES:
        key = str(line).replace(".", "_")
        ll_base = float(df_base[f"ll_over_{key}"].mean())
        ll_ref = float(df_ref[f"ll_over_{key}"].mean())

        p_base = float(df_base[f"p_over_{key}"].mean())
        r_base = float(df_base[f"over_{key}_rzecz"].mean())
        bias_base = r_base - p_base

        p_ref = float(df_ref[f"p_over_{key}"].mean())
        r_ref = float(df_ref[f"over_{key}_rzecz"].mean())
        bias_ref = r_ref - p_ref

        lines.append(
            f"  {line:4.1f} | "
            f"{ll_base:11.4f} | "
            f"{ll_ref:10.4f} | "
            f"{(ll_ref - ll_base):+14.4f} | "
            f"{bias_base:+9.3f} | "
            f"{bias_ref:+8.3f}"
        )
    return lines


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
            zk_gosp,
            zk_gosc
        FROM matches
        ORDER BY sezon, kolejka, match_id
    """, conn)
    conn.close()

    for col in ["zk_gosp", "zk_gosc", "kolejka"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["zk_gosp", "zk_gosc", "kolejka"]).copy()
    df["kolejka"] = df["kolejka"].astype(int)

    df = df.sort_values(
        by=["sezon", "kolejka", "match_id"],
        key=lambda s: s.map(season_sort_key) if s.name == "sezon" else s
    ).reset_index(drop=True)

    return df


def load_referee_features():
    if not REFEREE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Brak pliku {REFEREE_FEATURES_PATH}. "
            f"Najpierw uruchom scripts/model/build_referee_profiles.py"
        )

    df = pd.read_csv(REFEREE_FEATURES_PATH)

    required = {
        "match_id",
        "referee_name",
        "referee_full_name",
        "ref_matches_before",
        "ref_reliability_k10",
        "league_yc_avg_before",
        "ref_yc_shrunk_before",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Brakuje kolumn w referee_causal_features.csv: " + ", ".join(missing)
        )

    df = df[
        [
            "match_id",
            "referee_name",
            "referee_full_name",
            "ref_matches_before",
            "ref_reliability_k10",
            "league_yc_avg_before",
            "ref_yc_shrunk_before",
        ]
    ].copy()

    return df


# =============================================================================
# KAUZALNE ROLLING FEATURES
# =============================================================================

def build_causal_yc_features(df_all, window=8):
    """
    Buduje rolling features mecz po meczu, bez leakage.
    Ta sama logika co w baseline model_yellow_cards.py
    """
    df = df_all.copy()

    team_state = {}
    global_home_hist = []
    global_away_hist = []

    rows = []

    fallback_home = 1.99
    fallback_away = 2.19

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
            "zk_gosp_rzecz": float(row["zk_gosp"]),
            "zk_gosc_rzecz": float(row["zk_gosc"]),
            "suma_rzecz": float(row["zk_gosp"] + row["zk_gosc"]),
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

        home_for = float(row["zk_gosp"])
        home_against = float(row["zk_gosc"])
        away_for = float(row["zk_gosc"])
        away_against = float(row["zk_gosp"])

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

    return pd.DataFrame(rows)


def merge_referee_features(df_features, df_ref):
    df = df_features.merge(df_ref, on="match_id", how="left", validate="one_to_one")

    if df["referee_full_name"].isna().any():
        missing_ids = df.loc[df["referee_full_name"].isna(), "match_id"].tolist()[:10]
        raise RuntimeError(
            "Brak joinu referee features dla czesci meczow. "
            f"Przykladowe match_id: {missing_ids}"
        )

    df["league_yc_avg_before"] = pd.to_numeric(df["league_yc_avg_before"], errors="coerce")
    df["ref_yc_shrunk_before"] = pd.to_numeric(df["ref_yc_shrunk_before"], errors="coerce")
    df["ref_matches_before"] = pd.to_numeric(df["ref_matches_before"], errors="coerce")
    df["ref_reliability_k10"] = pd.to_numeric(df["ref_reliability_k10"], errors="coerce")

    if df["league_yc_avg_before"].isna().any() or df["ref_yc_shrunk_before"].isna().any():
        raise RuntimeError("Po merge referee features sa NaN w kolumnach yc avg before.")

    df["ref_yc_log_ratio"] = (
        np.log(df["ref_yc_shrunk_before"].astype(float) + 0.5)
        - np.log(df["league_yc_avg_before"].astype(float) + 0.5)
    )

    df["ref_yc_delta_raw"] = df["ref_yc_shrunk_before"] - df["league_yc_avg_before"]

    return df


# =============================================================================
# MODEL POISSON
# =============================================================================

def fit_poisson_model(df_train, use_referee=False):
    """
    Baseline:
    log(mu_home) = b_home
                 + w_for     * log(home_for_avg)
                 + w_against * log(away_against_avg)
                 + w_total   * log(home_total_for_avg)

    log(mu_away) = b_away
                 + w_for     * log(away_for_avg)
                 + w_against * log(home_against_avg)
                 + w_total   * log(away_total_for_avg)

    Challenger referee:
    obie strony dostaja dodatkowo:
                 + w_ref * ref_yc_log_ratio
    """
    req_cols = [
        "home_for_avg",
        "home_against_avg",
        "away_for_avg",
        "away_against_avg",
        "home_total_for_avg",
        "away_total_for_avg",
        "zk_gosp_rzecz",
        "zk_gosc_rzecz",
    ]
    if use_referee:
        req_cols.append("ref_yc_log_ratio")

    df = df_train.copy().dropna(subset=req_cols)

    y_home = df["zk_gosp_rzecz"].values.astype(float)
    y_away = df["zk_gosc_rzecz"].values.astype(float)

    x_home_for = np.log(df["home_for_avg"].values + 0.5)
    x_home_total = np.log(df["home_total_for_avg"].values + 0.5)
    x_away_against = np.log(df["away_against_avg"].values + 0.5)

    x_away_for = np.log(df["away_for_avg"].values + 0.5)
    x_away_total = np.log(df["away_total_for_avg"].values + 0.5)
    x_home_against = np.log(df["home_against_avg"].values + 0.5)

    if use_referee:
        x_ref = np.clip(df["ref_yc_log_ratio"].values.astype(float), -1.0, 1.0)

        def nll(params):
            b_home, b_away, w_for, w_against, w_total, w_ref = params

            mu_home = np.exp(
                b_home
                + w_for * x_home_for
                + w_against * x_away_against
                + w_total * x_home_total
                + w_ref * x_ref
            )
            mu_away = np.exp(
                b_away
                + w_for * x_away_for
                + w_against * x_home_against
                + w_total * x_away_total
                + w_ref * x_ref
            )

            mu_home = np.maximum(mu_home, 1e-8)
            mu_away = np.maximum(mu_away, 1e-8)

            ll = np.sum(y_home * np.log(mu_home) - mu_home)
            ll += np.sum(y_away * np.log(mu_away) - mu_away)

            return -float(ll)

        x0 = np.array([0.30, 0.35, 0.10, 0.15, 0.08, 0.05])

        bounds = [
            (-3.0, 3.0),
            (-3.0, 3.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
        ]

        result = minimize(
            nll, x0, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 5000}
        )

        b_home, b_away, w_for, w_against, w_total, w_ref = result.x

        return {
            "b_home": float(b_home),
            "b_away": float(b_away),
            "w_for": float(w_for),
            "w_against": float(w_against),
            "w_total": float(w_total),
            "w_ref": float(w_ref),
            "success": bool(result.success),
            "fun": float(result.fun),
            "use_referee": True,
        }

    else:
        def nll(params):
            b_home, b_away, w_for, w_against, w_total = params

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

            mu_home = np.maximum(mu_home, 1e-8)
            mu_away = np.maximum(mu_away, 1e-8)

            ll = np.sum(y_home * np.log(mu_home) - mu_home)
            ll += np.sum(y_away * np.log(mu_away) - mu_away)

            return -float(ll)

        x0 = np.array([0.30, 0.35, 0.10, 0.15, 0.08])

        bounds = [
            (-3.0, 3.0),
            (-3.0, 3.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
        ]

        result = minimize(
            nll, x0, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 5000}
        )

        b_home, b_away, w_for, w_against, w_total = result.x

        return {
            "b_home": float(b_home),
            "b_away": float(b_away),
            "w_for": float(w_for),
            "w_against": float(w_against),
            "w_total": float(w_total),
            "w_ref": 0.0,
            "success": bool(result.success),
            "fun": float(result.fun),
            "use_referee": False,
        }


def predict_yc(
    model,
    home_for,
    away_against,
    home_total_for,
    away_for,
    home_against,
    away_total_for,
    ref_yc_log_ratio=0.0,
):
    b_home = model["b_home"]
    b_away = model["b_away"]
    w_for = model["w_for"]
    w_against = model["w_against"]
    w_total = model["w_total"]
    w_ref = model.get("w_ref", 0.0)

    x_ref = float(np.clip(ref_yc_log_ratio, -1.0, 1.0)) if model.get("use_referee", False) else 0.0

    mu_home = np.exp(
        b_home
        + w_for * np.log(home_for + 0.5)
        + w_against * np.log(away_against + 0.5)
        + w_total * np.log(home_total_for + 0.5)
        + w_ref * x_ref
    )
    mu_away = np.exp(
        b_away
        + w_for * np.log(away_for + 0.5)
        + w_against * np.log(home_against + 0.5)
        + w_total * np.log(away_total_for + 0.5)
        + w_ref * x_ref
    )

    p_home_vec = np.array([poisson.pmf(k, mu_home) for k in range(MAX_YC + 1)])
    p_away_vec = np.array([poisson.pmf(k, mu_away) for k in range(MAX_YC + 1)])

    p_sum = np.convolve(p_home_vec, p_away_vec)[:MAX_YC + 1]
    total = p_sum.sum()
    if total <= 0:
        p_sum = np.zeros(MAX_YC + 1)
        p_sum[0] = 1.0
    else:
        p_sum = p_sum / total

    out = {
        "lambda_home": float(mu_home),
        "lambda_away": float(mu_away),
        "lambda_sum": float(mu_home + mu_away),
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

def run_backtesting(df_features, target_season, use_referee=False, label="baseline"):
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

        model = fit_poisson_model(df_train, use_referee=use_referee)

        df_round = df_target[df_target["kolejka"] == k]

        for _, mecz in df_round.iterrows():
            pred = predict_yc(
                model=model,
                home_for=mecz["home_for_avg"],
                away_against=mecz["away_against_avg"],
                home_total_for=mecz["home_total_for_avg"],
                away_for=mecz["away_for_avg"],
                home_against=mecz["home_against_avg"],
                away_total_for=mecz["away_total_for_avg"],
                ref_yc_log_ratio=float(mecz["ref_yc_log_ratio"]) if use_referee else 0.0,
            )

            suma = int(mecz["suma_rzecz"])

            row = {
                "match_id": mecz["match_id"],
                "sezon": target_season,
                "kolejka": int(k),
                "gospodarz": mecz["gospodarz"],
                "gosc": mecz["gosc"],
                "zk_gosp_rzecz": float(mecz["zk_gosp_rzecz"]),
                "zk_gosc_rzecz": float(mecz["zk_gosc_rzecz"]),
                "suma_rzecz": float(mecz["suma_rzecz"]),
                "referee_name": mecz["referee_name"],
                "referee_full_name": mecz["referee_full_name"],
                "ref_matches_before": float(mecz["ref_matches_before"]),
                "ref_reliability_k10": float(mecz["ref_reliability_k10"]),
                "league_yc_avg_before": float(mecz["league_yc_avg_before"]),
                "ref_yc_shrunk_before": float(mecz["ref_yc_shrunk_before"]),
                "ref_yc_delta_raw": float(mecz["ref_yc_delta_raw"]),
                "ref_yc_log_ratio": float(mecz["ref_yc_log_ratio"]),
                "lambda_home": round(pred["lambda_home"], 4),
                "lambda_away": round(pred["lambda_away"], 4),
                "lambda_suma": round(pred["lambda_sum"], 4),
                "fit_success": int(model["success"]),
                "fit_fun": round(model["fun"], 4),
                "coef_b_home": round(model["b_home"], 4),
                "coef_b_away": round(model["b_away"], 4),
                "coef_w_for": round(model["w_for"], 4),
                "coef_w_against": round(model["w_against"], 4),
                "coef_w_total": round(model["w_total"], 4),
                "coef_w_ref": round(model["w_ref"], 4),
                "model_variant": label,
                "use_referee": int(use_referee),
            }

            p_sum = pred["p_sum_vec"]
            for exact_k in range(EXACT_SUM_MAX + 1):
                row[f"p_sum_{exact_k}"] = round(float(p_sum[exact_k]), 6)
            tail = float(sum(p_sum[EXACT_SUM_MAX + 1:]))
            row[f"p_sum_{EXACT_SUM_MAX + 1}_plus"] = round(tail, 6)

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
            f"  {target_season} K{k:02d} {label:<8} | "
            f"b_h={model['b_home']:.3f} b_a={model['b_away']:.3f} "
            f"w_for={model['w_for']:.3f} w_against={model['w_against']:.3f} "
            f"w_total={model['w_total']:.3f} w_ref={model['w_ref']:.3f}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuje matches...")
    df_all = load_matches()

    print("\n2. Buduje kauzalne rolling features druzyn...")
    df_features = build_causal_yc_features(df_all, window=8)
    print(f"   Meczow z feature'ami: {len(df_features)}")

    print("\n3. Dogrywam kauzalne features sedziow...")
    df_ref = load_referee_features()
    df_features = merge_referee_features(df_features, df_ref)
    print(f"   Meczow po joinie sedziow: {len(df_features)}")

    print(f"\n4. Rolling backtesting {VAL_SEASON} - baseline...")
    df_val_base = run_backtesting(df_features, VAL_SEASON, use_referee=False, label="baseline")

    print(f"\n5. Rolling backtesting {VAL_SEASON} - referee...")
    df_val_ref = run_backtesting(df_features, VAL_SEASON, use_referee=True, label="referee")

    print(f"\n6. Rolling backtesting {TEST_SEASON} - baseline...")
    df_test_base = run_backtesting(df_features, TEST_SEASON, use_referee=False, label="baseline")

    print(f"\n7. Rolling backtesting {TEST_SEASON} - referee...")
    df_test_ref = run_backtesting(df_features, TEST_SEASON, use_referee=True, label="referee")

    if any(len(df) == 0 for df in [df_val_base, df_val_ref, df_test_base, df_test_ref]):
        raise RuntimeError("Brak wynikow w jednej z ramek backtestingu.")

    df_test_ref.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    bench = -np.log(0.5)

    val_base_avg = mean_ou_logloss(df_val_base)
    val_ref_avg = mean_ou_logloss(df_val_ref)
    test_base_avg = mean_ou_logloss(df_test_base)
    test_ref_avg = mean_ou_logloss(df_test_ref)

    ref_wins_val = 0
    ref_wins_test = 0
    for line in OU_LINES:
        key = str(line).replace(".", "_")
        if float(df_val_ref[f"ll_over_{key}"].mean()) < float(df_val_base[f"ll_over_{key}"].mean()):
            ref_wins_val += 1
        if float(df_test_ref[f"ll_over_{key}"].mean()) < float(df_test_base[f"ll_over_{key}"].mean()):
            ref_wins_test += 1

    lines = []
    lines.append("=" * 100)
    lines.append("MODEL ZOLTYCH KARTEK - REFEREE CHALLENGER OOS RAPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("1. ARCHITEKTURA")
    lines.append("-" * 100)
    lines.append("  Ta sama metodologia co oficjalny baseline model_yellow_cards.py")
    lines.append("  Poisson")
    lines.append("  log(mu_home) = b_home + w_for*log(home_for_avg) + w_against*log(away_against_avg) + w_total*log(home_total_for_avg)")
    lines.append("  log(mu_away) = b_away + w_for*log(away_for_avg) + w_against*log(home_against_avg) + w_total*log(away_total_for_avg)")
    lines.append("  Challenger dodaje:")
    lines.append("      + w_ref * ref_yc_log_ratio")
    lines.append("  gdzie:")
    lines.append("      ref_yc_log_ratio = log(ref_yc_shrunk_before + 0.5) - log(league_yc_avg_before + 0.5)")
    lines.append("  Pelny rozklad sumy zapisany jako p_sum_0 ... p_sum_12, p_sum_13_plus")
    lines.append(f"  Linie OU: {OU_LINES[0]} ... {OU_LINES[-1]}")
    lines.append("  Rolling window: 8 meczow")
    lines.append("  Porownanie 1:1: baseline vs referee")
    lines.append("")

    lines.append("2. WALIDACJA 2024/25 - PODSUMOWANIE")
    lines.append("-" * 100)
    lines.append(f"  Meczow baseline: {len(df_val_base)}")
    lines.append(f"  Meczow referee:  {len(df_val_ref)}")
    lines.append(f"  Avg log-loss baseline: {val_base_avg:.5f}")
    lines.append(f"  Avg log-loss referee:  {val_ref_avg:.5f}")
    lines.append(f"  Delta referee-baseline: {val_ref_avg - val_base_avg:+.5f}")
    lines.append(f"  Referee wygrywa na {ref_wins_val}/{len(OU_LINES)} liniach")
    lines.append("")
    lines.extend(format_compare_block(df_val_base, df_val_ref, "  Walidacja per linia:"))
    lines.append("")
    lines.extend(format_ou_block(df_val_ref, "  Walidacja O/U - referee:"))
    lines.append("")

    lines.append("3. TEST OOS 2025/26 - PODSUMOWANIE")
    lines.append("-" * 100)
    lines.append(f"  Meczow baseline: {len(df_test_base)}")
    lines.append(f"  Meczow referee:  {len(df_test_ref)}")
    lines.append(f"  Benchmark 50/50: {bench:.4f}")
    lines.append(f"  Avg log-loss baseline: {test_base_avg:.5f}")
    lines.append(f"  Avg log-loss referee:  {test_ref_avg:.5f}")
    lines.append(f"  Delta referee-baseline: {test_ref_avg - test_base_avg:+.5f}")
    lines.append(f"  Referee wygrywa na {ref_wins_test}/{len(OU_LINES)} liniach")
    lines.append("")
    lines.extend(format_compare_block(df_test_base, df_test_ref, "  Test per linia:"))
    lines.append("")
    lines.extend(format_ou_block(df_test_ref, "  OOS O/U - referee:"))
    lines.append("")

    lines.append("4. SREDNIE PARAMETRY OOS - REFEREE")
    lines.append("-" * 100)
    lines.append(f"  lambda_home mean: {df_test_ref['lambda_home'].mean():.2f} | rzecz: {df_test_ref['zk_gosp_rzecz'].mean():.2f}")
    lines.append(f"  lambda_away mean: {df_test_ref['lambda_away'].mean():.2f} | rzecz: {df_test_ref['zk_gosc_rzecz'].mean():.2f}")
    lines.append(f"  lambda_suma mean: {df_test_ref['lambda_suma'].mean():.2f} | rzecz: {df_test_ref['suma_rzecz'].mean():.2f}")
    lines.append(f"  coef b_home mean: {df_test_ref['coef_b_home'].mean():.4f}")
    lines.append(f"  coef b_away mean: {df_test_ref['coef_b_away'].mean():.4f}")
    lines.append(f"  coef w_for mean:  {df_test_ref['coef_w_for'].mean():.4f}")
    lines.append(f"  coef w_against mean: {df_test_ref['coef_w_against'].mean():.4f}")
    lines.append(f"  coef w_total mean:   {df_test_ref['coef_w_total'].mean():.4f}")
    lines.append(f"  coef w_ref mean:     {df_test_ref['coef_w_ref'].mean():.4f}")
    lines.append("")
    lines.append("  Referee signal on OOS test:")
    lines.append(f"  ref_yc_shrunk_before mean:  {df_test_ref['ref_yc_shrunk_before'].mean():.3f}")
    lines.append(f"  league_yc_avg_before mean:  {df_test_ref['league_yc_avg_before'].mean():.3f}")
    lines.append(f"  ref_yc_delta_raw mean:      {df_test_ref['ref_yc_delta_raw'].mean():.3f}")
    lines.append(f"  ref_yc_log_ratio mean:      {df_test_ref['ref_yc_log_ratio'].mean():.4f}")
    lines.append(f"  ref_matches_before mean:    {df_test_ref['ref_matches_before'].mean():.2f}")
    lines.append("")

    lines.append("5. PRZYKLADOWE ROZKLADY SUMY (pierwsze 5 meczow OOS - referee)")
    lines.append("-" * 100)
    sample = df_test_ref.head(5)
    for row in sample.itertuples(index=False):
        probs = []
        for k in range(0, 10):
            probs.append(f"{k}:{getattr(row, f'p_sum_{k}'):.3f}")
        lines.append(
            f"  K{row.kolejka:02d} | {row.gospodarz} vs {row.gosc} | "
            f"ref={row.referee_name} | suma_rzecz={row.suma_rzecz:.0f} | "
            + " ".join(probs)
        )
    lines.append("")

    lines.append("6. PLIKI")
    lines.append("-" * 100)
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