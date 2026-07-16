from __future__ import annotations

import argparse
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import nbinom, poisson


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "ekstraklasa.db"
REFEREE_FEATURES_PATH = ROOT / "data" / "processed" / "referee_causal_features.csv"

WINDOW = 8
EPS = 1e-12
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
PRIOR_NEW_TEAM = {"atak": 0.7744, "obrona": 1.0648}
SEASON_ORDER = {"2023/24": 1, "2024/25": 2, "2025/26": 3}

# Współczynniki z realnych modeli OOS
COEFS = {
    "corners": {
        "b_home": 0.9739,
        "b_away": 0.8467,
        "w_for": 0.0790,
        "w_against": 0.1716,
        "w_total": 0.1607,
        "alpha": 0.08,
    },
    "shots": {
        "b_home": 0.6394,
        "b_away": 0.5665,
        "w_for": 0.1724,
        "w_against": 0.4163,
        "w_total": 0.1750,
        "alpha": 0.05,
    },
    "sot": {
        "b_home": 0.7866,
        "b_away": 0.6471,
        "w_for": 0.0741,
        "w_against": 0.2453,
        "w_total": 0.1610,
    },
    "offsides": {
        "b_home": 0.0060,
        "b_away": -0.0179,
        "w_for": 0.1475,
        "w_against": 0.3866,
        "w_total": 0.0772,
    },
    "fouls": {
        "b_home": 0.4110,
        "b_away": 0.4189,
        "w_for": 0.1732,
        "w_against": 0.3371,
        "w_total": 0.3120,
        "w_ref": 0.8159,
        "alpha": 0.0090,
    },
    "yc": {
        "b_home": 0.5483,
        "b_away": 0.6190,
        "w_for": 0.1091,
        "w_against": 0.1138,
        "w_total": -0.0563,
        "w_ref": 0.8728,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def safe_float(x, fallback=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(fallback)
    except Exception:
        return float(fallback)


def safe_avg(values: list, fallback: float, window: int = WINDOW) -> float:
    vals = list(values)[-window:]
    return float(np.mean(vals)) if vals else float(fallback)


# =============================================================================
# DANE
# =============================================================================

def load_matches() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            match_id, sezon, kolejka, data_meczu,
            gospodarz, gosc,
            gole_gosp, gole_gosc,
            xg_gosp, xg_gosc,
            rozne_gosp, rozne_gosc,
            strzaly_gosp, strzaly_gosc,
            celne_gosp, celne_gosc,
            faule_gosp, faule_gosc,
            spalone_gosp, spalone_gosc,
            zk_gosp, zk_gosc,
            waga_sezonu
        FROM matches
        WHERE gole_gosp IS NOT NULL
          AND gole_gosc IS NOT NULL
    """, conn)
    conn.close()

    df["waga_sezonu"] = df["sezon"].map(WAGI_SEZONOW).fillna(1.0)
    df["xgh"] = pd.to_numeric(df["xg_gosp"], errors="coerce").fillna(df["gole_gosp"])
    df["xga"] = pd.to_numeric(df["xg_gosc"], errors="coerce").fillna(df["gole_gosc"])

    df = df.sort_values(
        by=["sezon", "kolejka", "match_id"],
        key=lambda s: s.map(SEASON_ORDER) if s.name == "sezon" else s
    ).reset_index(drop=True)

    return df


def load_referee_features() -> pd.DataFrame:
    if not REFEREE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Brak {REFEREE_FEATURES_PATH}")
    return pd.read_csv(REFEREE_FEATURES_PATH)


def build_team_name_map(df: pd.DataFrame) -> dict[str, str]:
    names = sorted(set(df["gospodarz"].tolist()) | set(df["gosc"].tolist()))
    return {normalize_text(n): n for n in names}


def resolve_team(user_value: str, team_map: dict[str, str]) -> tuple[str, bool]:
    norm = normalize_text(user_value)
    if norm in team_map:
        return team_map[norm], True
    return user_value.strip(), False


# =============================================================================
# BLOK 1: 1X2 / GOLE / BTTS
# =============================================================================

def przygotuj_mle(df: pd.DataFrame) -> dict:
    teams = sorted(set(df["gospodarz"]) | set(df["gosc"]))
    t2i = {t: i for i, t in enumerate(teams)}
    i2t = {i: t for t, i in t2i.items()}
    return {
        "n": len(teams), "i2t": i2t, "t2i": t2i,
        "hi": df["gospodarz"].map(t2i).values,
        "ai": df["gosc"].map(t2i).values,
        "gh": df["xgh"].values.astype(float),
        "ga": df["xga"].values.astype(float),
        "w": df["waga_sezonu"].values.astype(float),
    }


def nll_poisson(theta: np.ndarray, d: dict) -> float:
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

    lh = np.maximum(mh * a[d["hi"]] * b[d["ai"]], EPS)
    ll = np.maximum(ma * a[d["ai"]] * b[d["hi"]], EPS)

    return -np.sum(d["w"] * (d["gh"] * np.log(lh) - lh + d["ga"] * np.log(ll) - ll))


def fit_poisson_mle(df: pd.DataFrame) -> dict:
    d = przygotuj_mle(df)
    N = d["n"]
    t0 = np.zeros(2 + 2 * (N - 1))
    t0[0] = np.log(max(d["gh"].mean(), 0.01))
    t0[1] = np.log(max(d["ga"].mean(), 0.01))

    res = minimize(
        nll_poisson, t0, args=(d,),
        method="L-BFGS-B",
        options={"maxiter": 10000},
    )

    la = np.zeros(N)
    lb = np.zeros(N)
    la[:N - 1] = res.x[2:2 + N - 1]
    la[N - 1] = -la[:N - 1].sum()
    lb[:N - 1] = res.x[2 + N - 1:2 + 2 * (N - 1)]
    lb[N - 1] = -lb[:N - 1].sum()

    return {
        "mh": float(np.exp(res.x[0])),
        "ma": float(np.exp(res.x[1])),
        "alpha": {d["i2t"][i]: float(np.exp(la[i])) for i in range(N)},
        "beta":  {d["i2t"][i]: float(np.exp(lb[i])) for i in range(N)},
    }


def apply_priors(params: dict, all_seasons_df: pd.DataFrame) -> dict:
    for season, teams in PROMOTED.items():
        pr = PRIORS[season]
        df_prev = all_seasons_df[all_seasons_df["sezon"] == season]
        for team in teams:
            n = int(((df_prev["gospodarz"] == team) | (df_prev["gosc"] == team)).sum())
            a = params["alpha"].get(team, pr["atak"])
            b = params["beta"].get(team, pr["obrona"])
            params["alpha"][team] = (K_PRIOR * pr["atak"] + n * a) / (K_PRIOR + n)
            params["beta"][team]  = (K_PRIOR * pr["obrona"] + n * b) / (K_PRIOR + n)
    return params


def apply_prior_new_team(params: dict, team: str) -> dict:
    if team not in params["alpha"]:
        params["alpha"][team] = PRIOR_NEW_TEAM["atak"]
        params["beta"][team] = PRIOR_NEW_TEAM["obrona"]
    return params


def predict_goals_matrix(params: dict, home: str, away: str) -> dict:
    lh = params["mh"] * params["alpha"][home] * params["beta"][away]
    la = params["ma"] * params["alpha"][away] * params["beta"][home]

    m = np.zeros((MAX_GOLE, MAX_GOLE))
    for i in range(MAX_GOLE):
        for j in range(MAX_GOLE):
            m[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)

    total = m.sum()
    if total > 0:
        m /= total

    p_H = float(np.tril(m, -1).sum())
    p_D = float(np.trace(m))
    p_A = float(np.triu(m, 1).sum())

    p_under = {}
    for prog in [0, 1, 2, 3]:
        p_under[prog] = float(sum(
            m[i, j]
            for i in range(MAX_GOLE)
            for j in range(MAX_GOLE)
            if i + j <= prog
        ))

    p_btts_no = float(sum(
        m[i, j]
        for i in range(MAX_GOLE)
        for j in range(MAX_GOLE)
        if i == 0 or j == 0
    ))

    return {
        "lambda_home": float(lh),
        "lambda_away": float(la),
        "p_H": p_H,
        "p_D": p_D,
        "p_A": p_A,
        "p_over_05": 1.0 - p_under[0],
        "p_over_15": 1.0 - p_under[1],
        "p_over_25": 1.0 - p_under[2],
        "p_over_35": 1.0 - p_under[3],
        "p_under_05": p_under[0],
        "p_under_15": p_under[1],
        "p_under_25": p_under[2],
        "p_under_35": p_under[3],
        "p_btts_yes": 1.0 - p_btts_no,
        "p_btts_no": p_btts_no,
        "alpha_home": params["alpha"][home],
        "alpha_away": params["alpha"][away],
        "beta_home": params["beta"][home],
        "beta_away": params["beta"][away],
    }


def softmax_cal(ph, pd_, pa, T, bH, bD, bA):
    logits = np.log(np.maximum([ph, pd_, pa], EPS))
    e = np.exp((logits + [bH, bD, bA]) / T)
    s = e.sum()
    return float(e[0] / s), float(e[1] / s), float(e[2] / s)


def fit_cal_1x2(df_val: pd.DataFrame) -> np.ndarray:
    L2 = 0.05

    def obj(p):
        T, bH, bD, bA = p
        if T <= 0.1:
            return 999.0
        total = 0.0
        for r in df_val.itertuples():
            p_cal = softmax_cal(r.p_H_raw, r.p_D_raw, r.p_A_raw, T, bH, bD, bA)
            idx = {"H": 0, "D": 1, "A": 2}[r.wynik]
            total += -np.log(max(p_cal[idx], EPS))
        return total / len(df_val) + L2 * (bH**2 + bD**2 + bA**2)

    res = minimize(
        obj, [1.5, 0.0, 0.0, 0.0],
        method="L-BFGS-B",
        bounds=[(0.5, 5.0), (-1, 1), (-1, 1), (-1, 1)],
    )
    return res.x


def fit_btts_shift(df_val: pd.DataFrame) -> float:
    p_raw = df_val["p_btts_yes"].values
    y = df_val["btts_rzecz"].values

    def obj(shift):
        p_corr = np.clip(p_raw + shift, 0.01, 0.99)
        ll = -(y * np.log(p_corr) + (1 - y) * np.log(1 - p_corr))
        return float(ll.mean())

    res = minimize_scalar(obj, bounds=(-0.15, 0.15), method="bounded")
    return float(res.x)


def run_val_predictions(df_all: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_val = df_all[df_all["sezon"] == "2024/25"].copy()
    rows = []
    for _, row in df_val.iterrows():
        home, away = row["gospodarz"], row["gosc"]
        pred = predict_goals_matrix(params, home, away)
        gh, ga = int(row["gole_gosp"]), int(row["gole_gosc"])
        rows.append({
            "p_H_raw": pred["p_H"],
            "p_D_raw": pred["p_D"],
            "p_A_raw": pred["p_A"],
            "p_btts_yes": pred["p_btts_yes"],
            "btts_rzecz": int(gh >= 1 and ga >= 1),
            "wynik": "H" if gh > ga else ("D" if gh == ga else "A"),
        })
    return pd.DataFrame(rows)


# =============================================================================
# BLOK 2: ROLLING STATE
# =============================================================================

def init_team_state() -> dict:
    return {
        "corners_for": [], "corners_against": [], "corners_all": [],
        "shots_for": [], "shots_against": [], "shots_all": [],
        "sot_for": [], "sot_against": [], "sot_all": [],
        "offsides_for": [], "offsides_against": [], "offsides_all": [],
        "fouls_for": [], "fouls_against": [], "fouls_all": [],
        "yc_for": [], "yc_against": [], "yc_all": [],
    }


def build_rolling_state(df: pd.DataFrame) -> tuple[dict, dict]:
    state = {}
    global_avgs = {
        "corners": [], "shots": [], "sot": [],
        "offsides": [], "fouls": [], "yc": [],
    }

    for _, row in df.iterrows():
        home, away = row["gospodarz"], row["gosc"]
        if home not in state:
            state[home] = init_team_state()
        if away not in state:
            state[away] = init_team_state()

        ch = safe_float(row.get("rozne_gosp", 0))
        ca = safe_float(row.get("rozne_gosc", 0))
        sh = safe_float(row.get("strzaly_gosp", 0))
        sa = safe_float(row.get("strzaly_gosc", 0))
        soh = safe_float(row.get("celne_gosp", 0))
        soa = safe_float(row.get("celne_gosc", 0))
        oh = safe_float(row.get("spalone_gosp", 0))
        oa = safe_float(row.get("spalone_gosc", 0))
        fh = safe_float(row.get("faule_gosp", 0))
        fa = safe_float(row.get("faule_gosc", 0))
        zkh = safe_float(row.get("zk_gosp", 0))
        zka = safe_float(row.get("zk_gosc", 0))

        hs = state[home]
        aw = state[away]

        hs["corners_for"].append(ch); hs["corners_against"].append(ca); hs["corners_all"].append(ch)
        aw["corners_for"].append(ca); aw["corners_against"].append(ch); aw["corners_all"].append(ca)

        hs["shots_for"].append(sh); hs["shots_against"].append(sa); hs["shots_all"].append(sh)
        aw["shots_for"].append(sa); aw["shots_against"].append(sh); aw["shots_all"].append(sa)

        hs["sot_for"].append(soh); hs["sot_against"].append(soa); hs["sot_all"].append(soh)
        aw["sot_for"].append(soa); aw["sot_against"].append(soh); aw["sot_all"].append(soa)

        hs["offsides_for"].append(oh); hs["offsides_against"].append(oa); hs["offsides_all"].append(oh)
        aw["offsides_for"].append(oa); aw["offsides_against"].append(oh); aw["offsides_all"].append(oa)

        hs["fouls_for"].append(fh); hs["fouls_against"].append(fa); hs["fouls_all"].append(fh)
        aw["fouls_for"].append(fa); aw["fouls_against"].append(fh); aw["fouls_all"].append(fa)

        hs["yc_for"].append(zkh); hs["yc_against"].append(zka); hs["yc_all"].append(zkh)
        aw["yc_for"].append(zka); aw["yc_against"].append(zkh); aw["yc_all"].append(zka)

        global_avgs["corners"].append(ch + ca)
        global_avgs["shots"].append(sh + sa)
        global_avgs["sot"].append(soh + soa)
        global_avgs["offsides"].append(oh + oa)
        global_avgs["fouls"].append(fh + fa)
        global_avgs["yc"].append(zkh + zka)

    g = {k: float(np.mean(v)) if v else 0.0 for k, v in global_avgs.items()}
    return state, g


def get_stat_feature(state, team, key, fallback, is_new=False, prior_mult=1.0):
    if is_new or team not in state or len(state[team].get(key, [])) < 3:
        return fallback * prior_mult
    return safe_avg(state[team][key], fallback, WINDOW)


# =============================================================================
# BLOK 3: PREDYKCJE STATYSTYK
# =============================================================================

def predict_negbin_market(mu_h, mu_a, alpha, lines, max_k=60):
    r = 1.0 / alpha
    p_h = 1.0 / (1.0 + alpha * mu_h)
    p_a = 1.0 / (1.0 + alpha * mu_a)

    pmf_h = np.array([nbinom.pmf(k, r, p_h) for k in range(max_k + 1)])
    pmf_a = np.array([nbinom.pmf(k, r, p_a) for k in range(max_k + 1)])
    pmf_t = np.convolve(pmf_h, pmf_a)[:max_k + 1]
    s = pmf_t.sum()
    if s > 0:
        pmf_t /= s

    out = {"mu_h": mu_h, "mu_a": mu_a, "mu_total": mu_h + mu_a}
    for line in lines:
        p_over = float(sum(pmf_t[k] for k in range(len(pmf_t)) if k > line))
        p_over = min(max(p_over, 0.0), 1.0)
        key = str(line).replace(".", "_")
        out[f"p_over_{key}"] = p_over
        out[f"p_under_{key}"] = 1.0 - p_over
    return out


def predict_poisson_market(mu_h, mu_a, lines, max_k=30):
    pmf_h = np.array([poisson.pmf(k, max(mu_h, EPS)) for k in range(max_k + 1)])
    pmf_a = np.array([poisson.pmf(k, max(mu_a, EPS)) for k in range(max_k + 1)])
    pmf_t = np.convolve(pmf_h, pmf_a)[:max_k + 1]
    s = pmf_t.sum()
    if s > 0:
        pmf_t /= s

    out = {"mu_h": mu_h, "mu_a": mu_a, "mu_total": mu_h + mu_a}
    for line in lines:
        p_over = float(sum(pmf_t[k] for k in range(len(pmf_t)) if k > line))
        p_over = min(max(p_over, 0.0), 1.0)
        key = str(line).replace(".", "_")
        out[f"p_over_{key}"] = p_over
        out[f"p_under_{key}"] = 1.0 - p_over
    return out


def compute_stat_predictions(
    state, g, home, away, home_is_new, away_is_new,
    ref_fouls_log_ratio, ref_yc_log_ratio,
):
    def hval(key, fallback, is_def=False):
        mult = PRIOR_NEW_TEAM["obrona"] if is_def else PRIOR_NEW_TEAM["atak"]
        return get_stat_feature(state, home, key, fallback, home_is_new, mult if home_is_new else 1.0)

    def aval(key, fallback, is_def=False):
        mult = PRIOR_NEW_TEAM["obrona"] if is_def else PRIOR_NEW_TEAM["atak"]
        return get_stat_feature(state, away, key, fallback, away_is_new, mult if away_is_new else 1.0)

    g_c = g["corners"] / 2.0
    g_s = g["shots"] / 2.0
    g_sot = g["sot"] / 2.0
    g_o = g["offsides"] / 2.0
    g_f = g["fouls"] / 2.0
    g_y = g["yc"] / 2.0

    # Kornery
    cc = COEFS["corners"]
    mu_c_h = np.exp(
        cc["b_home"]
        + cc["w_for"] * np.log(hval("corners_for", g_c) + 0.5)
        + cc["w_against"] * np.log(aval("corners_against", g_c, True) + 0.5)
        + cc["w_total"] * np.log(hval("corners_all", g_c) + 0.5)
    )
    mu_c_a = np.exp(
        cc["b_away"]
        + cc["w_for"] * np.log(aval("corners_for", g_c) + 0.5)
        + cc["w_against"] * np.log(hval("corners_against", g_c, True) + 0.5)
        + cc["w_total"] * np.log(aval("corners_all", g_c) + 0.5)
    )
    corners = predict_negbin_market(mu_c_h, mu_c_a, cc["alpha"], [x + 0.5 for x in range(4, 12)], 30)

    # Strzaly
    cs = COEFS["shots"]
    mu_s_h = np.exp(
        cs["b_home"]
        + cs["w_for"] * np.log(hval("shots_for", g_s) + 0.5)
        + cs["w_against"] * np.log(aval("shots_against", g_s, True) + 0.5)
        + cs["w_total"] * np.log(hval("shots_all", g_s) + 0.5)
    )
    mu_s_a = np.exp(
        cs["b_away"]
        + cs["w_for"] * np.log(aval("shots_for", g_s) + 0.5)
        + cs["w_against"] * np.log(hval("shots_against", g_s, True) + 0.5)
        + cs["w_total"] * np.log(aval("shots_all", g_s) + 0.5)
    )
    shots = predict_negbin_market(mu_s_h, mu_s_a, cs["alpha"], [x + 0.5 for x in range(12, 36)], 60)

    # Strzaly celne
    cso = COEFS["sot"]
    mu_so_h = np.exp(
        cso["b_home"]
        + cso["w_for"] * np.log(hval("sot_for", g_sot) + 0.5)
        + cso["w_against"] * np.log(aval("sot_against", g_sot, True) + 0.5)
        + cso["w_total"] * np.log(hval("sot_all", g_sot) + 0.5)
    )
    mu_so_a = np.exp(
        cso["b_away"]
        + cso["w_for"] * np.log(aval("sot_for", g_sot) + 0.5)
        + cso["w_against"] * np.log(hval("sot_against", g_sot, True) + 0.5)
        + cso["w_total"] * np.log(aval("sot_all", g_sot) + 0.5)
    )
    sot = predict_poisson_market(mu_so_h, mu_so_a, [x + 0.5 for x in range(3, 16)], 30)

    # Spalone
    co = COEFS["offsides"]
    mu_o_h = np.exp(
        co["b_home"]
        + co["w_for"] * np.log(hval("offsides_for", g_o) + 0.5)
        + co["w_against"] * np.log(aval("offsides_against", g_o, True) + 0.5)
        + co["w_total"] * np.log(hval("offsides_all", g_o) + 0.5)
    )
    mu_o_a = np.exp(
        co["b_away"]
        + co["w_for"] * np.log(aval("offsides_for", g_o) + 0.5)
        + co["w_against"] * np.log(hval("offsides_against", g_o, True) + 0.5)
        + co["w_total"] * np.log(aval("offsides_all", g_o) + 0.5)
    )
    offsides = predict_poisson_market(mu_o_h, mu_o_a, [x + 0.5 for x in range(0, 7)], 20)

    # Faule
    cf = COEFS["fouls"]
    x_ref_f = float(np.clip(ref_fouls_log_ratio, -1.0, 1.0))
    mu_f_h = np.exp(
        cf["b_home"]
        + cf["w_for"] * np.log(hval("fouls_for", g_f) + 0.5)
        + cf["w_against"] * np.log(aval("fouls_against", g_f, True) + 0.5)
        + cf["w_total"] * np.log(hval("fouls_all", g_f) + 0.5)
        + cf["w_ref"] * x_ref_f
    )
    mu_f_a = np.exp(
        cf["b_away"]
        + cf["w_for"] * np.log(aval("fouls_for", g_f) + 0.5)
        + cf["w_against"] * np.log(hval("fouls_against", g_f, True) + 0.5)
        + cf["w_total"] * np.log(aval("fouls_all", g_f) + 0.5)
        + cf["w_ref"] * x_ref_f
    )
    fouls = predict_negbin_market(mu_f_h, mu_f_a, cf["alpha"], [x + 0.5 for x in range(15, 34)], 60)

    # ZK
    cy = COEFS["yc"]
    x_ref_y = float(np.clip(ref_yc_log_ratio, -1.0, 1.0))
    mu_y_h = np.exp(
        cy["b_home"]
        + cy["w_for"] * np.log(hval("yc_for", g_y) + 0.5)
        + cy["w_against"] * np.log(aval("yc_against", g_y, True) + 0.5)
        + cy["w_total"] * np.log(hval("yc_all", g_y) + 0.5)
        + cy["w_ref"] * x_ref_y
    )
    mu_y_a = np.exp(
        cy["b_away"]
        + cy["w_for"] * np.log(aval("yc_for", g_y) + 0.5)
        + cy["w_against"] * np.log(hval("yc_against", g_y, True) + 0.5)
        + cy["w_total"] * np.log(aval("yc_all", g_y) + 0.5)
        + cy["w_ref"] * x_ref_y
    )
    yc = predict_poisson_market(mu_y_h, mu_y_a, [x + 0.5 for x in range(0, 8)], 18)

    return {
        "corners": corners,
        "shots": shots,
        "sot": sot,
        "offsides": offsides,
        "fouls": fouls,
        "yc": yc,
    }


# =============================================================================
# SEDZIA
# =============================================================================

def get_referee_features(ref_name, df_ref, g_fouls, g_yc):
    if not ref_name:
        return {
            "ref_known": False,
            "ref_matches": 0,
            "ref_fouls_log_ratio": 0.0,
            "ref_yc_log_ratio": 0.0,
            "ref_fouls_shrunk": g_fouls,
            "ref_yc_shrunk": g_yc,
        }

    norm = normalize_text(ref_name)
    mask = (
        df_ref["referee_name"].fillna("").map(normalize_text) == norm
    ) | (
        df_ref["referee_full_name"].fillna("").map(normalize_text) == norm
    )
    found = df_ref[mask]

    if found.empty:
        print(f"  [INFO] Sedzia '{ref_name}' nieznany — prior ligowy.")
        return {
            "ref_known": False,
            "ref_matches": 0,
            "ref_fouls_log_ratio": 0.0,
            "ref_yc_log_ratio": 0.0,
            "ref_fouls_shrunk": g_fouls,
            "ref_yc_shrunk": g_yc,
        }

    last = found.iloc[-1]
    ref_fouls = safe_float(last["ref_fouls_shrunk_before"], g_fouls)
    lea_fouls = safe_float(last["league_fouls_avg_before"], g_fouls)
    ref_yc = safe_float(last["ref_yc_shrunk_before"], g_yc)
    lea_yc = safe_float(last["league_yc_avg_before"], g_yc)

    return {
        "ref_known": True,
        "ref_matches": int(safe_float(last["ref_matches_before"], 0)),
        "ref_fouls_log_ratio": float(np.clip(np.log(ref_fouls + 0.5) - np.log(lea_fouls + 0.5), -1.0, 1.0)),
        "ref_yc_log_ratio": float(np.clip(np.log(ref_yc + 0.5) - np.log(lea_yc + 0.5), -1.0, 1.0)),
        "ref_fouls_shrunk": ref_fouls,
        "ref_yc_shrunk": ref_yc,
    }


# =============================================================================
# RAPORT
# =============================================================================

def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def print_report(home, away, ref_name, ref, goals, cal, stats):
    sep = "=" * 72
    thin = "-" * 72

    p_H_cal, p_D_cal, p_A_cal = cal["p_H"], cal["p_D"], cal["p_A"]
    btts_shift = cal["btts_shift"]
    p_btts = min(max(goals["p_btts_yes"] + btts_shift, 0.01), 0.99)

    print(f"\n{sep}")
    print(f"  PREDYKCJA: {home} vs {away}")
    if ref_name:
        if ref["ref_known"]:
            print(f"  Sedzia: {ref_name} [znany, n={ref['ref_matches']}]")
        else:
            print(f"  Sedzia: {ref_name} [nieznany — prior ligowy]")
    else:
        print("  Sedzia: nieznany — prior ligowy")
    print(sep)

    print("\n  SILA DRUZYN (Poisson MLE)")
    print(thin)
    print(f"  {home:<34} atak={goals['alpha_home']:.3f}  obrona={goals['beta_home']:.3f}")
    print(f"  {away:<34} atak={goals['alpha_away']:.3f}  obrona={goals['beta_away']:.3f}")

    print("\n  1X2  (skalibrowany Softmax)")
    print(thin)
    print(f"  {home:<34}{pct(p_H_cal)}")
    print(f"  Remis{'':<30}{pct(p_D_cal)}")
    print(f"  {away:<34}{pct(p_A_cal)}")

    print(f"\n  GOLE  (ocz: {home} {goals['lambda_home']:.2f} | {away} {goals['lambda_away']:.2f})")
    print(thin)
    for line, key in [(0.5, "05"), (1.5, "15"), (2.5, "25"), (3.5, "35")]:
        print(f"  Over {line}  {pct(goals[f'p_over_{key}'])}   Under {line}  {pct(goals[f'p_under_{key}'])}")
    print(f"  BTTS Yes {pct(p_btts)}   BTTS No {pct(1 - p_btts)}")

    c = stats["corners"]
    print(f"\n  KORNERY  (ocz suma: {c['mu_total']:.1f})")
    print(thin)
    for line in [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}  {pct(c[f'p_over_{key}'])}   Under {line}  {pct(c[f'p_under_{key}'])}")

    s = stats["shots"]
    print(f"\n  STRZALY  (ocz suma: {s['mu_total']:.1f})")
    print(thin)
    for line in [19.5, 21.5, 23.5, 25.5, 27.5, 29.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}  {pct(s[f'p_over_{key}'])}   Under {line}  {pct(s[f'p_under_{key}'])}")

    so = stats["sot"]
    print(f"\n  STRZALY CELNE  (ocz suma: {so['mu_total']:.1f})")
    print(thin)
    for line in [5.5, 6.5, 7.5, 8.5, 9.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}  {pct(so[f'p_over_{key}'])}   Under {line}  {pct(so[f'p_under_{key}'])}")

    off = stats["offsides"]
    print(f"\n  SPALONE  (ocz suma: {off['mu_total']:.1f})")
    print(thin)
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}  {pct(off[f'p_over_{key}'])}   Under {line}  {pct(off[f'p_under_{key}'])}")

    f = stats["fouls"]
    ref_f_note = f"(sedzia avg faule: {ref['ref_fouls_shrunk']:.1f})" if ref["ref_known"] else "(prior ligowy)"
    print(f"\n  FAULE  (ocz suma: {f['mu_total']:.1f}) {ref_f_note}")
    print(thin)
    for line in [21.5, 23.5, 25.5, 27.5, 29.5, 31.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}  {pct(f[f'p_over_{key}'])}   Under {line}  {pct(f[f'p_under_{key}'])}")

    yc = stats["yc"]
    ref_yc_note = f"(sedzia avg ZK: {ref['ref_yc_shrunk']:.1f})" if ref["ref_known"] else "(prior ligowy)"
    print(f"\n  ZOLTE KARTKI  (ocz suma: {yc['mu_total']:.1f}) {ref_yc_note}")
    print(thin)
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}  {pct(yc[f'p_over_{key}'])}   Under {line}  {pct(yc[f'p_under_{key}'])}")

    print(f"\n{sep}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Predykcja meczu Ekstraklasy")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    print("1. Laduje dane historyczne...")
    df_all = load_matches()
    print(f"   Meczow: {len(df_all)}")

    print("2. Rozwiazuje nazwy druzyn...")
    team_map = build_team_name_map(df_all)
    home_resolved, home_known = resolve_team(args.home, team_map)
    away_resolved, away_known = resolve_team(args.away, team_map)

    if normalize_text(home_resolved) == normalize_text(away_resolved):
        print("Blad: gospodarz i gosc to ta sama druzyna.")
        sys.exit(1)

    home_is_new = not home_known
    away_is_new = not away_known

    if home_known:
        print(f"   Gospodarz: '{args.home}' -> '{home_resolved}'")
    else:
        print(f"   Gospodarz: '{args.home}' nieznany -> beniaminek (prior)")
    if away_known:
        print(f"   Gosc:      '{args.away}' -> '{away_resolved}'")
    else:
        print(f"   Gosc:      '{args.away}' nieznany -> beniaminek (prior)")

    print("3. Laduje profile sedziow...")
    df_ref = load_referee_features()

    print("4. Fit Poisson MLE na calej historii...")
    params = fit_poisson_mle(df_all)
    params = apply_priors(params, df_all)
    apply_prior_new_team(params, home_resolved)
    apply_prior_new_team(params, away_resolved)

    print("5. Kalibracja 1X2 i BTTS na 2024/25...")
    df_val_preds = run_val_predictions(df_all, params)
    cal_1x2_params = fit_cal_1x2(df_val_preds)
    btts_shift = fit_btts_shift(df_val_preds)
    T, bH, bD, bA = cal_1x2_params
    print(f"   T={T:.3f} bH={bH:.3f} bD={bD:.3f} bA={bA:.3f} | BTTS shift={btts_shift:+.3f}")

    print("6. Buduje rolling state druzyn...")
    state, g = build_rolling_state(df_all)

    print("7. Laduje profil sedziego...")
    ref = get_referee_features(args.ref, df_ref, g["fouls"] / 2.0, g["yc"] / 2.0)

    print("8. Licze predykcje...")
    goals = predict_goals_matrix(params, home_resolved, away_resolved)
    p_H_cal, p_D_cal, p_A_cal = softmax_cal(goals["p_H"], goals["p_D"], goals["p_A"], T, bH, bD, bA)
    cal = {"p_H": p_H_cal, "p_D": p_D_cal, "p_A": p_A_cal, "btts_shift": btts_shift}

    stats = compute_stat_predictions(
        state=state,
        g=g,
        home=home_resolved,
        away=away_resolved,
        home_is_new=home_is_new,
        away_is_new=away_is_new,
        ref_fouls_log_ratio=ref["ref_fouls_log_ratio"],
        ref_yc_log_ratio=ref["ref_yc_log_ratio"],
    )

    print_report(
        home=home_resolved,
        away=away_resolved,
        ref_name=args.ref,
        ref=ref,
        goals=goals,
        cal=cal,
        stats=stats,
    )


if __name__ == "__main__":
    main()