import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, LinearConstraint, Bounds

DB_PATH = Path("db/ekstraklasa.db")
FLASH_PATH = Path("data/raw/flash/flash_2023_24.csv")
TEAM_XG_PATH = Path("data/raw/ekstraklasa_org/druzyny/2023-2024_druzynowe-xg.csv")
TEAM_XGA_PATH = Path("data/raw/ekstraklasa_org/druzyny/2023-2024_druzynowe-xga.csv")

OUTPUT_PATH = Path("data/processed/matches_2023_24_xg_hybrid.csv")
TEAM_CHECK_PATH = Path("data/reports/model/matches_2023_24_xg_hybrid_team_check.csv")
REPORT_PATH = Path("data/reports/model/matches_2023_24_xg_hybrid_report.txt")

TEAM_TO_SLUG = {
    "Cracovia": "cracovia",
    "Górnik Zabrze": "gornik-zabrze",
    "Jagiellonia Białystok": "jagiellonia-bialystok",
    "Korona Kielce": "korona-kielce",
    "Lech Poznań": "lech-poznan",
    "Legia Warszawa": "legia-warszawa",
    "Piast Gliwice": "piast-gliwice",
    "Pogoń Szczecin": "pogon-szczecin",
    "Puszcza Niepołomice": "puszcza-niepolomice",
    "Radomiak Radom": "radomiak-radom",
    "Raków Częstochowa": "rakow-czestochowa",
    "Ruch Chorzów": "ruch-chorzow",
    "Stal Mielec": "stal-mielec",
    "Warta Poznań": "warta-poznan",
    "Widzew Łódź": "widzew-lodz",
    "Zagłębie Lubin": "zagebie-lubin",
    "ŁKS Łódź": "lks-lodz",
    "Śląsk Wrocław": "slask-wroclaw",
}


def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


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
            gole_gosp,
            gole_gosc,
            strzaly_gosp,
            strzaly_gosc,
            celne_gosp,
            celne_gosc,
            strzaly_zablokowane_gosp,
            strzaly_zablokowane_gosc,
            flash_id,
            flash_url
        FROM matches
        WHERE sezon = '2023/24'
        ORDER BY kolejka, match_id
    """, conn)
    conn.close()
    return df


def load_flash():
    df = safe_read_csv(FLASH_PATH).copy()
    required = {"flash_id", "xg_gosp", "xg_gosc"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Brak kolumn w {FLASH_PATH}: {sorted(missing)}")

    df["flash_id"] = df["flash_id"].astype(str).str.strip()
    for col in [
        "xg_gosp", "xg_gosc",
        "strzaly_gosp", "strzaly_gosc",
        "celne_gosp", "celne_gosc",
        "strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_official_totals():
    df_xg = safe_read_csv(TEAM_XG_PATH).copy()
    df_xga = safe_read_csv(TEAM_XGA_PATH).copy()

    required = {"klub_slug", "wartosc"}
    missing_xg = required - set(df_xg.columns)
    missing_xga = required - set(df_xga.columns)
    if missing_xg:
        raise RuntimeError(f"Brak kolumn w {TEAM_XG_PATH}: {sorted(missing_xg)}")
    if missing_xga:
        raise RuntimeError(f"Brak kolumn w {TEAM_XGA_PATH}: {sorted(missing_xga)}")

    df_xg = df_xg[["klub_slug", "nazwa", "wartosc"]].copy()
    df_xga = df_xga[["klub_slug", "nazwa", "wartosc"]].copy()

    df_xg["official_xg"] = pd.to_numeric(df_xg["wartosc"], errors="coerce")
    df_xga["official_xga"] = pd.to_numeric(df_xga["wartosc"], errors="coerce")

    out = df_xg[["klub_slug", "nazwa", "official_xg"]].merge(
        df_xga[["klub_slug", "official_xga"]],
        on="klub_slug",
        how="inner",
        validate="1:1"
    )
    return out


def merge_matches_flash(matches, flash):
    matches = matches.copy()
    matches["flash_id"] = matches["flash_id"].astype(str).str.strip()

    keep_cols = ["flash_id", "xg_gosp", "xg_gosc"]
    merged = matches.merge(
        flash[keep_cols].rename(columns={
            "xg_gosp": "xg_gosp_obs",
            "xg_gosc": "xg_gosc_obs",
        }),
        on="flash_id",
        how="left",
        validate="1:1"
    )

    merged["has_observed_xg"] = (
        merged["xg_gosp_obs"].notna() & merged["xg_gosc_obs"].notna()
    ).astype(int)

    return merged


def build_long_observed(merged):
    obs = merged[merged["has_observed_xg"] == 1].copy()

    for c in [
        "gole_gosp", "gole_gosc",
        "strzaly_gosp", "strzaly_gosc",
        "celne_gosp", "celne_gosc",
        "strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc",
    ]:
        obs[c] = pd.to_numeric(obs[c], errors="coerce").fillna(0.0)

    home = pd.DataFrame({
        "team_name": obs["gospodarz"],
        "side": "home",
        "goals": obs["gole_gosp"],
        "shots": obs["strzaly_gosp"],
        "shots_on_target": obs["celne_gosp"],
        "blocked": obs["strzaly_zablokowane_gosp"].fillna(0.0),
        "xg": obs["xg_gosp_obs"],
    })

    away = pd.DataFrame({
        "team_name": obs["gosc"],
        "side": "away",
        "goals": obs["gole_gosc"],
        "shots": obs["strzaly_gosc"],
        "shots_on_target": obs["celne_gosc"],
        "blocked": obs["strzaly_zablokowane_gosc"].fillna(0.0),
        "xg": obs["xg_gosc_obs"],
    })

    long_obs = pd.concat([home, away], ignore_index=True)
    long_obs["xg"] = pd.to_numeric(long_obs["xg"], errors="coerce")
    long_obs = long_obs.dropna(subset=["xg"]).reset_index(drop=True)
    return long_obs


def fit_proxy_model(long_obs):
    X = long_obs[["goals", "shots", "shots_on_target", "blocked"]].fillna(0.0).values
    y = long_obs["xg"].values.astype(float)

    model = LinearRegression(positive=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0.05, None)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(y - y_pred))

    return model, {"r2": float(r2), "mae": float(mae)}


def predict_missing_proxies(merged, model):
    miss = merged[merged["has_observed_xg"] == 0].copy()

    for c in [
        "gole_gosp", "gole_gosc",
        "strzaly_gosp", "strzaly_gosc",
        "celne_gosp", "celne_gosc",
        "strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc",
    ]:
        miss[c] = pd.to_numeric(miss[c], errors="coerce").fillna(0.0)

    X_home = miss[["gole_gosp", "strzaly_gosp", "celne_gosp", "strzaly_zablokowane_gosp"]].copy()
    X_home.columns = ["goals", "shots", "shots_on_target", "blocked"]

    X_away = miss[["gole_gosc", "strzaly_gosc", "celne_gosc", "strzaly_zablokowane_gosc"]].copy()
    X_away.columns = ["goals", "shots", "shots_on_target", "blocked"]

    miss["xg_gosp_proxy"] = np.clip(model.predict(X_home.values), 0.05, None)
    miss["xg_gosc_proxy"] = np.clip(model.predict(X_away.values), 0.05, None)

    return miss


def build_team_residuals(merged, official):
    # observed sums from known matches
    known = merged[merged["has_observed_xg"] == 1].copy()

    rows = []
    for _, row in known.iterrows():
        rows.append({
            "team_name": row["gospodarz"],
            "observed_xg": row["xg_gosp_obs"],
            "observed_xga": row["xg_gosc_obs"],
        })
        rows.append({
            "team_name": row["gosc"],
            "observed_xg": row["xg_gosc_obs"],
            "observed_xga": row["xg_gosp_obs"],
        })

    obs_team = pd.DataFrame(rows).groupby("team_name", as_index=False).agg(
        observed_xg=("observed_xg", "sum"),
        observed_xga=("observed_xga", "sum"),
    )

    team_df = pd.DataFrame({"team_name": sorted(set(merged["gospodarz"]) | set(merged["gosc"]))})
    team_df["klub_slug"] = team_df["team_name"].map(TEAM_TO_SLUG)

    if team_df["klub_slug"].isna().any():
        missing = team_df.loc[team_df["klub_slug"].isna(), "team_name"].tolist()
        raise RuntimeError(f"Brak mapowania team -> klub_slug dla: {missing}")

    out = team_df.merge(
        official[["klub_slug", "official_xg", "official_xga"]],
        on="klub_slug",
        how="left",
        validate="1:1"
    ).merge(
        obs_team,
        on="team_name",
        how="left"
    )

    for c in ["official_xg", "official_xga", "observed_xg", "observed_xga"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if out["official_xg"].isna().any() or out["official_xga"].isna().any():
        missing = out.loc[out["official_xg"].isna() | out["official_xga"].isna(), ["team_name", "klub_slug"]]
        raise RuntimeError(
            "Brak official_xg/xga po joinie slugów:\n" + missing.to_string(index=False)
        )

    out["observed_xg"] = out["observed_xg"].fillna(0.0)
    out["observed_xga"] = out["observed_xga"].fillna(0.0)

    out["residual_attack"] = out["official_xg"] - out["observed_xg"]
    out["residual_defense"] = out["official_xga"] - out["observed_xga"]

    if (out["residual_attack"] < -1e-6).any():
        bad = out.loc[out["residual_attack"] < -1e-6, ["team_name", "residual_attack"]]
        raise RuntimeError("Ujemne residual_attack:\n" + bad.to_string(index=False))

    if (out["residual_defense"] < -1e-6).any():
        bad = out.loc[out["residual_defense"] < -1e-6, ["team_name", "residual_defense"]]
        raise RuntimeError("Ujemne residual_defense:\n" + bad.to_string(index=False))

    # wyrównanie globalne atak/obrona (na wypadek różnic zaokrągleń)
    sum_att = out["residual_attack"].sum()
    sum_def = out["residual_defense"].sum()
    target_total = (sum_att + sum_def) / 2.0

    scale_att = target_total / sum_att if sum_att > 0 else 1.0
    scale_def = target_total / sum_def if sum_def > 0 else 1.0

    out["residual_attack_adj"] = out["residual_attack"] * scale_att
    out["residual_defense_adj"] = out["residual_defense"] * scale_def

    return out, {
        "sum_attack_before": float(sum_att),
        "sum_defense_before": float(sum_def),
        "target_total": float(target_total),
        "scale_attack": float(scale_att),
        "scale_defense": float(scale_def),
    }


def solve_missing_xg(missing_matches, team_residuals):
    miss = missing_matches.copy().reset_index(drop=True)
    teams = team_residuals["team_name"].tolist()
    team_to_idx = {t: i for i, t in enumerate(teams)}

    attack_target = team_residuals.set_index("team_name")["residual_attack_adj"].to_dict()
    defense_target = team_residuals.set_index("team_name")["residual_defense_adj"].to_dict()

    M = len(miss)
    T = len(teams)

    # zmienne: [xg_home_0..M-1, xg_away_0..M-1]
    x0 = np.concatenate([
        miss["xg_gosp_proxy"].clip(lower=0.05).values,
        miss["xg_gosc_proxy"].clip(lower=0.05).values,
    ])

    proxy = x0.copy()
    scale = np.maximum(proxy, 0.30)

    A = np.zeros((2 * T, 2 * M), dtype=float)
    b = np.zeros(2 * T, dtype=float)

    # targety
    for team, idx in team_to_idx.items():
        b[idx] = attack_target[team]
        b[T + idx] = defense_target[team]

    # constraints
    for m, row in miss.iterrows():
        home = row["gospodarz"]
        away = row["gosc"]
        ih = team_to_idx[home]
        ia = team_to_idx[away]

        # attack(home) += xg_home_m
        A[ih, m] += 1.0
        # attack(away) += xg_away_m
        A[ia, M + m] += 1.0

        # defense(home) += xg_away_m
        A[T + ih, M + m] += 1.0
        # defense(away) += xg_home_m
        A[T + ia, m] += 1.0

    def objective(x):
        z = (x - proxy) / scale
        return float(np.mean(z ** 2))

    def gradient(x):
        return (2.0 / len(x)) * (x - proxy) / (scale ** 2)

    linear_constraint = LinearConstraint(A, b, b)
    bounds = Bounds(lb=np.full(2 * M, 0.05), ub=np.full(2 * M, np.inf))

    result = minimize(
        objective,
        x0=x0,
        jac=gradient,
        method="trust-constr",
        constraints=[linear_constraint],
        bounds=bounds,
        options={"maxiter": 5000, "verbose": 0}
    )

    if not result.success:
        raise RuntimeError(f"Optymalizacja nie powiodła się: {result.message}")

    sol = result.x
    miss["xg_gosp_imputed"] = sol[:M]
    miss["xg_gosc_imputed"] = sol[M:]

    return miss, result


def build_hybrid_dataset(merged, missing_solved):
    out = merged.copy()

    out["xg_gosp_hybrid"] = out["xg_gosp_obs"]
    out["xg_gosc_hybrid"] = out["xg_gosc_obs"]
    out["xg_source"] = np.where(out["has_observed_xg"] == 1, "observed_flash", "missing")

    if len(missing_solved):
        imputed = missing_solved[["match_id", "xg_gosp_imputed", "xg_gosc_imputed"]].copy()
        out = out.merge(imputed, on="match_id", how="left")

        mask = out["has_observed_xg"] == 0
        out.loc[mask, "xg_gosp_hybrid"] = out.loc[mask, "xg_gosp_imputed"]
        out.loc[mask, "xg_gosc_hybrid"] = out.loc[mask, "xg_gosc_imputed"]
        out.loc[mask, "xg_source"] = "imputed_constrained"

    return out


def build_team_check(hybrid, official):
    rows = []
    for _, row in hybrid.iterrows():
        rows.append({
            "team_name": row["gospodarz"],
            "klub_slug": TEAM_TO_SLUG[row["gospodarz"]],
            "hybrid_xg": row["xg_gosp_hybrid"],
            "hybrid_xga": row["xg_gosc_hybrid"],
        })
        rows.append({
            "team_name": row["gosc"],
            "klub_slug": TEAM_TO_SLUG[row["gosc"]],
            "hybrid_xg": row["xg_gosc_hybrid"],
            "hybrid_xga": row["xg_gosp_hybrid"],
        })

    team_hybrid = pd.DataFrame(rows).groupby(["team_name", "klub_slug"], as_index=False).agg(
        hybrid_xg=("hybrid_xg", "sum"),
        hybrid_xga=("hybrid_xga", "sum"),
    )

    check = team_hybrid.merge(
        official[["klub_slug", "official_xg", "official_xga"]],
        on="klub_slug",
        how="left",
        validate="1:1"
    )

    check["delta_xg"] = check["hybrid_xg"] - check["official_xg"]
    check["delta_xga"] = check["hybrid_xga"] - check["official_xga"]
    return check.sort_values("team_name").reset_index(drop=True)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEAM_CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuję matches 2023/24...")
    matches = load_matches()

    print("2. Wczytuję flash_2023_24.csv...")
    flash = load_flash()

    print("3. Wczytuję official team totals xG/xGA...")
    official = load_official_totals()

    print("4. Łączę observed xG po flash_id...")
    merged = merge_matches_flash(matches, flash)

    n_total = len(merged)
    n_observed = int(merged["has_observed_xg"].sum())
    n_missing = n_total - n_observed

    print(f"   Mecze total:    {n_total}")
    print(f"   Observed xG:    {n_observed}")
    print(f"   Missing xG:     {n_missing}")

    print("5. Fituję model proxy xG na observed meczach...")
    long_obs = build_long_observed(merged)
    proxy_model, proxy_info = fit_proxy_model(long_obs)

    print(f"   R2 proxy:  {proxy_info['r2']:.4f}")
    print(f"   MAE proxy: {proxy_info['mae']:.4f}")

    print("6. Wyznaczam residuale drużynowe...")
    team_residuals, residual_info = build_team_residuals(merged, official)

    print(f"   Sum attack residual before scaling:  {residual_info['sum_attack_before']:.4f}")
    print(f"   Sum defense residual before scaling: {residual_info['sum_defense_before']:.4f}")
    print(f"   Target total after scaling:          {residual_info['target_total']:.4f}")
    print(f"   Scale attack:                        {residual_info['scale_attack']:.6f}")
    print(f"   Scale defense:                       {residual_info['scale_defense']:.6f}")

    print("7. Predykuję proxy dla missing meczów...")
    missing = predict_missing_proxies(merged, proxy_model)

    print("8. Rozwiązuję constrained imputation...")
    missing_solved, opt_result = solve_missing_xg(missing, team_residuals)

    print(f"   Opt success: {opt_result.success}")
    print(f"   Opt fun:     {opt_result.fun:.6f}")
    print(f"   Opt nit:     {opt_result.nit}")

    print("9. Buduję finalny hybrid dataset...")
    hybrid = build_hybrid_dataset(merged, missing_solved)

    if hybrid["xg_gosp_hybrid"].isna().any() or hybrid["xg_gosc_hybrid"].isna().any():
        raise RuntimeError("Są NaNy w finalnym hybrid xG.")

    print("10. Team check vs official totals...")
    team_check = build_team_check(hybrid, official)

    max_abs_delta_xg = float(team_check["delta_xg"].abs().max())
    max_abs_delta_xga = float(team_check["delta_xga"].abs().max())

    # save outputs
    out_cols = [
        "match_id", "sezon", "kolejka", "data_meczu",
        "gospodarz", "gosc",
        "gole_gosp", "gole_gosc",
        "strzaly_gosp", "strzaly_gosc",
        "celne_gosp", "celne_gosc",
        "flash_id", "flash_url",
        "has_observed_xg", "xg_source",
        "xg_gosp_obs", "xg_gosc_obs",
        "xg_gosp_hybrid", "xg_gosc_hybrid",
    ]
    hybrid[out_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    team_check.to_csv(TEAM_CHECK_PATH, index=False, encoding="utf-8-sig")

    coef = proxy_model.coef_
    intercept = float(proxy_model.intercept_)

    lines = []
    lines.append("=" * 90)
    lines.append("HYBRID xG 2023/24 — RAPORT")
    lines.append("=" * 90)
    lines.append("")
    lines.append("1. COVERAGE")
    lines.append("-" * 90)
    lines.append(f"  Mecze total:                  {n_total}")
    lines.append(f"  Mecze observed_flash:         {n_observed}")
    lines.append(f"  Mecze imputed_constrained:    {n_missing}")
    lines.append(f"  Coverage observed:            {n_observed / n_total:.1%}")
    lines.append("")
    lines.append("2. PROXY MODEL")
    lines.append("-" * 90)
    lines.append("  Model: LinearRegression(positive=True)")
    lines.append(f"  intercept:                    {intercept:.6f}")
    lines.append(f"  coef_goals:                   {coef[0]:.6f}")
    lines.append(f"  coef_shots:                   {coef[1]:.6f}")
    lines.append(f"  coef_shots_on_target:         {coef[2]:.6f}")
    lines.append(f"  coef_blocked:                 {coef[3]:.6f}")
    lines.append(f"  R2 on observed team-rows:     {proxy_info['r2']:.4f}")
    lines.append(f"  MAE on observed team-rows:    {proxy_info['mae']:.4f}")
    lines.append("")
    lines.append("3. RESIDUAL TEAM TOTALS")
    lines.append("-" * 90)
    lines.append(f"  Sum attack residual before:   {residual_info['sum_attack_before']:.4f}")
    lines.append(f"  Sum defense residual before:  {residual_info['sum_defense_before']:.4f}")
    lines.append(f"  Target total after scaling:   {residual_info['target_total']:.4f}")
    lines.append(f"  Scale attack:                 {residual_info['scale_attack']:.6f}")
    lines.append(f"  Scale defense:                {residual_info['scale_defense']:.6f}")
    lines.append("")
    lines.append("4. OPTIMIZATION")
    lines.append("-" * 90)
    lines.append(f"  success:                      {opt_result.success}")
    lines.append(f"  objective:                    {opt_result.fun:.8f}")
    lines.append(f"  iterations:                   {opt_result.nit}")
    lines.append("")
    lines.append("5. TEAM CHECK vs OFFICIAL TOTALS")
    lines.append("-" * 90)
    lines.append(f"  max |delta_xg|:               {max_abs_delta_xg:.6f}")
    lines.append(f"  max |delta_xga|:              {max_abs_delta_xga:.6f}")
    lines.append("")
    lines.append("6. SAMPLE IMPUTED MATCHES")
    lines.append("-" * 90)
    sample_imp = hybrid[hybrid["xg_source"] == "imputed_constrained"][
        ["kolejka", "gospodarz", "gosc", "gole_gosp", "gole_gosc", "xg_gosp_hybrid", "xg_gosc_hybrid"]
    ].head(20)
    for row in sample_imp.itertuples(index=False):
        lines.append(
            f"  K{int(row.kolejka):02d} | {row.gospodarz:24s} vs {row.gosc:24s} | "
            f"gole {row.gole_gosp:.0f}:{row.gole_gosc:.0f} | "
            f"xG {row.xg_gosp_hybrid:.2f}:{row.xg_gosc_hybrid:.2f}"
        )
    lines.append("")
    lines.append("7. OUTPUT")
    lines.append("-" * 90)
    lines.append(f"  Hybrid matches:               {OUTPUT_PATH}")
    lines.append(f"  Team check:                   {TEAM_CHECK_PATH}")
    lines.append(f"  Report:                       {REPORT_PATH}")

    report_text = "\n".join(lines)
    print()
    print(report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {OUTPUT_PATH}")
    print(f"Zapisano: {TEAM_CHECK_PATH}")
    print(f"Zapisano: {REPORT_PATH}")


if __name__ == "__main__":
    main()