import sqlite3
from pathlib import Path
import re
import pandas as pd
import numpy as np

DB_PATH = Path("db/ekstraklasa.db")
FLASH_PATH = Path("data/raw/flash/flash_2023_24.csv")
TEAM_XG_PATH = Path("data/raw/ekstraklasa_org/druzyny/2023-2024_druzynowe-xg.csv")
TEAM_XGA_PATH = Path("data/raw/ekstraklasa_org/druzyny/2023-2024_druzynowe-xga.csv")

REPORT_DIR = Path("data/reports/model")
TEAM_REPORT_CSV = REPORT_DIR / "audit_2023_24_xg_team_residuals.csv"
MISSING_MATCHES_CSV = REPORT_DIR / "audit_2023_24_xg_missing_matches.csv"
REPORT_TXT = REPORT_DIR / "audit_2023_24_xg_residuals.txt"


PL_TO_ASCII = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})


def norm(s: str) -> str:
    s = str(s).translate(PL_TO_ASCII).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def load_matches_2023_24():
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
            flash_id,
            flash_url
        FROM matches
        WHERE sezon = '2023/24'
        ORDER BY kolejka, match_id
    """, conn)
    conn.close()
    return df


def load_official_team_totals():
    df_xg = safe_read_csv(TEAM_XG_PATH).copy()
    df_xga = safe_read_csv(TEAM_XGA_PATH).copy()

    required = {"nazwa", "wartosc"}
    missing_xg = required - set(df_xg.columns)
    missing_xga = required - set(df_xga.columns)
    if missing_xg:
        raise RuntimeError(f"Brak kolumn w {TEAM_XG_PATH}: {sorted(missing_xg)}")
    if missing_xga:
        raise RuntimeError(f"Brak kolumn w {TEAM_XGA_PATH}: {sorted(missing_xga)}")

    df_xg = df_xg[["nazwa", "wartosc"]].copy()
    df_xga = df_xga[["nazwa", "wartosc"]].copy()

    df_xg["official_xg"] = pd.to_numeric(df_xg["wartosc"], errors="coerce")
    df_xga["official_xga"] = pd.to_numeric(df_xga["wartosc"], errors="coerce")

    df_xg["team_norm"] = df_xg["nazwa"].map(norm)
    df_xga["team_norm"] = df_xga["nazwa"].map(norm)

    official = df_xg[["nazwa", "team_norm", "official_xg"]].merge(
        df_xga[["team_norm", "official_xga"]],
        on="team_norm",
        how="inner"
    )

    official = official.rename(columns={"nazwa": "official_team_name"})
    return official


def build_team_summary(matches, merged, official):
    # wszystkie drużyny z sezonu
    teams = sorted(set(matches["gospodarz"]) | set(matches["gosc"]))
    team_df = pd.DataFrame({"team_name": teams})
    team_df["team_norm"] = team_df["team_name"].map(norm)

    # liczba meczów total
    rows_total = []
    for _, row in matches.iterrows():
        rows_total.append({"team_name": row["gospodarz"], "n_matches_total": 1})
        rows_total.append({"team_name": row["gosc"], "n_matches_total": 1})
    total_counts = pd.DataFrame(rows_total).groupby("team_name", as_index=False).sum()

    # znane xG
    known = merged[merged["has_observed_xg"] == 1].copy()

    rows_known = []
    for _, row in known.iterrows():
        rows_known.append({
            "team_name": row["gospodarz"],
            "observed_xg_known": row["xg_gosp_flash"],
            "observed_xga_known": row["xg_gosc_flash"],
            "n_matches_known": 1,
        })
        rows_known.append({
            "team_name": row["gosc"],
            "observed_xg_known": row["xg_gosc_flash"],
            "observed_xga_known": row["xg_gosp_flash"],
            "n_matches_known": 1,
        })

    known_team = pd.DataFrame(rows_known).groupby("team_name", as_index=False).agg(
        observed_xg_known=("observed_xg_known", "sum"),
        observed_xga_known=("observed_xga_known", "sum"),
        n_matches_known=("n_matches_known", "sum"),
    )

    # brakujące mecze
    missing = merged[merged["has_observed_xg"] == 0].copy()
    rows_missing = []
    for _, row in missing.iterrows():
        rows_missing.append({"team_name": row["gospodarz"], "n_matches_missing": 1})
        rows_missing.append({"team_name": row["gosc"], "n_matches_missing": 1})

    missing_team = pd.DataFrame(rows_missing).groupby("team_name", as_index=False).sum()

    summary = (
        team_df
        .merge(official, on="team_norm", how="left")
        .merge(total_counts, on="team_name", how="left")
        .merge(known_team, on="team_name", how="left")
        .merge(missing_team, on="team_name", how="left")
    )

    for col in [
        "n_matches_total",
        "n_matches_known",
        "n_matches_missing",
        "observed_xg_known",
        "observed_xga_known",
    ]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)

    summary["official_xg"] = pd.to_numeric(summary["official_xg"], errors="coerce")
    summary["official_xga"] = pd.to_numeric(summary["official_xga"], errors="coerce")

    summary["residual_xg_to_impute"] = summary["official_xg"] - summary["observed_xg_known"]
    summary["residual_xga_to_impute"] = summary["official_xga"] - summary["observed_xga_known"]

    summary["known_share_xg"] = np.where(
        summary["official_xg"] > 0,
        summary["observed_xg_known"] / summary["official_xg"],
        np.nan
    )
    summary["known_share_xga"] = np.where(
        summary["official_xga"] > 0,
        summary["observed_xga_known"] / summary["official_xga"],
        np.nan
    )

    summary["flag_negative_residual_xg"] = (summary["residual_xg_to_impute"] < 0).astype(int)
    summary["flag_negative_residual_xga"] = (summary["residual_xga_to_impute"] < 0).astype(int)

    summary["official_team_name"] = summary["official_team_name"].fillna("")
    summary["mapping_ok"] = (summary["official_team_name"] != "").astype(int)

    summary = summary.sort_values("team_name").reset_index(drop=True)
    return summary


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuję matches 2023/24...")
    matches = load_matches_2023_24()

    print("2. Wczytuję flash_2023_24.csv...")
    flash = safe_read_csv(FLASH_PATH).copy()

    required_flash = {"flash_id", "xg_gosp", "xg_gosc", "url"}
    missing_flash = required_flash - set(flash.columns)
    if missing_flash:
        raise RuntimeError(f"Brak wymaganych kolumn w {FLASH_PATH}: {sorted(missing_flash)}")

    flash["flash_id"] = flash["flash_id"].astype(str).str.strip()
    flash["xg_gosp"] = pd.to_numeric(flash["xg_gosp"], errors="coerce")
    flash["xg_gosc"] = pd.to_numeric(flash["xg_gosc"], errors="coerce")

    if flash["flash_id"].duplicated().any():
        dups = flash.loc[flash["flash_id"].duplicated(), "flash_id"].tolist()[:10]
        raise RuntimeError(f"Duplikaty flash_id w {FLASH_PATH}: {dups}")

    print("3. Łączę po flash_id...")
    matches["flash_id"] = matches["flash_id"].astype(str).str.strip()
    merged = matches.merge(
        flash[[
            "flash_id", "url", "xg_gosp", "xg_gosc",
            "strzaly_gosp", "strzaly_gosc", "celne_gosp", "celne_gosc"
        ]].rename(columns={
            "url": "flash_url_raw",
            "xg_gosp": "xg_gosp_flash",
            "xg_gosc": "xg_gosc_flash",
            "strzaly_gosp": "strzaly_gosp_flash",
            "strzaly_gosc": "strzaly_gosc_flash",
            "celne_gosp": "celne_gosp_flash",
            "celne_gosc": "celne_gosc_flash",
        }),
        on="flash_id",
        how="left",
        validate="1:1"
    )

    merged["has_observed_xg"] = (
        merged["xg_gosp_flash"].notna() & merged["xg_gosc_flash"].notna()
    ).astype(int)

    print("4. Wczytuję oficjalne sezonowe totale xG/xGA...")
    official = load_official_team_totals()

    print("5. Buduję summary drużynowe...")
    summary = build_team_summary(matches, merged, official)

    # brakujące mecze
    missing_matches = merged[merged["has_observed_xg"] == 0].copy()
    missing_matches = missing_matches[[
        "match_id", "kolejka", "data_meczu", "gospodarz", "gosc",
        "gole_gosp", "gole_gosc",
        "strzaly_gosp", "strzaly_gosc", "celne_gosp", "celne_gosc",
        "flash_id", "flash_url",
        "strzaly_gosp_flash", "strzaly_gosc_flash", "celne_gosp_flash", "celne_gosc_flash",
        "xg_gosp_flash", "xg_gosc_flash"
    ]].copy()

    # save csv
    summary.to_csv(TEAM_REPORT_CSV, index=False, encoding="utf-8-sig")
    missing_matches.to_csv(MISSING_MATCHES_CSV, index=False, encoding="utf-8-sig")

    # global stats
    observed_matches = int(merged["has_observed_xg"].sum())
    missing_matches_n = int((merged["has_observed_xg"] == 0).sum())

    observed_total_match_xg = float(
        merged.loc[merged["has_observed_xg"] == 1, ["xg_gosp_flash", "xg_gosc_flash"]].sum().sum()
    )

    official_total_xg = float(summary["official_xg"].sum())
    official_total_xga = float(summary["official_xga"].sum())

    observed_total_team_xg = float(summary["observed_xg_known"].sum())
    observed_total_team_xga = float(summary["observed_xga_known"].sum())

    residual_total_xg = float(summary["residual_xg_to_impute"].sum())
    residual_total_xga = float(summary["residual_xga_to_impute"].sum())

    unmatched_teams = summary[summary["mapping_ok"] == 0]["team_name"].tolist()
    neg_xg = int(summary["flag_negative_residual_xg"].sum())
    neg_xga = int(summary["flag_negative_residual_xga"].sum())

    lines = []
    lines.append("=" * 90)
    lines.append("AUDYT 2023/24 — OBSERVED MATCH xG vs OFFICIAL SEASON TOTALS")
    lines.append("=" * 90)
    lines.append("")
    lines.append("1. COVERAGE")
    lines.append("-" * 90)
    lines.append(f"  Mecze 2023/24 w DB:                  {len(matches)}")
    lines.append(f"  Mecze z observed xG z Flashscore:    {observed_matches}")
    lines.append(f"  Mecze bez xG do imputacji:           {missing_matches_n}")
    lines.append(f"  Coverage observed xG:                {observed_matches / len(matches):.1%}")
    lines.append("")
    lines.append("2. SUMY LIGOWE")
    lines.append("-" * 90)
    lines.append(f"  Official total xG (teams):           {official_total_xg:.2f}")
    lines.append(f"  Official total xGA (teams):          {official_total_xga:.2f}")
    lines.append(f"  Observed known total xG (team view): {observed_total_team_xg:.2f}")
    lines.append(f"  Observed known total xGA (team view):{observed_total_team_xga:.2f}")
    lines.append(f"  Residual xG to impute:               {residual_total_xg:.2f}")
    lines.append(f"  Residual xGA to impute:              {residual_total_xga:.2f}")
    lines.append(f"  Match-level observed xG sum:         {observed_total_match_xg:.2f}")
    lines.append("")
    lines.append("3. KOMPATYBILNOŚĆ ŹRÓDEŁ")
    lines.append("-" * 90)
    lines.append(f"  Drużyn bez mapowania do official:    {len(unmatched_teams)}")
    lines.append(f"  Drużyn z residual_xG < 0:            {neg_xg}")
    lines.append(f"  Drużyn z residual_xGA < 0:           {neg_xga}")
    if unmatched_teams:
        lines.append(f"  Lista bez mapowania: {', '.join(unmatched_teams)}")
    lines.append("")
    lines.append("4. TOP DRUŻYNY — RESIDUAL XG DO IMPUTACJI")
    lines.append("-" * 90)
    top_xg = summary.sort_values("residual_xg_to_impute", ascending=False)[[
        "team_name", "official_xg", "observed_xg_known", "residual_xg_to_impute",
        "n_matches_known", "n_matches_missing", "known_share_xg", "flag_negative_residual_xg"
    ]]
    for row in top_xg.head(18).itertuples(index=False):
        lines.append(
            f"  {row.team_name:28s} | official_xg={row.official_xg:6.2f} | "
            f"known={row.observed_xg_known:6.2f} | residual={row.residual_xg_to_impute:6.2f} | "
            f"known_m={int(row.n_matches_known):2d} | missing_m={int(row.n_matches_missing):2d} | "
            f"share={row.known_share_xg:5.1%} | neg={int(row.flag_negative_residual_xg)}"
        )
    lines.append("")
    lines.append("5. TOP DRUŻYNY — RESIDUAL XGA DO IMPUTACJI")
    lines.append("-" * 90)
    top_xga = summary.sort_values("residual_xga_to_impute", ascending=False)[[
        "team_name", "official_xga", "observed_xga_known", "residual_xga_to_impute",
        "n_matches_known", "n_matches_missing", "known_share_xga", "flag_negative_residual_xga"
    ]]
    for row in top_xga.head(18).itertuples(index=False):
        lines.append(
            f"  {row.team_name:28s} | official_xga={row.official_xga:6.2f} | "
            f"known={row.observed_xga_known:6.2f} | residual={row.residual_xga_to_impute:6.2f} | "
            f"known_m={int(row.n_matches_known):2d} | missing_m={int(row.n_matches_missing):2d} | "
            f"share={row.known_share_xga:5.1%} | neg={int(row.flag_negative_residual_xga)}"
        )
    lines.append("")
    lines.append("6. PLIKI")
    lines.append("-" * 90)
    lines.append(f"  Team residuals CSV: {TEAM_REPORT_CSV}")
    lines.append(f"  Missing matches CSV:{MISSING_MATCHES_CSV}")
    lines.append(f"  Report TXT:         {REPORT_TXT}")

    report_text = "\n".join(lines)
    print()
    print(report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {TEAM_REPORT_CSV}")
    print(f"Zapisano: {MISSING_MATCHES_CSV}")
    print(f"Zapisano: {REPORT_TXT}")


if __name__ == "__main__":
    main()