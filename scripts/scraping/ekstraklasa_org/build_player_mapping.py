import sqlite3
import pandas as pd
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_PATH = "db/ekstraklasa.db"
TARGET_SEASON = "2025/26"

REPORT_DIR = Path("data/reports/player_identity")
REVIEW_CSV = REPORT_DIR / "player_identity_review_2025_26.csv"
ROSTER_CSV = Path("data/processed/ekstra_player_roster_2023_2026.csv")

OUTPUT_MAPPING = Path("data/processed/player_mapping.csv")
OUTPUT_STATUS = Path("data/processed/player_mapping_status.csv")
OUTPUT_REPORT = REPORT_DIR / "player_mapping_report.txt"

TEAM_MAP = {
    "Arka Gdynia": "arka-gdynia",
    "Bruk-Bet Termalica Nieciecza": "nieciecza",
    "Cracovia": "cracovia",
    "GKS Katowice": "gks-katowice",
    "Górnik Zabrze": "gornik-zabrze",
    "Jagiellonia Białystok": "jagiellonia-bialystok",
    "Korona Kielce": "korona-kielce",
    "Lech Poznań": "lech-poznan",
    "Lechia Gdańsk": "lechia-gdansk",
    "Legia Warszawa": "legia-warszawa",
    "Motor Lublin": "motor-lublin",
    "Piast Gliwice": "piast-gliwice",
    "Pogoń Szczecin": "pogon-szczecin",
    "Radomiak Radom": "radomiak-radom",
    "Raków Częstochowa": "rakow-czestochowa",
    "Widzew Łódź": "widzew-lodz",
    "Wisła Płock": "wisla-plock",
    "Zagłębie Lubin": "zagebie-lubin",
}

MANUAL_ACCEPT = {
    ("Cracovia", "Skovgaard A."): (
        "andreas-skovgaard-larsen",
        "manual_accept_review_high_confidence",
    ),
    ("Jagiellonia Białystok", "Costa S. T."): (
        "tomas-costa-silva",
        "manual_accept_review_high_confidence",
    ),
    ("Legia Warszawa", "Gual M."): (
        "marc-gual-huguet",
        "manual_accept_review_high_confidence",
    ),
    ("Raków Częstochowa", "Silva J. C."): (
        "jean-carlos-silva-rocha",
        "manual_accept_review_needed",
    ),
}

MANUAL_REJECT = {
    ("Bruk-Bet Termalica Nieciecza", "Janicki M."):
        "manual_reject_wrong_person_rafa_janicki_is_not_m_janicki",
    ("Radomiak Radom", "Guilherme"):
        "manual_reject_no_current_ekstraklasa_org_data",
    ("Wisła Płock", "Zając F."):
        "manual_reject_wrong_person_jedrzej_zajac_is_not_f_zajac",
    ("Zagłębie Lubin", "Marek S."):
        "manual_reject_wrong_person_marek_mroz_is_not_s_marek",
    ("Zagłębie Lubin", "Urbański M."):
        "manual_reject_wrong_person_kacper_urbanski_is_not_m_urbanski",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERY
# ──────────────────────────────────────────────────────────────────────────────

def clean_flash_name(raw: str) -> str:
    raw = str(raw).strip()
    raw = re.sub(r"\s+\d+$", "", raw)
    return raw.strip()


def norm_text(x: str) -> str:
    return str(x).strip()


def load_flash_unique_players() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT DISTINCT player_name, team_name
        FROM lineups
        WHERE sezon = ?
        """,
        conn,
        params=(TARGET_SEASON,),
    )
    conn.close()

    df["flash_name"] = df["player_name"].apply(clean_flash_name)
    df["flash_team"] = df["team_name"].astype(str).str.strip()
    df["klub_slug"] = df["flash_team"].map(TEAM_MAP)

    df = df[["flash_team", "klub_slug", "flash_name"]].drop_duplicates()
    df = df.sort_values(["flash_team", "flash_name"]).reset_index(drop=True)
    return df


def load_review() -> pd.DataFrame:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(
            f"Brak pliku: {REVIEW_CSV}\n"
            f"Najpierw uruchom:\n"
            f"python scripts/audit/build_player_identity_review.py"
        )

    df = pd.read_csv(REVIEW_CSV)

    required = {
        "flash_team",
        "flash_name",
        "expected_klub_slug",
        "current_status",
        "existing_player_slug",
        "proposed_status",
        "proposed_player_slug",
        "evidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Brakuje kolumn w {REVIEW_CSV}: {sorted(missing)}")

    for col in df.columns:
        df[col] = df[col].fillna("")

    df["flash_team"] = df["flash_team"].astype(str).str.strip()
    df["flash_name"] = df["flash_name"].astype(str).apply(clean_flash_name)
    df["expected_klub_slug"] = df["expected_klub_slug"].astype(str).str.strip()
    df["current_status"] = df["current_status"].astype(str).str.strip()
    df["existing_player_slug"] = df["existing_player_slug"].astype(str).str.strip()
    df["proposed_status"] = df["proposed_status"].astype(str).str.strip()
    df["proposed_player_slug"] = df["proposed_player_slug"].astype(str).str.strip()
    df["evidence"] = df["evidence"].astype(str).str.strip()

    df = df.sort_values(["flash_team", "flash_name"]).reset_index(drop=True)
    return df


def load_roster_registry() -> dict:
    if not ROSTER_CSV.exists():
        raise FileNotFoundError(
            f"Brak pliku: {ROSTER_CSV}\n"
            f"Najpierw uruchom:\n"
            f"python scripts/audit/audit_player_identity_transfers.py"
        )

    roster = pd.read_csv(ROSTER_CSV)

    required = {"season", "player_slug", "klub_slug"}
    missing = required - set(roster.columns)
    if missing:
        raise RuntimeError(f"Brakuje kolumn w {ROSTER_CSV}: {sorted(missing)}")

    roster["season"] = roster["season"].astype(str).str.strip()
    roster["player_slug"] = roster["player_slug"].astype(str).str.strip()
    roster["klub_slug"] = roster["klub_slug"].astype(str).str.strip()
    if "nazwa" in roster.columns:
        roster["nazwa"] = roster["nazwa"].fillna("").astype(str)
    else:
        roster["nazwa"] = ""

    registry = {}

    for player_slug, g in roster.groupby("player_slug"):
        g = g.sort_values(["season", "klub_slug"]).copy()

        clubs_all = sorted(set(g["klub_slug"]))
        seasons_all = sorted(set(g["season"]))
        clubs_target = sorted(set(g.loc[g["season"] == TARGET_SEASON, "klub_slug"]))

        season_club_path = " | ".join(
            f"{row.season}:{row.klub_slug}"
            for row in g.itertuples(index=False)
        )

        registry[player_slug] = {
            "player_slug": player_slug,
            "clubs_all": clubs_all,
            "seasons_all": seasons_all,
            "clubs_target": clubs_target,
            "season_club_path": season_club_path,
            "sample_name": g["nazwa"].iloc[0] if len(g) else "",
        }

    return registry


def validate_review_vs_flash(flash_df: pd.DataFrame, review_df: pd.DataFrame) -> None:
    flash_keys = set(zip(flash_df["flash_team"], flash_df["flash_name"]))
    review_keys = set(zip(review_df["flash_team"], review_df["flash_name"]))

    missing_in_review = sorted(flash_keys - review_keys)
    extra_in_review = sorted(review_keys - flash_keys)

    if missing_in_review or extra_in_review:
        msg = ["Niespójność między DB lineups a player_identity_review_2025_26.csv"]
        if missing_in_review:
            msg.append("Brakuje w review:")
            for x in missing_in_review[:20]:
                msg.append(f"  {x[0]} | {x[1]}")
        if extra_in_review:
            msg.append("Nadwyżkowe w review:")
            for x in extra_in_review[:20]:
                msg.append(f"  {x[0]} | {x[1]}")
        raise RuntimeError("\n".join(msg))


def decide_row(row: pd.Series) -> tuple[str, str, str]:
    key = (row["flash_team"], row["flash_name"])
    proposed_status = row["proposed_status"]
    existing_player_slug = row["existing_player_slug"]
    proposed_player_slug = row["proposed_player_slug"]

    if proposed_status == "matched_existing":
        final_slug = existing_player_slug or proposed_player_slug
        return "matched_existing", final_slug, "existing_mapping"

    if proposed_status == "auto_same_club_history":
        return "matched_auto_same_club_history", proposed_player_slug, "review_auto_same_club_history"

    if proposed_status == "auto_transfer_candidate":
        return "matched_auto_transfer_candidate", proposed_player_slug, "review_auto_transfer_candidate"

    if key in MANUAL_ACCEPT:
        slug, method = MANUAL_ACCEPT[key]
        return "matched_manual_review", slug, method

    if key in MANUAL_REJECT:
        return "rejected_manual", "", MANUAL_REJECT[key]

    if proposed_status == "no_candidate":
        return "no_candidate", "", "review_no_candidate"

    return "unresolved_review", "", f"needs_manual_decision:{proposed_status}"


def enrich_with_registry(df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    clubs_all = []
    seasons_all = []
    clubs_target = []
    season_club_paths = []
    sample_names = []
    identity_in_target_club = []

    for row in df.itertuples(index=False):
        slug = getattr(row, "player_slug", "")
        if slug and slug in registry:
            rec = registry[slug]
            clubs_all.append(" | ".join(rec["clubs_all"]))
            seasons_all.append(" | ".join(rec["seasons_all"]))
            clubs_target.append(" | ".join(rec["clubs_target"]))
            season_club_paths.append(rec["season_club_path"])
            sample_names.append(rec["sample_name"])
            identity_in_target_club.append(int(row.klub_slug in rec["clubs_target"]))
        else:
            clubs_all.append("")
            seasons_all.append("")
            clubs_target.append("")
            season_club_paths.append("")
            sample_names.append("")
            identity_in_target_club.append("")

    out = df.copy()
    out["identity_clubs_all"] = clubs_all
    out["identity_seasons_all"] = seasons_all
    out["identity_target_season_clubs"] = clubs_target
    out["identity_season_club_path"] = season_club_paths
    out["identity_sample_name"] = sample_names
    out["identity_in_target_club"] = identity_in_target_club
    return out


def validate_accepted_slugs(status_df: pd.DataFrame, registry: dict) -> None:
    accepted = status_df[status_df["player_slug"] != ""].copy()
    missing = sorted(set(accepted["player_slug"]) - set(registry.keys()))
    if missing:
        raise RuntimeError(
            "Poniższe player_slug zostały zaakceptowane, ale nie istnieją w rosterze 3 sezonów:\n"
            + "\n".join(f"  {x}" for x in missing)
        )


def build_final_status(flash_df: pd.DataFrame, review_df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    merged = flash_df.merge(
        review_df,
        on=["flash_team", "flash_name"],
        how="left",
        suffixes=("", "_review"),
    )

    if merged["proposed_status"].isna().any():
        bad = merged[merged["proposed_status"].isna()][["flash_team", "flash_name"]]
        raise RuntimeError(
            "Brak review dla części zawodników:\n"
            + bad.to_string(index=False)
        )

    rows = []

    for row in merged.to_dict("records"):
        final_status, player_slug, match_method = decide_row(row)

        rows.append({
            "flash_team": row["flash_team"],
            "klub_slug": row["klub_slug"],
            "flash_name": row["flash_name"],

            "review_current_status": norm_text(row.get("current_status", "")),
            "review_existing_player_slug": norm_text(row.get("existing_player_slug", "")),
            "review_proposed_status": norm_text(row.get("proposed_status", "")),
            "review_proposed_player_slug": norm_text(row.get("proposed_player_slug", "")),
            "review_evidence": norm_text(row.get("evidence", "")),

            "final_status": final_status,
            "player_slug": player_slug,
            "match_method": match_method,
        })

    status_df = pd.DataFrame(rows)
    status_df = enrich_with_registry(status_df, registry)
    validate_accepted_slugs(status_df, registry)

    status_df = status_df.sort_values(["flash_team", "flash_name"]).reset_index(drop=True)
    return status_df


def build_mapping_output(status_df: pd.DataFrame) -> pd.DataFrame:
    matched_statuses = {
        "matched_existing",
        "matched_auto_same_club_history",
        "matched_auto_transfer_candidate",
        "matched_manual_review",
    }

    mapping_df = status_df[status_df["final_status"].isin(matched_statuses)].copy()

    mapping_df = mapping_df[
        ["flash_team", "klub_slug", "flash_name", "player_slug", "match_method"]
    ].sort_values(["flash_team", "flash_name"]).reset_index(drop=True)

    return mapping_df


def build_report(status_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("FINAL PLAYER MAPPING REPORT — FLASHSCORE <-> EKSTRAKLASA.ORG")
    lines.append("=" * 78)
    lines.append("")

    total = len(status_df)
    vc = status_df["final_status"].value_counts()

    lines.append("1. Statusy końcowe")
    lines.append("-" * 78)
    for status, cnt in vc.items():
        lines.append(f"  {status:32s} | {cnt:3d} | {100*cnt/total:5.1f}%")
    lines.append("")

    matched_mask = status_df["final_status"].str.startswith("matched_")
    matched = status_df[matched_mask]
    unmatched = status_df[~matched_mask]

    lines.append("2. Podsumowanie biznesowe")
    lines.append("-" * 78)
    lines.append(f"  Łącznie zawodników Flashscore 2025/26: {total}")
    lines.append(f"  Finalnie dopasowanych:                {len(matched)} ({100*len(matched)/total:.1f}%)")
    lines.append(f"  Bez dopasowania / odrzuconych:        {len(unmatched)} ({100*len(unmatched)/total:.1f}%)")
    lines.append("")

    lines.append("3. Akceptowane transfery")
    lines.append("-" * 78)
    auto_transfer = status_df[status_df["final_status"] == "matched_auto_transfer_candidate"]
    if len(auto_transfer):
        for row in auto_transfer.itertuples(index=False):
            lines.append(
                f"  {row.flash_team:30s} | {row.flash_name:22s} -> {row.player_slug:35s} | {row.identity_season_club_path}"
            )
    else:
        lines.append("  brak")
    lines.append("")

    lines.append("4. Ręcznie zaakceptowane review")
    lines.append("-" * 78)
    manual = status_df[status_df["final_status"] == "matched_manual_review"]
    if len(manual):
        for row in manual.itertuples(index=False):
            lines.append(
                f"  {row.flash_team:30s} | {row.flash_name:22s} -> {row.player_slug:35s} | {row.match_method}"
            )
    else:
        lines.append("  brak")
    lines.append("")

    lines.append("5. Ręcznie odrzucone")
    lines.append("-" * 78)
    rejected = status_df[status_df["final_status"] == "rejected_manual"]
    if len(rejected):
        for row in rejected.itertuples(index=False):
            lines.append(
                f"  {row.flash_team:30s} | {row.flash_name:22s} | {row.match_method}"
            )
    else:
        lines.append("  brak")
    lines.append("")

    lines.append("6. Brak kandydata")
    lines.append("-" * 78)
    no_cand = status_df[status_df["final_status"] == "no_candidate"]
    if len(no_cand):
        for row in no_cand.itertuples(index=False):
            lines.append(f"  {row.flash_team:30s} | {row.flash_name}")
    else:
        lines.append("  brak")
    lines.append("")

    lines.append("7. UWAGA DO DALSZEGO JOINA")
    lines.append("-" * 78)
    lines.append("  Dla zaakceptowanych transferów downstream join ma być po player_slug,")
    lines.append("  NIE po (player_slug, klub_slug).")
    lines.append("")

    lines.append("8. Zapisane pliki")
    lines.append("-" * 78)
    lines.append(f"  {OUTPUT_MAPPING}")
    lines.append(f"  {OUTPUT_STATUS}")
    lines.append(f"  {OUTPUT_REPORT}")
    lines.append("")

    return "\n".join(lines)


def main():
    flash_df = load_flash_unique_players()
    review_df = load_review()
    registry = load_roster_registry()

    validate_review_vs_flash(flash_df, review_df)

    status_df = build_final_status(flash_df, review_df, registry)
    mapping_df = build_mapping_output(status_df)
    report = build_report(status_df)

    OUTPUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    mapping_df.to_csv(OUTPUT_MAPPING, index=False, encoding="utf-8")
    status_df.to_csv(OUTPUT_STATUS, index=False, encoding="utf-8")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()