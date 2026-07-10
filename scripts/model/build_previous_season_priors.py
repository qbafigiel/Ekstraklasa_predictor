import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports/model")

SEASON_TO_LABEL = {
    "2023/24": "2023_24",
    "2024/25": "2024_25",
    "2025/26": "2025_26",
}

PREVIOUS_SEASON = {
    "2024/25": "2023/24",
    "2025/26": "2024/25",
}

PL_TO_ASCII = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})


def to_ascii(s):
    s = str(s).translate(PL_TO_ASCII).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def get_label(season):
    return SEASON_TO_LABEL[season]


def get_mapping_status_path(season):
    label = get_label(season)
    specific = PROCESSED_DIR / f"player_mapping_status_{label}.csv"
    legacy = PROCESSED_DIR / "player_mapping_status.csv"

    if specific.exists():
        return specific
    if season == "2025/26" and legacy.exists():
        return legacy
    return specific


def get_rankings_path(season):
    return PROCESSED_DIR / f"zawodnicy_ekstraklasa_org_{get_label(season)}.csv"


def get_output_path(target_season):
    return PROCESSED_DIR / f"player_priors_{get_label(target_season)}.csv"


def get_report_path(target_season):
    return REPORT_DIR / f"player_priors_report_{get_label(target_season)}.txt"


def detect_minutes_column(df):
    if df.empty:
        return None

    normalized = {col: to_ascii(col) for col in df.columns}

    exact_preference = [
        "minuty",
        "minutes",
        "czasgry",
        "rozegraneminuty",
        "liczbaminut",
        "min",
    ]
    for wanted in exact_preference:
        for col, norm in normalized.items():
            if norm == wanted:
                return col

    contains_preference = [
        "minut",
        "minutes",
        "czasgry",
    ]
    for wanted in contains_preference:
        for col, norm in normalized.items():
            if wanted in norm:
                return col

    return None


def load_mapping_status(target_season):
    path = get_mapping_status_path(target_season)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku mapping status: {path}")

    df = pd.read_csv(path)
    required = {"flash_team", "flash_name", "final_status", "player_slug"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Brak wymaganych kolumn w {path}: {sorted(missing)}")

    for col in ["flash_team", "flash_name", "final_status", "player_slug"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df, path


def load_previous_rankings(prev_season):
    path = get_rankings_path(prev_season)
    if not path.exists():
        raise FileNotFoundError(f"Brak rankingu poprzedniego sezonu: {path}")

    df = pd.read_csv(path)
    if "player_slug" not in df.columns:
        raise RuntimeError(f"Brak kolumny player_slug w {path}")

    df["player_slug"] = df["player_slug"].fillna("").astype(str).str.strip()
    df = df[df["player_slug"] != ""].copy()

    raw_rows = len(df)
    raw_unique = df["player_slug"].nunique()
    duplicate_rows = raw_rows - raw_unique

    minutes_col = detect_minutes_column(df)

    df["_non_null_count"] = df.notna().sum(axis=1)

    if minutes_col is not None:
        df["_minutes_sort"] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(-1)
    else:
        df["_minutes_sort"] = -1

    df = df.sort_values(
        ["player_slug", "_minutes_sort", "_non_null_count"],
        ascending=[True, False, False]
    ).copy()

    dedup = df.drop_duplicates(subset=["player_slug"], keep="first").copy()
    dedup = dedup.drop(columns=["_non_null_count", "_minutes_sort"], errors="ignore")

    info = {
        "path": path,
        "raw_rows": raw_rows,
        "raw_unique_player_slugs": raw_unique,
        "duplicate_rows_removed": duplicate_rows,
        "minutes_column_used_for_dedupe": minutes_col or "",
        "final_rows_after_dedupe": len(dedup),
    }
    return dedup, info


def rename_prior_columns(prev_rankings):
    renamed = prev_rankings.copy()

    rename_map = {}
    for col in renamed.columns:
        if col == "player_slug":
            continue
        if col == "klub_slug":
            rename_map[col] = "prior_prev_klub_slug"
        elif col == "nazwa":
            rename_map[col] = "prior_prev_name"
        else:
            rename_map[col] = f"prior_{col}"

    renamed = renamed.rename(columns=rename_map)
    return renamed, rename_map


def build_one_season(target_season):
    if target_season not in PREVIOUS_SEASON:
        raise RuntimeError(
            f"Nieobsługiwany target season: {target_season}. "
            f"Dostępne: {sorted(PREVIOUS_SEASON.keys())}"
        )

    prev_season = PREVIOUS_SEASON[target_season]
    output_path = get_output_path(target_season)
    report_path = get_report_path(target_season)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mapping_df, mapping_path = load_mapping_status(target_season)
    prev_rankings_df, prev_info = load_previous_rankings(prev_season)
    prev_rankings_renamed, rename_map = rename_prior_columns(prev_rankings_df)

    merged = mapping_df.merge(
        prev_rankings_renamed,
        on="player_slug",
        how="left",
        indicator=True,
    )

    merged["prior_available"] = (merged["_merge"] == "both").astype(int)
    merged["prior_source_season"] = np.where(
        merged["prior_available"] == 1,
        prev_season,
        ""
    )

    merged["prior_missing_reason"] = ""
    merged.loc[merged["player_slug"] == "", "prior_missing_reason"] = "unmapped_flash_player"
    merged.loc[
        (merged["player_slug"] != "") & (merged["prior_available"] == 0),
        "prior_missing_reason"
    ] = "mapped_but_no_previous_season_ranking"

    merged = merged.drop(columns=["_merge"])

    prior_metric_cols = [c for c in merged.columns if c.startswith("prior_")]
    helper_priority = [
        "prior_available",
        "prior_source_season",
        "prior_missing_reason",
        "prior_prev_name",
        "prior_prev_klub_slug",
    ]
    helper_existing = [c for c in helper_priority if c in merged.columns]
    other_prior_cols = [c for c in prior_metric_cols if c not in helper_existing]

    base_cols_preferred = [
        "flash_team",
        "klub_slug",
        "flash_name",
        "review_current_status",
        "review_existing_player_slug",
        "review_proposed_status",
        "review_proposed_player_slug",
        "review_evidence",
        "final_status",
        "player_slug",
        "match_method",
        "identity_clubs_all",
        "identity_seasons_all",
        "identity_target_season_clubs",
        "identity_season_club_path",
        "identity_sample_name",
        "identity_in_target_club",
    ]
    base_cols = [c for c in base_cols_preferred if c in merged.columns]

    ordered_cols = base_cols + helper_existing + sorted(other_prior_cols)
    merged = merged[ordered_cols].copy()

    merged.to_csv(output_path, index=False, encoding="utf-8")

    matched_mask = merged["final_status"].astype(str).str.startswith("matched_")
    matched_total = int(matched_mask.sum())
    matched_with_prior = int(((matched_mask) & (merged["prior_available"] == 1)).sum())
    matched_without_prior = matched_total - matched_with_prior

    total_rows = len(merged)
    total_with_prior = int((merged["prior_available"] == 1).sum())
    total_without_prior = total_rows - total_with_prior

    lines = []
    lines.append("=" * 78)
    lines.append(f"PREVIOUS-SEASON PRIORS REPORT [{target_season} <- {prev_season}]")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. INPUT")
    lines.append("-" * 78)
    lines.append(f"  Mapping status: {mapping_path}")
    lines.append(f"  Previous rankings: {prev_info['path']}")
    lines.append("")
    lines.append("2. RANKINGS DEDUPE")
    lines.append("-" * 78)
    lines.append(f"  Raw rows:                    {prev_info['raw_rows']}")
    lines.append(f"  Unique player_slug raw:      {prev_info['raw_unique_player_slugs']}")
    lines.append(f"  Duplicate rows removed:      {prev_info['duplicate_rows_removed']}")
    lines.append(f"  Minutes column used:         {prev_info['minutes_column_used_for_dedupe'] or 'brak'}")
    lines.append(f"  Final rows after dedupe:     {prev_info['final_rows_after_dedupe']}")
    lines.append("")
    lines.append("3. COVERAGE")
    lines.append("-" * 78)
    lines.append(f"  Total flash identities:      {total_rows}")
    lines.append(f"  With prior stats:            {total_with_prior}")
    lines.append(f"  Without prior stats:         {total_without_prior}")
    lines.append("")
    lines.append(f"  Matched identities:          {matched_total}")
    lines.append(f"  Matched with prior stats:    {matched_with_prior}")
    lines.append(f"  Matched without prior stats: {matched_without_prior}")
    if matched_total > 0:
        lines.append(f"  Matched prior coverage:      {matched_with_prior / matched_total:.1%}")
    lines.append("")
    lines.append("4. FINAL STATUS COUNTS")
    lines.append("-" * 78)
    status_counts = merged["final_status"].value_counts()
    for status, count in status_counts.items():
        lines.append(f"  {status:32s} | {int(count):4d}")
    lines.append("")
    lines.append("5. OUTPUT")
    lines.append("-" * 78)
    lines.append(f"  {output_path}")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nZapisano raport: {report_path}")

    return {
        "target_season": target_season,
        "previous_season": prev_season,
        "total_rows": total_rows,
        "matched_total": matched_total,
        "matched_with_prior": matched_with_prior,
        "matched_prior_coverage": (matched_with_prior / matched_total) if matched_total else 0.0,
        "output_path": str(output_path),
        "report_path": str(report_path),
    }


def build_all():
    summaries = []
    for season in ["2024/25", "2025/26"]:
        summaries.append(build_one_season(season))
        print("\n")

    print("=" * 78)
    print("SUMMARY — ALL SEASONS")
    print("=" * 78)
    for s in summaries:
        print(
            f"{s['target_season']} <- {s['previous_season']} | "
            f"matched={s['matched_total']} | "
            f"with_prior={s['matched_with_prior']} | "
            f"coverage={s['matched_prior_coverage']:.1%}"
        )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", choices=["2024/25", "2025/26"])
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        build_all()
    else:
        build_one_season(args.season)


if __name__ == "__main__":
    main()