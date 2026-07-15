from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "ekstraklasa.db"

OUT_PROFILES_CSV = ROOT / "data" / "processed" / "referee_profiles.csv"
OUT_PROFILES_BY_SEASON_CSV = ROOT / "data" / "processed" / "referee_profiles_by_season.csv"
OUT_CAUSAL_CSV = ROOT / "data" / "processed" / "referee_causal_features.csv"
REPORT_PATH = ROOT / "data" / "reports" / "model" / "referee_profiles_report.txt"

K_PRIOR = 10

# Tylko dla absolutnego początku historii w bazie.
# Potem prior ligowy już idzie kauzalnie z rzeczywistych danych.
DEFAULT_LEAGUE_FOULS = 25.0
DEFAULT_LEAGUE_YC = 4.5
DEFAULT_LEAGUE_DISMISSALS = 0.20


def ensure_parent_dirs() -> None:
    OUT_PROFILES_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row[1] for row in rows]


def validate_schema(conn: sqlite3.Connection) -> None:
    matches_cols = set(get_table_columns(conn, "matches"))
    referees_cols = set(get_table_columns(conn, "match_referees"))

    required_matches = {
        "match_id",
        "sezon",
        "kolejka",
        "data_meczu",
        "faule_gosp",
        "faule_gosc",
        "zk_gosp",
        "zk_gosc",
        "czk_gosp",
        "czk_gosc",
        "druga_zk_gosp",
        "druga_zk_gosc",
    }
    required_referees = {
        "match_id",
        "referee_name",
        "referee_full_name",
    }

    missing_matches = sorted(required_matches - matches_cols)
    missing_referees = sorted(required_referees - referees_cols)

    if missing_matches or missing_referees:
        parts = []
        if missing_matches:
            parts.append("Brak kolumn w matches: " + ", ".join(missing_matches))
        if missing_referees:
            parts.append("Brak kolumn w match_referees: " + ", ".join(missing_referees))
        raise RuntimeError("Niepoprawny schemat bazy.\n" + "\n".join(parts))


def load_data(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT
            m.match_id,
            m.sezon,
            m.kolejka,
            m.data_meczu,
            r.referee_name,
            r.referee_full_name,
            m.faule_gosp,
            m.faule_gosc,
            m.zk_gosp,
            m.zk_gosc,
            COALESCE(m.czk_gosp, 0) AS czk_gosp,
            COALESCE(m.czk_gosc, 0) AS czk_gosc,
            COALESCE(m.druga_zk_gosp, 0) AS druga_zk_gosp,
            COALESCE(m.druga_zk_gosc, 0) AS druga_zk_gosc
        FROM matches m
        JOIN match_referees r
          ON m.match_id = r.match_id
        WHERE r.referee_full_name IS NOT NULL
          AND TRIM(r.referee_full_name) <> ''
          AND m.faule_gosp IS NOT NULL
          AND m.faule_gosc IS NOT NULL
          AND m.zk_gosp IS NOT NULL
          AND m.zk_gosc IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        raise RuntimeError("Brak danych po JOIN matches + match_referees.")

    # Porządkowanie czasu: najpierw próbujemy normalne parsowanie daty.
    df["data_dt"] = pd.to_datetime(df["data_meczu"], errors="coerce", dayfirst=True)

    # Stabilne sortowanie kauzalne
    df = df.sort_values(
        by=["data_dt", "sezon", "kolejka", "match_id"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    # Kolumny łączne
    df["fouls_total"] = df["faule_gosp"] + df["faule_gosc"]
    df["yc_total"] = df["zk_gosp"] + df["zk_gosc"]
    df["straight_red_total"] = df["czk_gosp"] + df["czk_gosc"]
    df["second_yellow_red_total"] = df["druga_zk_gosp"] + df["druga_zk_gosc"]
    df["dismissals_total"] = df["straight_red_total"] + df["second_yellow_red_total"]

    return df


def compute_overall_profiles(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("referee_full_name", dropna=False)

    out = grouped.agg(
        matches_count=("match_id", "count"),

        fouls_total_avg=("fouls_total", "mean"),
        fouls_home_avg=("faule_gosp", "mean"),
        fouls_away_avg=("faule_gosc", "mean"),
        fouls_total_std=("fouls_total", "std"),

        yc_total_avg=("yc_total", "mean"),
        yc_home_avg=("zk_gosp", "mean"),
        yc_away_avg=("zk_gosc", "mean"),
        yc_total_std=("yc_total", "std"),

        straight_red_total_avg=("straight_red_total", "mean"),
        second_yellow_red_total_avg=("second_yellow_red_total", "mean"),
        dismissals_total_avg=("dismissals_total", "mean"),
        dismissals_total_std=("dismissals_total", "std"),
    ).reset_index()

    out["fouls_total_std"] = out["fouls_total_std"].fillna(0.0)
    out["yc_total_std"] = out["yc_total_std"].fillna(0.0)
    out["dismissals_total_std"] = out["dismissals_total_std"].fillna(0.0)

    out["reliability_k10"] = out["matches_count"] / (out["matches_count"] + K_PRIOR)

    out = out.sort_values(
        by=["matches_count", "referee_full_name"],
        ascending=[False, True]
    ).reset_index(drop=True)

    numeric_cols = [c for c in out.columns if c not in {"referee_full_name"}]
    out[numeric_cols] = out[numeric_cols].round(4)

    return out


def compute_profiles_by_season(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["sezon", "referee_full_name"], dropna=False)

    out = grouped.agg(
        matches_count=("match_id", "count"),

        fouls_total_avg=("fouls_total", "mean"),
        fouls_home_avg=("faule_gosp", "mean"),
        fouls_away_avg=("faule_gosc", "mean"),

        yc_total_avg=("yc_total", "mean"),
        yc_home_avg=("zk_gosp", "mean"),
        yc_away_avg=("zk_gosc", "mean"),

        straight_red_total_avg=("straight_red_total", "mean"),
        second_yellow_red_total_avg=("second_yellow_red_total", "mean"),
        dismissals_total_avg=("dismissals_total", "mean"),
    ).reset_index()

    out["reliability_k10"] = out["matches_count"] / (out["matches_count"] + K_PRIOR)

    out = out.sort_values(
        by=["sezon", "matches_count", "referee_full_name"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    numeric_cols = [c for c in out.columns if c not in {"sezon", "referee_full_name"}]
    out[numeric_cols] = out[numeric_cols].round(4)

    return out


def compute_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Liga przed meczem
    out["league_match_index"] = np.arange(len(out))

    out["league_fouls_sum_before"] = out["fouls_total"].cumsum() - out["fouls_total"]
    out["league_yc_sum_before"] = out["yc_total"].cumsum() - out["yc_total"]
    out["league_dismissals_sum_before"] = out["dismissals_total"].cumsum() - out["dismissals_total"]

    out["league_fouls_avg_before"] = np.where(
        out["league_match_index"] > 0,
        out["league_fouls_sum_before"] / out["league_match_index"],
        DEFAULT_LEAGUE_FOULS,
    )
    out["league_yc_avg_before"] = np.where(
        out["league_match_index"] > 0,
        out["league_yc_sum_before"] / out["league_match_index"],
        DEFAULT_LEAGUE_YC,
    )
    out["league_dismissals_avg_before"] = np.where(
        out["league_match_index"] > 0,
        out["league_dismissals_sum_before"] / out["league_match_index"],
        DEFAULT_LEAGUE_DISMISSALS,
    )

    # Historia sędziego przed meczem
    out["ref_matches_before"] = out.groupby("referee_full_name").cumcount()

    out["ref_fouls_sum_before"] = out.groupby("referee_full_name")["fouls_total"].cumsum() - out["fouls_total"]
    out["ref_yc_sum_before"] = out.groupby("referee_full_name")["yc_total"].cumsum() - out["yc_total"]
    out["ref_dismissals_sum_before"] = (
        out.groupby("referee_full_name")["dismissals_total"].cumsum() - out["dismissals_total"]
    )

    # Raw averages przed meczem
    out["ref_fouls_raw_avg_before"] = np.where(
        out["ref_matches_before"] > 0,
        out["ref_fouls_sum_before"] / out["ref_matches_before"],
        out["league_fouls_avg_before"],
    )
    out["ref_yc_raw_avg_before"] = np.where(
        out["ref_matches_before"] > 0,
        out["ref_yc_sum_before"] / out["ref_matches_before"],
        out["league_yc_avg_before"],
    )
    out["ref_dismissals_raw_avg_before"] = np.where(
        out["ref_matches_before"] > 0,
        out["ref_dismissals_sum_before"] / out["ref_matches_before"],
        out["league_dismissals_avg_before"],
    )

    # Shrinkage do średniej ligowej
    out["ref_fouls_shrunk_before"] = (
        out["ref_fouls_sum_before"] + K_PRIOR * out["league_fouls_avg_before"]
    ) / (out["ref_matches_before"] + K_PRIOR)

    out["ref_yc_shrunk_before"] = (
        out["ref_yc_sum_before"] + K_PRIOR * out["league_yc_avg_before"]
    ) / (out["ref_matches_before"] + K_PRIOR)

    out["ref_dismissals_shrunk_before"] = (
        out["ref_dismissals_sum_before"] + K_PRIOR * out["league_dismissals_avg_before"]
    ) / (out["ref_matches_before"] + K_PRIOR)

    out["ref_reliability_k10"] = out["ref_matches_before"] / (out["ref_matches_before"] + K_PRIOR)

    keep_cols = [
        "match_id",
        "sezon",
        "kolejka",
        "data_meczu",
        "referee_name",
        "referee_full_name",

        "ref_matches_before",
        "ref_reliability_k10",

        "league_fouls_avg_before",
        "league_yc_avg_before",
        "league_dismissals_avg_before",

        "ref_fouls_raw_avg_before",
        "ref_yc_raw_avg_before",
        "ref_dismissals_raw_avg_before",

        "ref_fouls_shrunk_before",
        "ref_yc_shrunk_before",
        "ref_dismissals_shrunk_before",
    ]

    out = out[keep_cols].copy()

    numeric_cols = [
        "ref_matches_before",
        "ref_reliability_k10",
        "league_fouls_avg_before",
        "league_yc_avg_before",
        "league_dismissals_avg_before",
        "ref_fouls_raw_avg_before",
        "ref_yc_raw_avg_before",
        "ref_dismissals_raw_avg_before",
        "ref_fouls_shrunk_before",
        "ref_yc_shrunk_before",
        "ref_dismissals_shrunk_before",
    ]
    out[numeric_cols] = out[numeric_cols].round(4)

    return out


def build_report(
    profiles_df: pd.DataFrame,
    profiles_by_season_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    league_fouls_final = raw_df["fouls_total"].mean()
    league_yc_final = raw_df["yc_total"].mean()
    league_dismissals_final = raw_df["dismissals_total"].mean()

    lines: list[str] = []
    lines.append("REFEREE PROFILES REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {now}")
    lines.append(f"Mecze z profilem sędziego: {len(raw_df)}")
    lines.append(f"Liczba unikalnych sędziów: {profiles_df['referee_full_name'].nunique()}")
    lines.append("")
    lines.append("Konfiguracja:")
    lines.append(f"- K_PRIOR = {K_PRIOR}")
    lines.append(f"- DEFAULT_LEAGUE_FOULS = {DEFAULT_LEAGUE_FOULS}")
    lines.append(f"- DEFAULT_LEAGUE_YC = {DEFAULT_LEAGUE_YC}")
    lines.append(f"- DEFAULT_LEAGUE_DISMISSALS = {DEFAULT_LEAGUE_DISMISSALS}")
    lines.append("")
    lines.append("Średnie ligowe na pełnej próbie:")
    lines.append(f"- faule total: {league_fouls_final:.3f}")
    lines.append(f"- żółte kartki total: {league_yc_final:.3f}")
    lines.append(f"- wykluczenia total: {league_dismissals_final:.3f}")
    lines.append("")
    lines.append("Pliki wyjściowe:")
    lines.append(f"- {OUT_PROFILES_CSV}")
    lines.append(f"- {OUT_PROFILES_BY_SEASON_CSV}")
    lines.append(f"- {OUT_CAUSAL_CSV}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("TOP 15 sędziów wg liczby meczów — overall")
    lines.append("")

    top15 = profiles_df.head(15)
    for _, row in top15.iterrows():
        lines.append(
            f"{row['referee_full_name']}: "
            f"n={int(row['matches_count'])}, "
            f"fouls={row['fouls_total_avg']:.2f}, "
            f"yc={row['yc_total_avg']:.2f}, "
            f"dismissals={row['dismissals_total_avg']:.2f}, "
            f"rel={row['reliability_k10']:.3f}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Próbka causal features — ostatnie 10 meczów")
    lines.append("")

    tail = causal_df.tail(10)
    for _, row in tail.iterrows():
        lines.append(
            f"match_id={row['match_id']}, sezon={row['sezon']}, kolejka={row['kolejka']}, "
            f"ref={row['referee_full_name']}, n_before={int(row['ref_matches_before'])}, "
            f"fouls_shrunk={row['ref_fouls_shrunk_before']:.2f}, "
            f"yc_shrunk={row['ref_yc_shrunk_before']:.2f}, "
            f"dismissals_shrunk={row['ref_dismissals_shrunk_before']:.3f}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Sezony w profiles_by_season:")
    for sezon, n_rows in profiles_by_season_df.groupby("sezon").size().items():
        lines.append(f"- {sezon}: {n_rows} profili sędzia-sezon")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_parent_dirs()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono bazy: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        validate_schema(conn)
        raw_df = load_data(conn)

        profiles_df = compute_overall_profiles(raw_df)
        profiles_by_season_df = compute_profiles_by_season(raw_df)
        causal_df = compute_causal_features(raw_df)

        profiles_df.to_csv(OUT_PROFILES_CSV, index=False, encoding="utf-8-sig")
        profiles_by_season_df.to_csv(OUT_PROFILES_BY_SEASON_CSV, index=False, encoding="utf-8-sig")
        causal_df.to_csv(OUT_CAUSAL_CSV, index=False, encoding="utf-8-sig")

        build_report(
            profiles_df=profiles_df,
            profiles_by_season_df=profiles_by_season_df,
            causal_df=causal_df,
            raw_df=raw_df,
        )

        print("OK: zbudowano profile sedziowskie")
        print(f"CSV overall:      {OUT_PROFILES_CSV}")
        print(f"CSV by season:    {OUT_PROFILES_BY_SEASON_CSV}")
        print(f"CSV causal:       {OUT_CAUSAL_CSV}")
        print(f"Raport:           {REPORT_PATH}")
        print(f"Mecze:            {len(raw_df)}")
        print(f"Sedziowie:        {profiles_df['referee_full_name'].nunique()}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()