import sqlite3
import re
from pathlib import Path

import pandas as pd

DB_PATH = "db/ekstraklasa.db"
LINEUP_VALUES_PATH = Path("data/processed/match_lineup_values.csv")
REPORT_DIR = Path("data/reports/model")
REPORT_PATH = REPORT_DIR / "test_lineup_correlation_report.txt"


PL_TO_ASCII = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})


def norm(s: str) -> str:
    s = str(s).translate(PL_TO_ASCII).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def detect_column(columns, candidates_exact=(), candidates_contains=()):
    normalized = {col: norm(col) for col in columns}

    for wanted in candidates_exact:
        wanted_n = norm(wanted)
        for col, col_n in normalized.items():
            if col_n == wanted_n:
                return col

    for wanted in candidates_contains:
        wanted_n = norm(wanted)
        for col, col_n in normalized.items():
            if wanted_n in col_n:
                return col

    return None


def detect_match_columns(match_cols):
    match_id_col = detect_column(
        match_cols,
        candidates_exact=["match_id", "id_meczu", "mecz_id"],
        candidates_contains=["match_id", "id_meczu", "mecz_id"],
    )

    home_goals_col = detect_column(
        match_cols,
        candidates_exact=[
            "home_goals", "gole_gospodarze", "gospodarze_gole", "goals_home",
            "wynik_gospodarze", "bramki_gospodarze", "gole_gosp", "bramki_gosp"
        ],
        candidates_contains=[
            "home_goals", "gole_gospodarze", "gospodarze_gole",
            "wynik_gospodarze", "bramki_gospodarze", "gole_gosp", "bramki_gosp"
        ],
    )

    away_goals_col = detect_column(
        match_cols,
        candidates_exact=[
            "away_goals", "gole_goscie", "goscie_gole", "goals_away",
            "wynik_gosci", "bramki_gosci", "gole_gosc", "bramki_gosc"
        ],
        candidates_contains=[
            "away_goals", "gole_gosci", "goscie_gole",
            "wynik_gosci", "bramki_gosci", "gole_gosc", "bramki_gosc"
        ],
    )

    home_xg_col = detect_column(
        match_cols,
        candidates_exact=[
            "home_xg", "xg_gospodarze", "gospodarze_xg", "xg_home", "xg_gosp"
        ],
        candidates_contains=[
            "home_xg", "xg_gospodarze", "gospodarze_xg", "xg_home", "xg_gosp"
        ],
    )

    away_xg_col = detect_column(
        match_cols,
        candidates_exact=[
            "away_xg", "xg_goscie", "goscie_xg", "xg_away", "xg_gosc"
        ],
        candidates_contains=[
            "away_xg", "xg_goscie", "goscie_xg", "xg_away", "xg_gosc"
        ],
    )

    missing = []
    if match_id_col is None:
        missing.append("match_id")
    if home_goals_col is None:
        missing.append("home_goals")
    if away_goals_col is None:
        missing.append("away_goals")
    if home_xg_col is None:
        missing.append("home_xg")
    if away_xg_col is None:
        missing.append("away_xg")

    if missing:
        raise RuntimeError(
            "Nie udało się wykryć kolumn w tabeli matches: "
            + ", ".join(missing)
            + "\nDostępne kolumny:\n  - "
            + "\n  - ".join(match_cols)
        )

    return {
        "match_id": match_id_col,
        "home_goals": home_goals_col,
        "away_goals": away_goals_col,
        "home_xg": home_xg_col,
        "away_xg": away_xg_col,
    }


def safe_corr(a, b):
    s = pd.Series(a)
    t = pd.Series(b)
    if len(s) < 3:
        return float("nan")
    if s.nunique(dropna=True) <= 1 or t.nunique(dropna=True) <= 1:
        return float("nan")
    return s.corr(t)


def fmt_corr(x):
    if pd.isna(x):
        return "nan"
    return f"{x:.3f}"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Wczytuję lineup values...")
    if not LINEUP_VALUES_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {LINEUP_VALUES_PATH}")

    lineups = pd.read_csv(LINEUP_VALUES_PATH)

    required_lineup_cols = {
        "sezon",
        "match_id",
        "home_lineup_offense",
        "away_lineup_offense",
        "diff_lineup_offense",
        "home_lineup_defense",
        "away_lineup_defense",
        "diff_lineup_defense",
        "home_lineup_minutes",
        "away_lineup_minutes",
        "diff_lineup_minutes",
    }
    missing_lineup = required_lineup_cols - set(lineups.columns)
    if missing_lineup:
        raise RuntimeError(
            f"Brak wymaganych kolumn w {LINEUP_VALUES_PATH}: {sorted(missing_lineup)}"
        )

    lineups = lineups[lineups["sezon"].isin(["2024/25", "2025/26"])].copy()

    print("2. Wczytuję schemat tabeli matches...")
    conn = sqlite3.connect(DB_PATH)
    schema = pd.read_sql_query("PRAGMA table_info(matches)", conn)
    match_cols = schema["name"].tolist()

    detected = detect_match_columns(match_cols)
    print("   Wykryte kolumny:")
    for k, v in detected.items():
        print(f"   - {k:10s} -> {v}")

    sql = f"""
        SELECT
            "{detected['match_id']}" AS match_id,
            "{detected['home_goals']}" AS home_goals,
            "{detected['away_goals']}" AS away_goals,
            "{detected['home_xg']}" AS home_xg,
            "{detected['away_xg']}" AS away_xg
        FROM matches
    """

    print("3. Wczytuję dane meczowe...")
    matches = pd.read_sql_query(sql, conn)
    conn.close()

    for col in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    print("4. Łączę lineup values z meczami...")
    df = pd.merge(lineups, matches, on="match_id", how="inner")

    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    df_goal = df.dropna(subset=["goal_diff"]).copy()
    df_xg = df.dropna(subset=["xg_diff"]).copy()

    df_xg_no_promoted = df_xg[
        (pd.to_numeric(df_xg["home_lineup_minutes"], errors="coerce").fillna(0) > 1000)
        & (pd.to_numeric(df_xg["away_lineup_minutes"], errors="coerce").fillna(0) > 1000)
    ].copy()

    goal_corr_off = safe_corr(df_goal["diff_lineup_offense"], df_goal["goal_diff"])
    goal_corr_def = safe_corr(df_goal["diff_lineup_defense"], df_goal["goal_diff"])
    goal_corr_min = safe_corr(df_goal["diff_lineup_minutes"], df_goal["goal_diff"])

    xg_corr_off = safe_corr(df_xg["diff_lineup_offense"], df_xg["xg_diff"])
    xg_corr_def = safe_corr(df_xg["diff_lineup_defense"], df_xg["xg_diff"])
    xg_corr_min = safe_corr(df_xg["diff_lineup_minutes"], df_xg["xg_diff"])

    xg_np_corr_off = safe_corr(df_xg_no_promoted["diff_lineup_offense"], df_xg_no_promoted["xg_diff"])
    xg_np_corr_def = safe_corr(df_xg_no_promoted["diff_lineup_defense"], df_xg_no_promoted["xg_diff"])
    xg_np_corr_min = safe_corr(df_xg_no_promoted["diff_lineup_minutes"], df_xg_no_promoted["xg_diff"])

    lines = []
    lines.append("=" * 78)
    lines.append("TEST KORELACJI — LINEUP VALUES vs WYNIKI MECZÓW")
    lines.append("=" * 78)
    lines.append("")
    lines.append("1. WYKRYTE KOLUMNY MATCHES")
    lines.append("-" * 78)
    for k, v in detected.items():
        lines.append(f"  {k:10s} -> {v}")
    lines.append("")
    lines.append("2. LICZNOŚCI")
    lines.append("-" * 78)
    lines.append(f"  Mecze lineup (2024/25 + 2025/26): {len(lineups)}")
    lines.append(f"  Mecze po joinie:                  {len(df)}")
    lines.append(f"  Mecze z goal_diff:                {len(df_goal)}")
    lines.append(f"  Mecze z xg_diff:                  {len(df_xg)}")
    lines.append(f"  Mecze xg bez beniaminków:         {len(df_xg_no_promoted)}")
    lines.append("")
    lines.append("3. KORELACJA Z GOAL_DIFF")
    lines.append("-" * 78)
    lines.append(f"  diff_lineup_offense  vs goal_diff: {fmt_corr(goal_corr_off)}")
    lines.append(f"  diff_lineup_defense  vs goal_diff: {fmt_corr(goal_corr_def)}")
    lines.append(f"  diff_lineup_minutes  vs goal_diff: {fmt_corr(goal_corr_min)}")
    lines.append("")
    lines.append("4. KORELACJA Z xG_DIFF")
    lines.append("-" * 78)
    lines.append(f"  diff_lineup_offense  vs xg_diff:   {fmt_corr(xg_corr_off)}")
    lines.append(f"  diff_lineup_defense  vs xg_diff:   {fmt_corr(xg_corr_def)}")
    lines.append(f"  diff_lineup_minutes  vs xg_diff:   {fmt_corr(xg_corr_min)}")
    lines.append("")
    lines.append("5. KORELACJA Z xG_DIFF — BEZ BENIAMINKÓW")
    lines.append("-" * 78)
    lines.append(f"  diff_lineup_offense  vs xg_diff:   {fmt_corr(xg_np_corr_off)}")
    lines.append(f"  diff_lineup_defense  vs xg_diff:   {fmt_corr(xg_np_corr_def)}")
    lines.append(f"  diff_lineup_minutes  vs xg_diff:   {fmt_corr(xg_np_corr_min)}")
    lines.append("")
    lines.append("6. INTERPRETACJA")
    lines.append("-" * 78)
    lines.append("  < 0.10      -> raczej szum")
    lines.append("  0.10-0.20   -> umiarkowany sygnał, dobry jako feature dodatkowy")
    lines.append("  > 0.20      -> silny sygnał")
    lines.append("")
    lines.append("7. RAPORT")
    lines.append("-" * 78)
    lines.append(f"  {REPORT_PATH}")

    report_text = "\n".join(lines)

    print()
    print(report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano raport: {REPORT_PATH}")


if __name__ == "__main__":
    main()