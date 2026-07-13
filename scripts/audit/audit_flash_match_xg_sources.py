import re
from pathlib import Path
import sqlite3
import pandas as pd

RAW_FLASH_DIR = Path("data/raw/flash")
DB_PATH = Path("db/ekstraklasa.db")
REPORT_DIR = Path("data/reports/model")
REPORT_TXT = REPORT_DIR / "audit_flash_match_xg_sources.txt"
REPORT_CSV = REPORT_DIR / "audit_flash_match_xg_sources.csv"

PL_TO_ASCII = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})


def norm(s: str) -> str:
    s = str(s).translate(PL_TO_ASCII).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def detect_first(columns, exact=(), contains=()):
    normalized = {c: norm(c) for c in columns}
    for wanted in exact:
        wanted_n = norm(wanted)
        for c, cn in normalized.items():
            if cn == wanted_n:
                return c
    for wanted in contains:
        wanted_n = norm(wanted)
        for c, cn in normalized.items():
            if wanted_n in cn:
                return c
    return None


def detect_columns(columns):
    return {
        "xg_home_col": detect_first(columns,
            exact=["xg_gosp", "home_xg", "xg_home"],
            contains=["xg_gosp", "home_xg", "xg_home"]),
        "xg_away_col": detect_first(columns,
            exact=["xg_gosc", "away_xg", "xg_away"],
            contains=["xg_gosc", "away_xg", "xg_away"]),
        "season_col": detect_first(columns,
            exact=["sezon", "season"],
            contains=["sezon", "season"]),
        "flash_id_col": detect_first(columns,
            exact=["flash_id", "match_id_flash", "id_flash"],
            contains=["flash_id", "id_flash"]),
        "flash_url_col": detect_first(columns,
            exact=["flash_url", "url", "match_url"],
            contains=["flash_url", "match_url"]),
        "home_team_col": detect_first(columns,
            exact=["gospodarz", "home_team"],
            contains=["gospodarz", "home_team"]),
        "away_team_col": detect_first(columns,
            exact=["gosc", "away_team"],
            contains=["gosc", "away_team"]),
        "round_col": detect_first(columns,
            exact=["kolejka", "round"],
            contains=["kolejka", "round"]),
    }


def load_matches_2023_24():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT match_id, sezon, kolejka, gospodarz, gosc, flash_id, flash_url
        FROM matches WHERE sezon = '2023/24'
        ORDER BY kolejka, match_id
    """, conn)
    conn.close()
    return df


def safe_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "cp1250", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Nie udało się wczytać: {path}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FLASH_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu: {RAW_FLASH_DIR}")

    matches_2023 = load_matches_2023_24()
    flash_id_set = set(matches_2023["flash_id"].dropna().astype(str).str.strip())
    flash_url_set = set(matches_2023["flash_url"].dropna().astype(str).str.strip())

    csv_files = sorted(RAW_FLASH_DIR.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Brak plików CSV w {RAW_FLASH_DIR}")

    print(f"Znaleziono {len(csv_files)} plików CSV w {RAW_FLASH_DIR}")
    print(f"Meczów 2023/24 w DB: {len(matches_2023)}")
    print()

    rows = []

    for path in csv_files:
        rel = str(path.relative_to(RAW_FLASH_DIR))

        try:
            df = safe_read_csv(path)
        except Exception as e:
            rows.append({
                "file": rel, "rows": None, "has_xg_pair": 0,
                "xg_home_col": "", "xg_away_col": "", "season_col": "",
                "flash_id_col": "", "flash_url_col": "",
                "home_team_col": "", "away_team_col": "", "round_col": "",
                "rows_with_both_xg": None, "rows_2023_24": None,
                "matched_flash_id_2023_24": None, "matched_flash_url_2023_24": None,
                "status": f"read_error: {e}",
            })
            continue

        cols = list(df.columns)
        det = detect_columns(cols)
        has_xg_pair = int(det["xg_home_col"] is not None and det["xg_away_col"] is not None)
        rows_total = len(df)
        df_work = df.copy()

        rows_with_both_xg = None
        if has_xg_pair:
            h = pd.to_numeric(df_work[det["xg_home_col"]], errors="coerce")
            a = pd.to_numeric(df_work[det["xg_away_col"]], errors="coerce")
            rows_with_both_xg = int((h.notna() & a.notna()).sum())

        rows_2023_24 = None
        if det["season_col"] is not None:
            season_series = df_work[det["season_col"]].astype(str).str.strip()
            mask = season_series.eq("2023/24") | season_series.eq("2023-2024")
            rows_2023_24 = int(mask.sum())
        elif "2023" in rel:
            rows_2023_24 = rows_total

        matched_flash_id = None
        if det["flash_id_col"] is not None:
            vals = set(df_work[det["flash_id_col"]].dropna().astype(str).str.strip())
            matched_flash_id = len(vals & flash_id_set)

        matched_flash_url = None
        if det["flash_url_col"] is not None:
            vals = set(df_work[det["flash_url_col"]].dropna().astype(str).str.strip())
            matched_flash_url = len(vals & flash_url_set)

        rows.append({
            "file": rel,
            "rows": rows_total,
            "has_xg_pair": has_xg_pair,
            "xg_home_col": det["xg_home_col"] or "",
            "xg_away_col": det["xg_away_col"] or "",
            "season_col": det["season_col"] or "",
            "flash_id_col": det["flash_id_col"] or "",
            "flash_url_col": det["flash_url_col"] or "",
            "home_team_col": det["home_team_col"] or "",
            "away_team_col": det["away_team_col"] or "",
            "round_col": det["round_col"] or "",
            "rows_with_both_xg": rows_with_both_xg,
            "rows_2023_24": rows_2023_24,
            "matched_flash_id_2023_24": matched_flash_id,
            "matched_flash_url_2023_24": matched_flash_url,
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["has_xg_pair", "rows_with_both_xg", "rows"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    out.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=" * 90)
    lines.append("AUDYT ŹRÓDEŁ MATCH xG — FLASHSCORE 2023/24")
    lines.append("=" * 90)
    lines.append(f"Plików CSV:              {len(csv_files)}")
    lines.append(f"Meczów 2023/24 w DB:     {len(matches_2023)}")
    lines.append("")

    xg_files = out[out["has_xg_pair"] == 1].copy()
    lines.append(f"PLIKI Z KOLUMNAMI xG ({len(xg_files)})")
    lines.append("-" * 90)

    if len(xg_files) == 0:
        lines.append("  Brak plików z wykrytymi parami kolumn xG.")
    else:
        for row in xg_files.itertuples(index=False):
            lines.append(
                f"  {row.file}"
                f" | rows={row.rows}"
                f" | both_xg={row.rows_with_both_xg}"
                f" | rows_2023_24={row.rows_2023_24}"
                f" | xg_cols=({row.xg_home_col}, {row.xg_away_col})"
                f" | flash_id_match={row.matched_flash_id_2023_24}"
                f" | flash_url_match={row.matched_flash_url_2023_24}"
                f" | season_col={row.season_col or '-'}"
            )

    lines.append("")
    lines.append("WSZYSTKIE PLIKI (TOP 30)")
    lines.append("-" * 90)
    for row in out.head(30).itertuples(index=False):
        lines.append(
            f"  {row.file}"
            f" | xg_pair={row.has_xg_pair}"
            f" | rows={row.rows}"
            f" | both_xg={row.rows_with_both_xg}"
            f" | rows_2023_24={row.rows_2023_24}"
            f" | flash_id_match={row.matched_flash_id_2023_24}"
            f" | status={row.status}"
        )

    lines.append("")
    lines.append(f"CSV: {REPORT_CSV}")
    lines.append(f"TXT: {REPORT_TXT}")

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {REPORT_TXT}")
    print(f"Zapisano: {REPORT_CSV}")


if __name__ == "__main__":
    main()