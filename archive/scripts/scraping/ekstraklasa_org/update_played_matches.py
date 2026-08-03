"""
update_played_matches.py
========================
V1: aktualizacja rozegranych meczów z ekstraklasa.org do:
- matches
- fixtures_upcoming

Założenia:
- źródłem wejściowym są rekordy fixtures_upcoming z source_url
- parser próbuje wyciągnąć wynik i statystyki ze strony meczu
- jeśli nie da się sparsować wyniku, fixture NIE jest aktualizowany
- jeśli część statystyk nie istnieje / nie zostanie znaleziona, zapisujemy NULL

To jest wersja szkieletowa pod deadline przed startem ligi.
Priorytet:
1) spiąć rozegrany mecz z fixtures_upcoming
2) zapisać wynik do matches
3) złapać tyle statystyk, ile się da bez psucia pipeline
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


DB_PATH = Path("db/ekstraklasa.db")
DEBUG_DIR = Path("data/raw/ekstraklasa_org_debug")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
    "2026/27": 1.0,
}

POLISH_L_MAP = str.maketrans({
    "ł": "l",
    "Ł": "L",
})


MATCHES_DEFAULTS = [
    "match_id", "sezon", "waga_sezonu", "kolejka", "data_meczu",
    "gospodarz", "gosc",
    "gole_gosp", "gole_gosc",
    "posiadanie_gosp", "posiadanie_gosc",
    "strzaly_gosp", "strzaly_gosc",
    "celne_gosp", "celne_gosc",
    "strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc",
    "strzaly_niecelne_gosp", "strzaly_niecelne_gosc",
    "rozne_gosp", "rozne_gosc",
    "faule_gosp", "faule_gosc",
    "spalone_gosp", "spalone_gosc",
    "zk_gosp", "zk_gosc",
    "czk_gosp", "czk_gosc",
    "druga_zk_gosp", "druga_zk_gosc",
    "dosrodkowania_gosp", "dosrodkowania_gosc",
    "dosrodkowania_celne_gosp", "dosrodkowania_celne_gosc",
    "odbiory_gosp", "odbiory_gosc",
    "podania_gosp", "podania_gosc",
    "podania_celne_gosp", "podania_celne_gosc",
    "xg_gosp", "xg_gosc",
    "flash_id", "flash_url",
]

STAT_LABEL_MAP = {
    # wynik / xg
    "xg": "xg",
    "expectedgoals": "xg",
    "expectedgoal": "xg",
    "goalsexpected": "xg",
    "goalsxg": "xg",

    # posiadanie
    "posiadanie": "posiadanie",
    "posiadaniepilki": "posiadanie",
    "possession": "posiadanie",
    "ballpossession": "posiadanie",

    # strzaly
    "strzaly": "strzaly",
    "strzalyogolem": "strzaly",
    "shots": "strzaly",
    "totalshots": "strzaly",

    "strzalycelne": "celne",
    "celnestrzaly": "celne",
    "shotsontarget": "celne",
    "ontargetshots": "celne",
    "targetshots": "celne",

    "strzalyzablokowane": "strzaly_zablokowane",
    "blockedshots": "strzaly_zablokowane",

    "strzalyniecelne": "strzaly_niecelne",
    "shotsofftarget": "strzaly_niecelne",
    "offtargetshots": "strzaly_niecelne",
    "missedshots": "strzaly_niecelne",

    # rożne
    "rozne": "rozne",
    "rzutyrozne": "rozne",
    "corners": "rozne",
    "cornerkicks": "rozne",

    # faule
    "faule": "faule",
    "fouls": "faule",
    "foulscommitted": "faule",

    # spalone
    "spalone": "spalone",
    "offsides": "spalone",
    "offside": "spalone",

    # kartki
    "zoltekartki": "zk",
    "kartkizolte": "zk",
    "yellowcards": "zk",
    "yellowcard": "zk",

    "czerwonakartki": "czk",
    "kartkiczerwone": "czk",
    "redcards": "czk",
    "redcard": "czk",

    "drugazoltekartka": "druga_zk",
    "drugazolta": "druga_zk",
    "secondyellowredcards": "druga_zk",
    "secondyellow": "druga_zk",

    # dośrodkowania
    "dosrodkowania": "dosrodkowania",
    "crosses": "dosrodkowania",
    "totalcrosses": "dosrodkowania",

    "celnedosrodkowania": "dosrodkowania_celne",
    "dokladnedosrodkowania": "dosrodkowania_celne",
    "accuratecrosses": "dosrodkowania_celne",
    "successfulcrosses": "dosrodkowania_celne",

    # odbiory / tackles
    "odbiory": "odbiory",
    "tackles": "odbiory",
    "won tackles": "odbiory",
    "wontackles": "odbiory",

    # podania
    "podania": "podania",
    "passes": "podania",
    "totalpasses": "podania",

    "celnepodania": "podania_celne",
    "dokladnepodania": "podania_celne",
    "accuratepasses": "podania_celne",
    "successfulpasses": "podania_celne",
}

CANONICAL_TO_MATCH_COLS = {
    "posiadanie": ("posiadanie_gosp", "posiadanie_gosc"),
    "strzaly": ("strzaly_gosp", "strzaly_gosc"),
    "celne": ("celne_gosp", "celne_gosc"),
    "strzaly_zablokowane": ("strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc"),
    "strzaly_niecelne": ("strzaly_niecelne_gosp", "strzaly_niecelne_gosc"),
    "rozne": ("rozne_gosp", "rozne_gosc"),
    "faule": ("faule_gosp", "faule_gosc"),
    "spalone": ("spalone_gosp", "spalone_gosc"),
    "zk": ("zk_gosp", "zk_gosc"),
    "czk": ("czk_gosp", "czk_gosc"),
    "druga_zk": ("druga_zk_gosp", "druga_zk_gosc"),
    "dosrodkowania": ("dosrodkowania_gosp", "dosrodkowania_gosc"),
    "dosrodkowania_celne": ("dosrodkowania_celne_gosp", "dosrodkowania_celne_gosc"),
    "odbiory": ("odbiory_gosp", "odbiory_gosc"),
    "podania": ("podania_gosp", "podania_gosc"),
    "podania_celne": ("podania_celne_gosp", "podania_celne_gosc"),
    "xg": ("xg_gosp", "xg_gosc"),
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).translate(POLISH_L_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label(text: Any) -> str:
    s = normalize_text(text)
    s = re.sub(r"[%:/\-–—\.\,\(\)\[\]]", " ", s)
    s = re.sub(r"\s+", "", s)
    return s


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    s = s.replace("\xa0", " ").replace("%", "").replace(",", ".").strip()
    s = s.replace("–", "-").replace("—", "-")

    if s.lower() in {"null", "none", "nan", "n/a", "brak", "-", ""}:
        return None

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None

    try:
        return float(m.group(0))
    except ValueError:
        return None


def safe_int(value: Any) -> Optional[int]:
    x = safe_float(value)
    if x is None:
        return None
    return int(round(x))


def walk_nodes(node: Any) -> Generator[Any, None, None]:
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from walk_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_nodes(v)


def extract_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None

    direct = safe_float(value)
    if direct is not None:
        return direct

    if isinstance(value, dict):
        for key in [
            "value", "stat", "score", "total", "current",
            "displayValue", "display", "amount", "number"
        ]:
            if key in value:
                got = safe_float(value.get(key))
                if got is not None:
                    return got

    if isinstance(value, list) and len(value) == 1:
        return extract_scalar(value[0])

    return None


def detect_home_away_pair(node: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pair_keys = [
        ("home", "away"),
        ("homeValue", "awayValue"),
        ("left", "right"),
        ("team1Value", "team2Value"),
        ("valueHome", "valueAway"),
        ("home_score", "away_score"),
        ("scoreHome", "scoreAway"),
        ("goalsHome", "goalsAway"),
        ("homeGoals", "awayGoals"),
        ("resultHome", "resultAway"),
        ("homeXg", "awayXg"),
        ("xgHome", "xgAway"),
        ("local", "visitor"),
        ("hosts", "guests"),
    ]

    for hk, ak in pair_keys:
        if hk in node and ak in node:
            hv = extract_scalar(node.get(hk))
            av = extract_scalar(node.get(ak))
            if hv is not None and av is not None:
                return hv, av

    nested_pairs = [
        ("home", "away"),
        ("local", "visitor"),
    ]
    for hk, ak in nested_pairs:
        if hk in node and ak in node and isinstance(node.get(hk), dict) and isinstance(node.get(ak), dict):
            hv = extract_scalar(node[hk])
            av = extract_scalar(node[ak])
            if hv is not None and av is not None:
                return hv, av

    return None


def parse_json_scripts(soup: BeautifulSoup) -> List[Any]:
    objects: List[Any] = []

    for script in soup.find_all("script"):
        text = script.string if script.string is not None else script.get_text()
        if not text:
            continue
        text = text.strip()
        if not text:
            continue

        script_id = script.get("id", "")
        script_type = script.get("type", "")

        if script_id == "__NEXT_DATA__":
            try:
                objects.append(json.loads(text))
            except Exception:
                pass

        if script_type == "application/ld+json":
            try:
                objects.append(json.loads(text))
            except Exception:
                pass

        m = re.search(r"window\.__NEXT_DATA__\s*=\s*(\{.*\})\s*;?", text, flags=re.S)
        if m:
            try:
                objects.append(json.loads(m.group(1)))
            except Exception:
                pass

    return objects


def extract_score_candidates_from_objects(objects: Iterable[Any]) -> List[Tuple[int, int, int]]:
    candidates: List[Tuple[int, int, int]] = []

    strong_pairs = [
        ("homeScore", "awayScore"),
        ("scoreHome", "scoreAway"),
        ("goalsHome", "goalsAway"),
        ("homeGoals", "awayGoals"),
        ("resultHome", "resultAway"),
    ]

    medium_pairs = [
        ("home", "away"),
        ("local", "visitor"),
    ]

    for obj in objects:
        for node in walk_nodes(obj):
            if not isinstance(node, dict):
                continue

            for hk, ak in strong_pairs:
                if hk in node and ak in node:
                    gh = safe_int(node.get(hk))
                    ga = safe_int(node.get(ak))
                    if gh is not None and ga is not None and 0 <= gh <= 20 and 0 <= ga <= 20:
                        candidates.append((100, gh, ga))

            for hk, ak in medium_pairs:
                if hk in node and ak in node:
                    hv = node.get(hk)
                    av = node.get(ak)

                    if isinstance(hv, dict) and isinstance(av, dict):
                        gh = safe_int(hv.get("score"))
                        ga = safe_int(av.get("score"))
                        if gh is not None and ga is not None and 0 <= gh <= 20 and 0 <= ga <= 20:
                            candidates.append((90, gh, ga))

                        gh = safe_int(hv.get("goals"))
                        ga = safe_int(av.get("goals"))
                        if gh is not None and ga is not None and 0 <= gh <= 20 and 0 <= ga <= 20:
                            candidates.append((90, gh, ga))

    return candidates


def extract_stats_from_objects(objects: Iterable[Any]) -> Dict[str, Tuple[float, float]]:
    found: Dict[str, Tuple[float, float]] = {}

    label_keys = [
        "label", "name", "title", "statName", "key", "caption",
        "metric", "parameter", "type", "category"
    ]

    for obj in objects:
        for node in walk_nodes(obj):
            if not isinstance(node, dict):
                continue

            label = None
            for lk in label_keys:
                if lk in node and isinstance(node.get(lk), (str, int, float)):
                    label = str(node.get(lk))
                    break

            pair = detect_home_away_pair(node)
            if not label or pair is None:
                continue

            normalized = normalize_label(label)
            canonical = STAT_LABEL_MAP.get(normalized)
            if canonical and canonical not in found:
                found[canonical] = pair

    return found


def extract_score_from_html_text(soup: BeautifulSoup) -> Optional[Tuple[int, int]]:
    text = soup.get_text(" ", strip=True)
    if not text:
        return None

    patterns = [
        r"\b(\d{1,2})\s*[:\-]\s*(\d{1,2})\b",
    ]

    candidates: List[Tuple[int, int]] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            gh = safe_int(m.group(1))
            ga = safe_int(m.group(2))
            if gh is not None and ga is not None and 0 <= gh <= 20 and 0 <= ga <= 20:
                # odfiltruj typowe "2026-07"
                if gh > 9 or ga > 9:
                    continue
                candidates.append((gh, ga))

    if candidates:
        return candidates[0]

    return None


def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def parse_match_page(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    objects = parse_json_scripts(soup)

    parsed: Dict[str, Any] = {
        "gole_gosp": None,
        "gole_gosc": None,
        "stats": {},
    }

    score_candidates = extract_score_candidates_from_objects(objects)
    if score_candidates:
        score_candidates = sorted(score_candidates, key=lambda x: x[0], reverse=True)
        _, gh, ga = score_candidates[0]
        parsed["gole_gosp"] = gh
        parsed["gole_gosc"] = ga
    else:
        score_from_text = extract_score_from_html_text(soup)
        if score_from_text is not None:
            parsed["gole_gosp"], parsed["gole_gosc"] = score_from_text

    parsed["stats"] = extract_stats_from_objects(objects)

    # fallback xG z meta/html jeśli w stats nie znaleziono
    if "xg" not in parsed["stats"]:
        text = soup.get_text(" ", strip=True)
        text_norm = normalize_text(text)
        m = re.search(
            r"xg[^0-9]{0,20}(\d+(?:[\.,]\d+)?)\D+(\d+(?:[\.,]\d+)?)",
            text_norm,
            flags=re.I
        )
        if m:
            xh = safe_float(m.group(1))
            xa = safe_float(m.group(2))
            if xh is not None and xa is not None:
                parsed["stats"]["xg"] = (xh, xa)

    return parsed


def get_matches_columns(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("PRAGMA table_info(matches)").fetchall()
    return [row[1] for row in rows]


def build_match_row(fixture: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    row = {col: None for col in MATCHES_DEFAULTS}

    row["sezon"] = fixture["sezon"]
    row["waga_sezonu"] = WAGI_SEZONOW.get(fixture["sezon"], 1.0)
    row["kolejka"] = fixture["kolejka"]
    row["data_meczu"] = fixture.get("data_planowana")
    row["gospodarz"] = fixture["gospodarz"]
    row["gosc"] = fixture["gosc"]
    row["gole_gosp"] = parsed.get("gole_gosp")
    row["gole_gosc"] = parsed.get("gole_gosc")
    row["flash_id"] = None
    row["flash_url"] = None

    stats = parsed.get("stats", {})
    for canonical, (home_col, away_col) in CANONICAL_TO_MATCH_COLS.items():
        if canonical in stats:
            row[home_col] = stats[canonical][0]
            row[away_col] = stats[canonical][1]

    return row


def find_existing_match_id(conn: sqlite3.Connection, fixture: Dict[str, Any]) -> Optional[int]:
    if fixture.get("played_match_id") is not None:
        return int(fixture["played_match_id"])

    row = conn.execute(
        """
        SELECT match_id
        FROM matches
        WHERE sezon = ?
          AND kolejka = ?
          AND gospodarz = ?
          AND gosc = ?
        LIMIT 1
        """,
        (fixture["sezon"], fixture["kolejka"], fixture["gospodarz"], fixture["gosc"])
    ).fetchone()

    if row:
        return int(row[0])

    return None


def next_match_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(match_id), 0) + 1 FROM matches").fetchone()
    return int(row[0])


def insert_or_update_match(
    conn: sqlite3.Connection,
    fixture: Dict[str, Any],
    parsed: Dict[str, Any],
    dry_run: bool = False,
) -> Tuple[int, str]:
    match_row = build_match_row(fixture, parsed)
    existing_match_id = find_existing_match_id(conn, fixture)

    matches_columns = get_matches_columns(conn)
    valid_columns = [c for c in match_row.keys() if c in matches_columns]

    if existing_match_id is None:
        match_id = next_match_id(conn)
        match_row["match_id"] = match_id

        cols = [c for c in valid_columns if c != "match_id"]
        cols = ["match_id"] + cols

        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO matches ({', '.join(cols)}) VALUES ({placeholders})"
        values = [match_row.get(c) for c in cols]

        if not dry_run:
            conn.execute(sql, values)

        return match_id, "insert"

    match_id = int(existing_match_id)
    match_row["match_id"] = match_id

    update_cols = [c for c in valid_columns if c != "match_id"]
    assignments = ", ".join([f"{c} = ?" for c in update_cols])
    sql = f"UPDATE matches SET {assignments} WHERE match_id = ?"
    values = [match_row.get(c) for c in update_cols] + [match_id]

    if not dry_run:
        conn.execute(sql, values)

    return match_id, "update"


def update_fixture_status(
    conn: sqlite3.Connection,
    fixture_id: str,
    match_id: int,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return

    conn.execute(
        """
        UPDATE fixtures_upcoming
        SET status = 'played',
            played_match_id = ?,
            updated_at = ?
        WHERE fixture_id = ?
        """,
        (match_id, now_iso(), fixture_id)
    )


def save_debug_html(fixture_id: str, html: str) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{fixture_id}.html"
    path.write_text(html, encoding="utf-8")
    return path


def load_fixtures(
    conn: sqlite3.Connection,
    sezon: str,
    kolejka: Optional[int],
    fixture_id: Optional[str],
) -> List[Dict[str, Any]]:
    where = ["sezon = ?"]
    params: List[Any] = [sezon]

    if kolejka is not None:
        where.append("kolejka = ?")
        params.append(kolejka)

    if fixture_id is not None:
        where.append("fixture_id = ?")
        params.append(fixture_id)

    sql = f"""
        SELECT *
        FROM fixtures_upcoming
        WHERE {' AND '.join(where)}
        ORDER BY kolejka, data_planowana, gospodarz
    """

    df = pd.read_sql_query(sql, conn, params=params)
    return df.to_dict(orient="records")


def should_skip_fixture(fixture: Dict[str, Any], force: bool) -> Tuple[bool, str]:
    status = fixture.get("status")
    if status == "played" and not force:
        return True, "status=played"

    url = fixture.get("source_url")
    if not url:
        return True, "brak source_url"

    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True, help="np. 2026/27")
    parser.add_argument("--kolejka", type=int, default=None, help="np. 1")
    parser.add_argument("--fixture-id", default=None, help="pojedynczy fixture_id")
    parser.add_argument("--dry-run", action="store_true", help="bez zapisu do DB")
    parser.add_argument("--force", action="store_true", help="przetwarzaj nawet status=played")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Brak bazy: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        fixtures = load_fixtures(conn, args.sezon, args.kolejka, args.fixture_id)

        print("=" * 78)
        print("UPDATE PLAYED MATCHES - EKSTRAKLASA.ORG V1")
        print("=" * 78)
        print(f"Sezon:     {args.sezon}")
        print(f"Kolejka:   {args.kolejka}")
        print(f"Fixture:   {args.fixture_id}")
        print(f"Dry-run:   {args.dry_run}")
        print(f"Force:     {args.force}")
        print(f"Fixtures:  {len(fixtures)}")
        print("")

        if not fixtures:
            print("Brak fixture do przetworzenia.")
            return

        processed = 0
        inserted = 0
        updated = 0
        fixture_updated = 0
        skipped = 0
        failed = 0

        for fixture in fixtures:
            processed += 1
            fixture_id = fixture["fixture_id"]
            home = fixture["gospodarz"]
            away = fixture["gosc"]
            url = fixture.get("source_url")

            print("-" * 78)
            print(f"[{processed}/{len(fixtures)}] {home} vs {away}")
            print(f"fixture_id: {fixture_id}")
            print(f"url:        {url}")

            skip, reason = should_skip_fixture(fixture, args.force)
            if skip:
                skipped += 1
                print(f"SKIP: {reason}")
                continue

            try:
                html = fetch_page(url)
            except Exception as e:
                failed += 1
                print(f"ERROR fetch: {e}")
                continue

            try:
                parsed = parse_match_page(html)
            except Exception as e:
                failed += 1
                debug_path = save_debug_html(fixture_id, html)
                print(f"ERROR parse: {e}")
                print(f"DEBUG HTML: {debug_path}")
                continue

            gh = parsed.get("gole_gosp")
            ga = parsed.get("gole_gosc")
            if gh is None or ga is None:
                failed += 1
                debug_path = save_debug_html(fixture_id, html)
                print("BRAK WYNIKU - nie aktualizuję DB.")
                print(f"DEBUG HTML: {debug_path}")
                continue

            print(f"Wynik:      {gh}-{ga}")

            if "xg" in parsed.get("stats", {}):
                xh, xa = parsed["stats"]["xg"]
                print(f"xG:         {xh} - {xa}")
            else:
                print("xG:         brak")

            found_stats = sorted(parsed.get("stats", {}).keys())
            print(f"Statystyki: {', '.join(found_stats) if found_stats else 'brak'}")

            try:
                match_id, mode = insert_or_update_match(
                    conn=conn,
                    fixture=fixture,
                    parsed=parsed,
                    dry_run=args.dry_run,
                )
                if mode == "insert":
                    inserted += 1
                else:
                    updated += 1

                update_fixture_status(
                    conn=conn,
                    fixture_id=fixture_id,
                    match_id=match_id,
                    dry_run=args.dry_run,
                )
                fixture_updated += 1

                print(f"DB MATCH:   {mode} -> match_id={match_id}")
                print("FIXTURE:    status=played")
            except Exception as e:
                failed += 1
                debug_path = save_debug_html(fixture_id, html)
                print(f"ERROR DB: {e}")
                print(f"DEBUG HTML: {debug_path}")
                continue

        if not args.dry_run:
            conn.commit()

        print("")
        print("=" * 78)
        print("PODSUMOWANIE")
        print("=" * 78)
        print(f"Przetworzone:            {processed}")
        print(f"Inserted matches:        {inserted}")
        print(f"Updated matches:         {updated}")
        print(f"Updated fixtures:        {fixture_updated}")
        print(f"Skipped:                 {skipped}")
        print(f"Failed:                  {failed}")
        print(f"Tryb dry-run:            {args.dry_run}")
        print("=" * 78)

    finally:
        conn.close()


if __name__ == "__main__":
    main()