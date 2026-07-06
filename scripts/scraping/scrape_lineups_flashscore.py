"""
scrape_lineups_flashscore_v2.py
===============================
Scrapuje z Flashscore:
1) starterów
2) ławkę
3) absencje / wykluczonych z gry
4) trenerów
5) zmiany w meczu

Źródło:
matches.flash_url
np.
https://www.flashscore.pl/mecz/pilka-nozna/.../szczegoly/statystyki/?mid=CCPtlaaU

zamieniamy na:
https://www.flashscore.pl/mecz/pilka-nozna/.../szczegoly/sklady/?mid=CCPtlaaU
"""

import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("db/ekstraklasa.db")
DEBUG_DIR = Path("data/debug_lineups")

# Na start test:
LIMIT_MATCHES = 10
# Po udanym teście:
# LIMIT_MATCHES = None

HEADLESS = True
PAGE_TIMEOUT_MS = 60000
WAIT_SELECTOR_TIMEOUT_MS = 20000
SLEEP_AFTER_LOAD = 2.5
SLEEP_BETWEEN_MATCHES = 1.2

PLAYER_NODE_SELECTOR = ".lf__participantNew"

RE_NUMBER = re.compile(r"^\d+$")
RE_MINUTE = re.compile(r"^(\d+)(?:\+(\d+))?'$")
RE_RATING = re.compile(r"^\d+(?:\.\d+)?$")


# ============================================================
# DB SCHEMA
# ============================================================

def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS lineups (
            sezon TEXT NOT NULL,
            match_id INTEGER NOT NULL,
            team_side TEXT NOT NULL,          -- home / away
            team_name TEXT NOT NULL,
            player_name TEXT NOT NULL,
            shirt_number TEXT,
            is_starter INTEGER NOT NULL,      -- 1 starter, 0 bench
            player_order INTEGER NOT NULL,
            is_goalkeeper INTEGER NOT NULL DEFAULT 0,
            is_captain INTEGER NOT NULL DEFAULT 0,
            raw_text TEXT,
            scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (sezon, match_id, team_side, is_starter, player_order)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_lineups_match
        ON lineups (sezon, match_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_absences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sezon TEXT NOT NULL,
            match_id INTEGER NOT NULL,
            team_side TEXT NOT NULL,
            team_name TEXT NOT NULL,
            player_name TEXT NOT NULL,
            reason_raw TEXT,
            raw_text TEXT,
            scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_absences_match
        ON match_absences (sezon, match_id)
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_coaches (
            sezon TEXT NOT NULL,
            match_id INTEGER NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_coach TEXT,
            away_coach TEXT,
            raw_text TEXT,
            scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (sezon, match_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_substitutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sezon TEXT NOT NULL,
            match_id INTEGER NOT NULL,
            team_side TEXT NOT NULL,
            team_name TEXT NOT NULL,
            minute_raw TEXT,
            minute_num INTEGER,
            player_in TEXT,
            player_out TEXT,
            rating_raw TEXT,
            raw_text TEXT,
            scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_subs_match
        ON match_substitutions (sezon, match_id)
    """)

    conn.commit()


# ============================================================
# DB HELPERS
# ============================================================

def get_matches(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = """
        SELECT
            match_id,
            sezon,
            data_meczu,
            gospodarz,
            gosc,
            flash_id,
            flash_url
        FROM matches
        WHERE flash_id IS NOT NULL
          AND flash_url IS NOT NULL
        ORDER BY data_meczu ASC, match_id ASC
    """

    if LIMIT_MATCHES is not None:
        sql += f" LIMIT {int(LIMIT_MATCHES)}"

    cur.execute(sql)
    return cur.fetchall()


def delete_existing_for_match(conn: sqlite3.Connection, sezon: str, match_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM lineups WHERE sezon = ? AND match_id = ?", (sezon, match_id))
    cur.execute("DELETE FROM match_absences WHERE sezon = ? AND match_id = ?", (sezon, match_id))
    cur.execute("DELETE FROM match_coaches WHERE sezon = ? AND match_id = ?", (sezon, match_id))
    cur.execute("DELETE FROM match_substitutions WHERE sezon = ? AND match_id = ?", (sezon, match_id))
    conn.commit()


def save_parsed_data(conn: sqlite3.Connection, data: Dict) -> None:
    cur = conn.cursor()

    # lineups
    cur.executemany("""
        INSERT INTO lineups (
            sezon, match_id, team_side, team_name,
            player_name, shirt_number, is_starter,
            player_order, is_goalkeeper, is_captain, raw_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            r["sezon"], r["match_id"], r["team_side"], r["team_name"],
            r["player_name"], r["shirt_number"], r["is_starter"],
            r["player_order"], r["is_goalkeeper"], r["is_captain"], r["raw_text"]
        )
        for r in data["lineups"]
    ])

    # absences
    cur.executemany("""
        INSERT INTO match_absences (
            sezon, match_id, team_side, team_name,
            player_name, reason_raw, raw_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            r["sezon"], r["match_id"], r["team_side"], r["team_name"],
            r["player_name"], r["reason_raw"], r["raw_text"]
        )
        for r in data["absences"]
    ])

    # substitutions
    cur.executemany("""
        INSERT INTO match_substitutions (
            sezon, match_id, team_side, team_name,
            minute_raw, minute_num, player_in,
            player_out, rating_raw, raw_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            r["sezon"], r["match_id"], r["team_side"], r["team_name"],
            r["minute_raw"], r["minute_num"], r["player_in"],
            r["player_out"], r["rating_raw"], r["raw_text"]
        )
        for r in data["substitutions"]
    ])

    # coaches
    cur.execute("""
        INSERT INTO match_coaches (
            sezon, match_id, home_team, away_team,
            home_coach, away_coach, raw_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        data["coaches"]["sezon"],
        data["coaches"]["match_id"],
        data["coaches"]["home_team"],
        data["coaches"]["away_team"],
        data["coaches"]["home_coach"],
        data["coaches"]["away_coach"],
        data["coaches"]["raw_text"],
    ))

    conn.commit()


# ============================================================
# HELPERS
# ============================================================

def build_lineups_url(flash_url: str) -> str:
    return flash_url.replace("/szczegoly/statystyki/", "/szczegoly/sklady/")


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def stripped_parts(node) -> List[str]:
    parts = [clean_text(x) for x in node.stripped_strings]
    return [p for p in parts if p]


def node_side(node) -> str:
    classes = node.get("class", [])
    return "away" if "lf__isReversed" in classes else "home"


def node_team_name(side: str, home_team: str, away_team: str) -> str:
    return away_team if side == "away" else home_team


def minute_to_num(minute_raw: Optional[str]) -> Optional[int]:
    if not minute_raw:
        return None
    m = RE_MINUTE.match(minute_raw)
    if not m:
        return None
    base = int(m.group(1))
    extra = int(m.group(2)) if m.group(2) else 0
    return base + extra


def raw_from_parts(parts: List[str]) -> str:
    return " | ".join(parts)


def save_debug_html(sezon: str, match_id: int, html: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / f"{sezon.replace('/', '_')}_{match_id}.html"
    out.write_text(html, encoding="utf-8")


# ============================================================
# PARSERS
# ============================================================

def parse_numbered_player(parts: List[str]) -> Optional[Dict]:
    """
    Przykłady:
    ["50", "Abramowicz S.", "(B)"]
    ["6", "Romanczuk T.", "(C)"]
    ["20", "Miki Villar"]
    """
    if len(parts) < 2:
        return None
    if not RE_NUMBER.match(parts[0]):
        return None

    shirt_number = parts[0]
    name_chunks = []
    is_goalkeeper = 0
    is_captain = 0

    for p in parts[1:]:
        if p == "(B)":
            is_goalkeeper = 1
        elif p == "(C)":
            is_captain = 1
        else:
            name_chunks.append(p)

    player_name = clean_text(" ".join(name_chunks))
    if not player_name:
        return None

    return {
        "shirt_number": shirt_number,
        "player_name": player_name,
        "is_goalkeeper": is_goalkeeper,
        "is_captain": is_captain,
        "raw_text": raw_from_parts(parts),
    }


def parse_absence(parts: List[str]) -> Optional[Dict]:
    """
    Przykład:
    ["Kidric R.", "Kontuzja"]
    ["Thiago", "Czerwona kartka"]
    """
    if not parts:
        return None

    player_name = clean_text(parts[0])
    reason_raw = clean_text(" ".join(parts[1:])) if len(parts) > 1 else None

    if not player_name:
        return None

    return {
        "player_name": player_name,
        "reason_raw": reason_raw,
        "raw_text": raw_from_parts(parts),
    }


def parse_coach(parts: List[str]) -> Optional[str]:
    if not parts:
        return None
    return clean_text(" ".join(parts))


def parse_substitution(parts: List[str]) -> Optional[Dict]:
    """
    Przykłady:
    ["Nene", "8.3", "Kubicki J.", "62'"]
    ["Diaby-Fadiga L.", "Pululu A.", "81'"]
    """
    if len(parts) < 3:
        return None

    minute_raw = parts[-1] if RE_MINUTE.match(parts[-1]) else None
    core = parts[:-1] if minute_raw else parts[:]

    if len(core) < 2:
        return None

    rating_raw = None
    player_in = None
    player_out = None

    # wariant z ratingiem
    if len(core) >= 3 and RE_RATING.match(core[1]):
        player_in = clean_text(core[0])
        rating_raw = core[1]
        player_out = clean_text(" ".join(core[2:]))

    # wariant bez ratingu
    else:
        player_in = clean_text(core[0])
        player_out = clean_text(" ".join(core[1:]))

    if not player_in or not player_out:
        return None

    return {
        "player_in": player_in,
        "player_out": player_out,
        "rating_raw": rating_raw,
        "minute_raw": minute_raw,
        "minute_num": minute_to_num(minute_raw),
        "raw_text": raw_from_parts(parts),
    }


# ============================================================
# EXTRACTION
# ============================================================

def extract_match_data(
    soup: BeautifulSoup,
    sezon: str,
    match_id: int,
    home_team: str,
    away_team: str,
) -> Dict:
    nodes = soup.select(PLAYER_NODE_SELECTOR)
    if not nodes:
        raise ValueError("Nie znaleziono żadnych .lf__participantNew")

    numbered_candidates = []
    substitution_rows = []
    other_by_side = {"home": [], "away": []}

    for node in nodes:
        parts = stripped_parts(node)
        if not parts:
            continue

        side = node_side(node)
        team_name = node_team_name(side, home_team, away_team)
        classes = node.get("class", [])

        # Zmiany
        if "lf__participantNew--substituedPlayer" in classes:
            sub = parse_substitution(parts)
            if sub:
                substitution_rows.append({
                    "sezon": sezon,
                    "match_id": match_id,
                    "team_side": side,
                    "team_name": team_name,
                    **sub
                })
            continue

        # Zawodnicy z numerem
        pl = parse_numbered_player(parts)
        if pl:
            numbered_candidates.append({
                "sezon": sezon,
                "match_id": match_id,
                "team_side": side,
                "team_name": team_name,
                **pl
            })
            continue

        # Reszta = absencje i trenerzy
        other_by_side[side].append({
            "parts": parts,
            "raw_text": raw_from_parts(parts),
            "team_name": team_name,
        })

    # --------------------------------------------------------
    # LINEUPS
    # pierwsze 22 ponumerowane rekordy = starterzy
    # reszta ponumerowanych = ławka
    # --------------------------------------------------------
    if len(numbered_candidates) < 22:
        raise ValueError(f"Za mało ponumerowanych zawodników: {len(numbered_candidates)}")

    starters_raw = numbered_candidates[:22]
    bench_raw = numbered_candidates[22:]

    home_starters_raw = [r for r in starters_raw if r["team_side"] == "home"]
    away_starters_raw = [r for r in starters_raw if r["team_side"] == "away"]

    if len(home_starters_raw) != 11 or len(away_starters_raw) != 11:
        raise ValueError(
            f"Starterzy nie wyszli 11/11: home={len(home_starters_raw)}, away={len(away_starters_raw)}"
        )

    lineup_rows = []

    h_order = 0
    a_order = 0
    for r in starters_raw:
        if r["team_side"] == "home":
            h_order += 1
            order = h_order
        else:
            a_order += 1
            order = a_order

        lineup_rows.append({
            **r,
            "is_starter": 1,
            "player_order": order,
        })

    hb_order = 0
    ab_order = 0
    for r in bench_raw:
        if r["team_side"] == "home":
            hb_order += 1
            order = hb_order
        else:
            ab_order += 1
            order = ab_order

        lineup_rows.append({
            **r,
            "is_starter": 0,
            "player_order": order,
        })

    # --------------------------------------------------------
    # ABSENCES + COACHES
    # logika:
    # dla każdej strony ostatni nieponumerowany rekord = trener
    # wcześniejsze = absencje
    # --------------------------------------------------------
    absence_rows = []
    coach_home = None
    coach_away = None

    for side in ["home", "away"]:
        items = other_by_side[side]
        if not items:
            continue

        coach_item = items[-1]
        coach_name = parse_coach(coach_item["parts"])

        if side == "home":
            coach_home = coach_name
        else:
            coach_away = coach_name

        for item in items[:-1]:
            abs_parsed = parse_absence(item["parts"])
            if not abs_parsed:
                continue

            absence_rows.append({
                "sezon": sezon,
                "match_id": match_id,
                "team_side": side,
                "team_name": item["team_name"],
                **abs_parsed
            })

    coaches_row = {
        "sezon": sezon,
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_coach": coach_home,
        "away_coach": coach_away,
        "raw_text": f"home={coach_home or ''} | away={coach_away or ''}",
    }

    return {
        "lineups": lineup_rows,
        "absences": absence_rows,
        "substitutions": substitution_rows,
        "coaches": coaches_row,
    }


# ============================================================
# SCRAPING
# ============================================================

async def scrape_one_match(
    browser,
    sezon: str,
    match_id: int,
    home_team: str,
    away_team: str,
    flash_url: str,
) -> Dict:
    url = build_lineups_url(flash_url)

    # nowa sesja / context dla każdego meczu
    context = await browser.new_context(locale="pl-PL")
    page = await context.new_page()

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
        await page.wait_for_selector(PLAYER_NODE_SELECTOR, timeout=WAIT_SELECTOR_TIMEOUT_MS)
        await asyncio.sleep(SLEEP_AFTER_LOAD)

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")

        data = extract_match_data(
            soup=soup,
            sezon=sezon,
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
        )
        return data

    except Exception:
        try:
            html = await page.content()
            save_debug_html(sezon, match_id, html)
        except Exception:
            pass
        raise

    finally:
        await context.close()


# ============================================================
# MAIN
# ============================================================

async def main():
    if not DB_PATH.exists():
        print(f"Brak bazy: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    matches = get_matches(conn)

    print("=" * 80)
    print("SCRAPING FLASHSCORE /SKLADY")
    print("=" * 80)
    print(f"Mecze do obrobienia: {len(matches)}")
    print(f"HEADLESS: {HEADLESS}")
    print(f"LIMIT_MATCHES: {LIMIT_MATCHES}")
    print()

    ok_count = 0
    err_count = 0

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=HEADLESS)

        for i, row in enumerate(matches, start=1):
            sezon = row["sezon"]
            match_id = row["match_id"]
            home_team = row["gospodarz"]
            away_team = row["gosc"]
            flash_url = row["flash_url"]

            try:
                parsed = await scrape_one_match(
                    browser=browser,
                    sezon=sezon,
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    flash_url=flash_url,
                )

                delete_existing_for_match(conn, sezon, match_id)
                save_parsed_data(conn, parsed)

                hs = sum(1 for r in parsed["lineups"] if r["team_side"] == "home" and r["is_starter"] == 1)
                as_ = sum(1 for r in parsed["lineups"] if r["team_side"] == "away" and r["is_starter"] == 1)
                hb = sum(1 for r in parsed["lineups"] if r["team_side"] == "home" and r["is_starter"] == 0)
                ab = sum(1 for r in parsed["lineups"] if r["team_side"] == "away" and r["is_starter"] == 0)

                ha = sum(1 for r in parsed["absences"] if r["team_side"] == "home")
                aa = sum(1 for r in parsed["absences"] if r["team_side"] == "away")

                hsub = sum(1 for r in parsed["substitutions"] if r["team_side"] == "home")
                asub = sum(1 for r in parsed["substitutions"] if r["team_side"] == "away")

                home_coach = parsed["coaches"]["home_coach"]
                away_coach = parsed["coaches"]["away_coach"]

                print(
                    f"[{i:03d}/{len(matches)}] OK  | {sezon} | {home_team} vs {away_team} | "
                    f"START {hs}-{as_} | BENCH {hb}-{ab} | ABS {ha}-{aa} | SUB {hsub}-{asub} | "
                    f"COACH {home_coach} / {away_coach}"
                )
                ok_count += 1

            except Exception as e:
                print(
                    f"[{i:03d}/{len(matches)}] ERR | {sezon} | {home_team} vs {away_team} | "
                    f"{type(e).__name__}: {e}"
                )
                err_count += 1

            await asyncio.sleep(SLEEP_BETWEEN_MATCHES)

        await browser.close()

    conn.close()

    print()
    print("=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)
    print(f"OK : {ok_count}")
    print(f"ERR: {err_count}")
    print(f"Debug HTML (błędy): {DEBUG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())