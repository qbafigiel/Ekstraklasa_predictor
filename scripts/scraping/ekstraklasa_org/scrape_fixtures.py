"""
scrape_fixtures.py
==================
Scraper terminarza ekstraklasa.org dla wskazanego sezonu.
"""

import argparse
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "db" / "ekstraklasa.db"
REPORT_PATH = ROOT / "data" / "reports" / "model" / "scrape_fixtures_report.txt"

BASE_URL = "https://ekstraklasa.org/terminarz/{season_slug}/{kolejka}-kolejka/"

NAME_ALIASES = {
    "Wisła Kraków": "Wisła Kraków",
    "Wieczysta Kraków": "Wieczysta Kraków",
    "Śląsk Wrocław": "Śląsk Wrocław",
}

FIXTURE_LINK_RE = re.compile(r"^/mecz/([0-9a-f-]{36})/([^/]+)/statystyki/?$")


def normalize_ascii(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def season_to_slug(sezon: str) -> str:
    if "/" not in sezon:
        raise ValueError(f"Bledny format sezonu: {sezon}")
    left, right = sezon.split("/")
    left = left.strip()
    right = right.strip()
    if len(right) == 2:
        right = str(int(left[:2] + right))
    return f"{left}-{right}"


def load_team_name_map(conn):
    rows = conn.execute("""
        SELECT DISTINCT gospodarz FROM matches
        UNION
        SELECT DISTINCT gosc FROM matches
    """).fetchall()
    name_map = {}
    for (name,) in rows:
        if name:
            name_map[normalize_ascii(name)] = name
    for k, v in NAME_ALIASES.items():
        name_map[normalize_ascii(k)] = v
    return name_map


def resolve_team_name(raw, name_map):
    norm = normalize_ascii(raw)
    if norm in name_map:
        return name_map[norm]
    return raw.strip()


def extract_fixture_from_block(fixture_block, sezon, kolejka, name_map):
    link = None
    for a in fixture_block.find_all("a", href=True):
        if FIXTURE_LINK_RE.match(a["href"]):
            link = a
            break
    if link is None:
        return None

    m = FIXTURE_LINK_RE.match(link["href"])
    uuid = m.group(1)
    source_url = f"https://ekstraklasa.org{link['href']}"

    # Nazwy druzyn - <p> z klasa 'uppercase', ale ODRZUCAMY skroty (max 4 znaki)
    team_names = []
    for p in fixture_block.find_all("p"):
        cls = p.get("class") or []
        if "uppercase" not in " ".join(cls):
            continue
        txt = p.get_text(strip=True)
        if not txt or len(txt) <= 4:
            continue
        if re.match(r"^\d", txt):
            continue
        if txt.lower() in ("dni", "godz", "min", "sek", "kup bilet"):
            continue
        if txt in team_names:
            continue
        team_names.append(txt)

    if len(team_names) < 2:
        return None

    home_raw = team_names[0]
    away_raw = team_names[1]
    home = resolve_team_name(home_raw, name_map)
    away = resolve_team_name(away_raw, name_map)

    block_text = fixture_block.get_text(" ", strip=True)

    data_planowana = None
    godzina = None

    time_match = re.search(r"(\d{1,2}):(\d{2})\s+(\d{1,2})\.(\d{1,2})", block_text)
    if time_match:
        godzina = f"{int(time_match.group(1)):02d}:{int(time_match.group(2)):02d}"
        day = int(time_match.group(3))
        month = int(time_match.group(4))
        left_year = int(sezon.split("/")[0])
        right_year = left_year + 1
        year = left_year if month >= 7 else right_year
        try:
            data_planowana = datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass

    stadion = None
    for p in fixture_block.find_all(["p", "span", "div"]):
        txt = p.get_text(strip=True)
        if not txt or len(txt) > 120:
            continue
        low = txt.lower()
        if any(k in low for k in ["stadion", "arena", "obiekt"]) and "," in txt:
            stadion = txt
            break

    return {
        "fixture_id": uuid,
        "sezon": sezon,
        "kolejka": kolejka,
        "gospodarz": home,
        "gosc": away,
        "gospodarz_raw": home_raw,
        "gosc_raw": away_raw,
        "data_planowana": data_planowana,
        "godzina": godzina,
        "stadion": stadion,
        "source_url": source_url,
        "ekstraklasa_uuid": uuid,
    }


def extract_fixtures_from_html(html, sezon, kolejka, name_map):
    soup = BeautifulSoup(html, "html.parser")
    fixtures = []
    seen_uuids = set()

    for a in soup.find_all("a", href=True):
        if not FIXTURE_LINK_RE.match(a["href"]):
            continue

        fixture_block = a.parent
        if fixture_block is None:
            continue
        fixture_block = fixture_block.parent
        if fixture_block is None:
            continue
        fixture_block = fixture_block.parent

        if fixture_block is None:
            continue

        fixture = extract_fixture_from_block(fixture_block, sezon, kolejka, name_map)
        if fixture is None:
            continue

        if fixture["fixture_id"] in seen_uuids:
            continue
        seen_uuids.add(fixture["fixture_id"])

        fixtures.append(fixture)

    return fixtures


def scrape_kolejka(page, season_slug, kolejka, sezon, name_map):
    url = BASE_URL.format(season_slug=season_slug, kolejka=kolejka)
    print(f"  [{kolejka:2d}] {url}")

    page.goto(url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)

    html = page.content()
    return extract_fixtures_from_html(html, sezon, kolejka, name_map)


def upsert_fixtures(conn, fixtures):
    inserted = 0
    updated = 0

    for f in fixtures:
        existing = conn.execute(
            """
            SELECT fixture_id FROM fixtures_upcoming
            WHERE sezon = ? AND kolejka = ? AND gospodarz = ? AND gosc = ?
            """,
            (f["sezon"], f["kolejka"], f["gospodarz"], f["gosc"]),
        ).fetchone()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if existing is None:
            conn.execute(
                """
                INSERT INTO fixtures_upcoming (
                    fixture_id, sezon, kolejka, gospodarz, gosc,
                    data_planowana, godzina, stadion,
                    referee_full_name, source_url, ekstraklasa_uuid,
                    status, played_match_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'scheduled', NULL, ?, ?)
                """,
                (
                    f["fixture_id"], f["sezon"], f["kolejka"],
                    f["gospodarz"], f["gosc"],
                    f["data_planowana"], f["godzina"], f["stadion"],
                    f["source_url"], f["ekstraklasa_uuid"],
                    now, now,
                ),
            )
            inserted += 1
        else:
            conn.execute(
                """
                UPDATE fixtures_upcoming
                SET data_planowana = COALESCE(?, data_planowana),
                    godzina = COALESCE(?, godzina),
                    stadion = COALESCE(?, stadion),
                    source_url = ?,
                    ekstraklasa_uuid = ?,
                    updated_at = ?
                WHERE sezon = ? AND kolejka = ? AND gospodarz = ? AND gosc = ?
                """,
                (
                    f["data_planowana"], f["godzina"], f["stadion"],
                    f["source_url"], f["ekstraklasa_uuid"], now,
                    f["sezon"], f["kolejka"], f["gospodarz"], f["gosc"],
                ),
            )
            updated += 1

    conn.commit()
    return inserted, updated


def parse_kolejki_arg(arg):
    if not arg:
        return list(range(1, 35))
    if "-" in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in arg.split(",")]


def build_report(sezon, all_fixtures, stats, unknown_teams):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("SCRAPE FIXTURES REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {now}")
    lines.append(f"Sezon:     {sezon}")
    lines.append("")
    lines.append(f"Fixture zescrapowanych:   {stats['total']}")
    lines.append(f"Insert:                   {stats['inserted']}")
    lines.append(f"Update:                   {stats['updated']}")
    lines.append(f"Kolejki:                  {stats['kolejki']}")
    lines.append("")

    if unknown_teams:
        lines.append("UWAGA - nierozpoznane nazwy druzyn:")
        for t in sorted(unknown_teams):
            lines.append(f"  - {t}")
        lines.append("")

    lines.append("PROBKA (pierwsze 10 fixture):")
    for f in all_fixtures[:10]:
        lines.append(
            f"  K{f['kolejka']:02d} {f['gospodarz']:<28} vs {f['gosc']:<28} | "
            f"{f['data_planowana']} {f['godzina']} | uuid={f['ekstraklasa_uuid'][:8]}"
        )

    lines.append("")
    lines.append("PODZIAL PER KOLEJKA:")
    per_kolejka = {}
    for f in all_fixtures:
        per_kolejka.setdefault(f["kolejka"], 0)
        per_kolejka[f["kolejka"]] += 1
    for k in sorted(per_kolejka.keys()):
        lines.append(f"  Kolejka {k:2d}: {per_kolejka[k]} meczow")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nRaport: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True)
    parser.add_argument("--kolejki", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    season_slug = season_to_slug(args.sezon)
    kolejki = parse_kolejki_arg(args.kolejki)

    print(f"Sezon:        {args.sezon}")
    print(f"Season slug:  {season_slug}")
    print(f"Kolejki:      {kolejki[0]}-{kolejki[-1]} ({len(kolejki)} razem)")
    print(f"Dry run:      {args.dry_run}")
    print()

    if not DB_PATH.exists():
        print(f"Blad: baza {DB_PATH} nie istnieje.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    name_map = load_team_name_map(conn)
    print(f"Zaladowano {len(name_map)} nazw druzyn z bazy + aliasow.")

    all_fixtures = []
    unknown_teams = set()
    stats = {"total": 0, "inserted": 0, "updated": 0, "kolejki": 0}

    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/122.0"
            )
            page = context.new_page()

            for kolejka in kolejki:
                try:
                    fixtures = scrape_kolejka(page, season_slug, kolejka, args.sezon, name_map)
                except Exception as e:
                    print(f"    BLAD kolejka {kolejka}: {e}")
                    continue

                if not fixtures:
                    print(f"    Brak fixture w kolejce {kolejka}")
                    continue

                for f in fixtures:
                    if normalize_ascii(f["gospodarz_raw"]) not in name_map:
                        unknown_teams.add(f["gospodarz_raw"])
                    if normalize_ascii(f["gosc_raw"]) not in name_map:
                        unknown_teams.add(f["gosc_raw"])

                all_fixtures.extend(fixtures)
                stats["total"] += len(fixtures)
                stats["kolejki"] += 1

                if not args.dry_run:
                    ins, upd = upsert_fixtures(conn, fixtures)
                    stats["inserted"] += ins
                    stats["updated"] += upd
                    print(f"    OK: {len(fixtures)} fixture (insert={ins}, update={upd})")
                else:
                    print(f"    DRY: {len(fixtures)} fixture (bez zapisu)")

                time.sleep(1.0)

            browser.close()

    finally:
        conn.close()

    build_report(args.sezon, all_fixtures, stats, unknown_teams)


if __name__ == "__main__":
    main()