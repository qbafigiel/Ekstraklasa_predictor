"""
link_fixtures_flashscore.py
===========================
Stabilny linker fixtures_upcoming -> Flashscore.

Podejście:
- otwieramy stronę terminarza Ekstraklasy na Flashscore przez Playwright
- czytamy CAŁY tekst z wierszy meczowych div[id^='g_1_']
- nie polegamy na kruchych selektorach nazw drużyn
- matchujemy po aliasach nazw klubów
- zapisujemy flash_id + flash_url do fixtures_upcoming
- zapisujemy debug CSV z widokiem tego, co naprawdę odczytaliśmy z Flashscore

To jest wersja produkcyjna v1 pod deadline.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


DB_PATH = Path("db/ekstraklasa.db")
DEBUG_DIR = Path("data/processed/flashscore_link_debug")

FLASH_URL_DEFAULT = "https://www.flashscore.pl/pilka-nozna/polska/ekstraklasa/terminarz/"

POLISH_L_MAP = str.maketrans({
    "ł": "l",
    "Ł": "L",
})

# Aliasy pod matchowanie tekstu z wiersza Flashscore
TEAM_ALIASES = {
    "Cracovia": ["cracovia"],
    "GKS Katowice": ["gks katowice", "katowice"],
    "Górnik Zabrze": ["gornik zabrze", "gornik"],
    "Jagiellonia Białystok": ["jagiellonia bialystok", "jagiellonia"],
    "Korona Kielce": ["korona kielce", "korona"],
    "Lech Poznań": ["lech poznan", "lech"],
    "Legia Warszawa": ["legia warszawa", "legia"],
    "Motor Lublin": ["motor lublin", "motor"],
    "Piast Gliwice": ["piast gliwice", "piast"],
    "Pogoń Szczecin": ["pogon szczecin", "pogon"],
    "Radomiak Radom": ["radomiak radom", "radomiak"],
    "Raków Częstochowa": ["rakow czestochowa", "rakow"],
    "Widzew Łódź": ["widzew lodz", "widzew"],
    "Wisła Kraków": ["wisla krakow"],
    "Wisła Płock": ["wisla plock", "plock"],
    "Wieczysta Kraków": ["wieczysta krakow", "wieczysta"],
    "Zagłębie Lubin": ["zaglebie lubin", "zaglebie"],
    "Śląsk Wrocław": ["slask wroclaw", "slask"],
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).translate(POLISH_L_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def ensure_flash_columns(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fixtures_upcoming)").fetchall()]

    if "flash_id" not in cols:
        conn.execute("ALTER TABLE fixtures_upcoming ADD COLUMN flash_id TEXT")
        print("Dodano kolumne: fixtures_upcoming.flash_id")

    if "flash_url" not in cols:
        conn.execute("ALTER TABLE fixtures_upcoming ADD COLUMN flash_url TEXT")
        print("Dodano kolumne: fixtures_upcoming.flash_url")

    conn.commit()


def get_fixture_rows(conn: sqlite3.Connection, sezon: str, kolejka: int) -> List[Dict[str, Any]]:
    df = pd.read_sql_query(
        """
        SELECT *
        FROM fixtures_upcoming
        WHERE sezon = ? AND kolejka = ?
        ORDER BY data_planowana, godzina, gospodarz
        """,
        conn,
        params=(sezon, kolejka),
    )
    return df.to_dict(orient="records")


def get_aliases(team_name: str) -> List[str]:
    aliases = TEAM_ALIASES.get(team_name, [normalize_text(team_name)])
    aliases = [normalize_text(x) for x in aliases if normalize_text(x)]
    # deduplikacja zachowująca kolejność
    seen = set()
    out = []
    for a in aliases:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def accept_cookies(page) -> bool:
    selectors = [
        "button:has-text('Akceptuję')",
        "button:has-text('Akceptuj')",
        "button:has-text('Accept')",
        "#onetrust-accept-btn-handler",
        "[id*='accept']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=2500)
                page.wait_for_timeout(1200)
                return True
        except Exception:
            pass
    return False


def click_show_more(page, max_clicks: int = 12) -> int:
    clicks = 0
    selectors = [
        "button:has-text('Pokaż więcej meczów')",
        "button:has-text('Pokaż więcej')",
        "a:has-text('Pokaż więcej')",
        "div:has-text('Pokaż więcej meczów')",
        "div:has-text('Show more matches')",
    ]

    for _ in range(max_clicks):
        clicked = False
        for sel in selectors:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.click(timeout=2500)
                    page.wait_for_timeout(1200)
                    clicks += 1
                    clicked = True
                    break
            except Exception:
                pass
        if not clicked:
            break

    return clicks


def safe_inner_text(locator) -> str:
    try:
        return locator.inner_text(timeout=3000).strip()
    except Exception:
        return ""


def safe_inner_html(locator) -> str:
    try:
        return locator.inner_html(timeout=3000)
    except Exception:
        return ""


def safe_attr(locator, name: str) -> Optional[str]:
    try:
        return locator.get_attribute(name, timeout=3000)
    except Exception:
        return None


def extract_anchor_href_from_row(row_locator) -> Optional[str]:
    candidates = [
        "a[href*='/mecz/']",
        "a[href*='/match/']",
        "a",
    ]
    for sel in candidates:
        try:
            loc = row_locator.locator(sel)
            if loc.count() > 0:
                href = loc.first.get_attribute("href", timeout=2000)
                if href:
                    return href
        except Exception:
            pass
    return None


def build_flash_url(flash_id: str, href: Optional[str]) -> str:
    if href:
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.startswith("/"):
            return f"https://www.flashscore.pl{href}"
    return f"https://www.flashscore.pl/mecz/{flash_id}/#/szczegoly/statystyki"


def fetch_flashscore_rows(url: str, headless: bool = True) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1200},
            locale="pl-PL",
        )
        page = context.new_page()

        print(f"Otwieram: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)

        accepted = accept_cookies(page)
        if accepted:
            print("Cookies: zaakceptowane")
        else:
            print("Cookies: brak / juz zaakceptowane")

        # lekki scroll i ewentualne dociągnięcie większej liczby wierszy
        page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.35)")
        page.wait_for_timeout(1500)

        try:
            page.wait_for_selector("div[id^='g_1_']", timeout=30000)
        except PlaywrightTimeoutError:
            browser.close()
            raise RuntimeError("Nie znaleziono wierszy meczowych div[id^='g_1_'] na stronie Flashscore.")

        more_clicks = click_show_more(page)
        if more_clicks > 0:
            print(f'Klikniecia "pokaz wiecej": {more_clicks}')

        page.wait_for_timeout(1500)

        rows = page.locator("div[id^='g_1_']")
        count = rows.count()
        print(f"Znaleziono wierszy meczowych: {count}")

        seen_ids = set()

        for i in range(count):
            row = rows.nth(i)
            row_id = safe_attr(row, "id") or ""
            flash_id = row_id.replace("g_1_", "").strip()

            if not flash_id or flash_id in seen_ids:
                continue

            row_text = safe_inner_text(row)
            row_html = safe_inner_html(row)
            href = extract_anchor_href_from_row(row)
            flash_url = build_flash_url(flash_id, href)
            row_text_norm = normalize_text(row_text)

            rows_out.append({
                "flash_id": flash_id,
                "flash_url": flash_url,
                "row_text": row_text,
                "row_text_norm": row_text_norm,
                "row_html_preview": row_html[:300],
            })
            seen_ids.add(flash_id)

        browser.close()

    return rows_out


def score_row_for_fixture(row_text_norm: str, home_team: str, away_team: str) -> int:
    home_aliases = get_aliases(home_team)
    away_aliases = get_aliases(away_team)

    best_home = 0
    best_away = 0

    for idx, alias in enumerate(home_aliases):
        if alias and alias in row_text_norm:
            best_home = max(best_home, 100 if idx == 0 else 70)

    for idx, alias in enumerate(away_aliases):
        if alias and alias in row_text_norm:
            best_away = max(best_away, 100 if idx == 0 else 70)

    if best_home == 0 or best_away == 0:
        return 0

    return best_home + best_away


def match_fixtures_to_rows(
    fixtures: List[Dict[str, Any]],
    fs_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    used_flash_ids = set()
    results = []

    for fx in fixtures:
        home = fx["gospodarz"]
        away = fx["gosc"]

        candidates = []
        for row in fs_rows:
            if row["flash_id"] in used_flash_ids:
                continue

            score = score_row_for_fixture(row["row_text_norm"], home, away)
            if score > 0:
                candidates.append((score, row))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            results.append({
                "fixture_id": fx["fixture_id"],
                "gospodarz": home,
                "gosc": away,
                "matched": False,
                "flash_id": None,
                "flash_url": None,
                "score": 0,
                "row_text": None,
            })
            continue

        best_score, best_row = candidates[0]
        used_flash_ids.add(best_row["flash_id"])

        results.append({
            "fixture_id": fx["fixture_id"],
            "gospodarz": home,
            "gosc": away,
            "matched": True,
            "flash_id": best_row["flash_id"],
            "flash_url": best_row["flash_url"],
            "score": best_score,
            "row_text": best_row["row_text"],
        })

    return results


def save_debug_files(kolejka: int, fs_rows: List[Dict[str, Any]], matches: List[Dict[str, Any]]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    fs_path = DEBUG_DIR / f"flashscore_rows_K{kolejka:02d}.csv"
    mt_path = DEBUG_DIR / f"flashscore_matches_K{kolejka:02d}.csv"

    pd.DataFrame(fs_rows).to_csv(fs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(matches).to_csv(mt_path, index=False, encoding="utf-8-sig")

    print(f"DEBUG rows:    {fs_path}")
    print(f"DEBUG matches: {mt_path}")


def update_db(
    conn: sqlite3.Connection,
    matched_rows: List[Dict[str, Any]],
    dry_run: bool = False,
) -> int:
    updated = 0
    for row in matched_rows:
        if not row["matched"]:
            continue

        if not dry_run:
            conn.execute(
                """
                UPDATE fixtures_upcoming
                SET flash_id = ?, flash_url = ?, updated_at = CURRENT_TIMESTAMP
                WHERE fixture_id = ?
                """,
                (row["flash_id"], row["flash_url"], row["fixture_id"]),
            )
        updated += 1

    if not dry_run:
        conn.commit()

    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", default="2026/27")
    parser.add_argument("--kolejka", type=int, required=True)
    parser.add_argument("--url", default=FLASH_URL_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headed", action="store_true", help="Uruchom przegladarke z oknem")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        ensure_flash_columns(conn)

        fixtures = get_fixture_rows(conn, args.sezon, args.kolejka)
        print("=" * 78)
        print("LINK FIXTURES -> FLASHSCORE")
        print("=" * 78)
        print(f"Sezon:     {args.sezon}")
        print(f"Kolejka:   {args.kolejka}")
        print(f"Fixtures:  {len(fixtures)}")
        print(f"Dry-run:   {args.dry_run}")
        print(f"Headed:    {args.headed}")
        print("")

        if not fixtures:
            print("Brak fixtures do linkowania.")
            return

        fs_rows = fetch_flashscore_rows(args.url, headless=not args.headed)

        print("")
        print("PODGLAD WIERSZY FLASHSCORE")
        print("-" * 78)
        for i, row in enumerate(fs_rows[:15], start=1):
            preview = row["row_text"].replace("\n", " | ")[:140]
            print(f"[{i:02d}] {row['flash_id']} | {preview}")

        matched_rows = match_fixtures_to_rows(fixtures, fs_rows)

        print("")
        print("WYNIK MATCHOWANIA")
        print("-" * 78)
        ok = 0
        miss = 0
        for row in matched_rows:
            if row["matched"]:
                ok += 1
                print(f"OK   | {row['gospodarz']} vs {row['gosc']} -> {row['flash_id']} | score={row['score']}")
                print(f"     | row: {row['row_text'].replace(chr(10), ' | ')[:180]}")
            else:
                miss += 1
                print(f"BRAK | {row['gospodarz']} vs {row['gosc']}")

        updated = update_db(conn, matched_rows, dry_run=args.dry_run)
        save_debug_files(args.kolejka, fs_rows, matched_rows)

        print("")
        print("=" * 78)
        print("PODSUMOWANIE")
        print("=" * 78)
        print(f"Zmatchowane: {ok}")
        print(f"Braki:       {miss}")
        print(f"Zapisane DB: {updated}")
        print(f"Dry-run:     {args.dry_run}")
        print("=" * 78)

    finally:
        conn.close()


if __name__ == "__main__":
    main()