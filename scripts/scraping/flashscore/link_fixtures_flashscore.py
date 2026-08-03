"""
link_fixtures_flashscore.py
===========================
Linkowanie fixtures_upcoming -> Flashscore na podstawie strony terminarza.

Wersja oparta na REALNEJ strukturze aktualnego Flashscore:
- czyta anchory a[href*="/mecz/"]
- używa tekstu anchorów + href do matchowania drużyn
- zapisuje flash_url do fixtures_upcoming
- flash_id zostawia NULL, bo na stronie terminarza nie ma już starego event ID
- to wystarczy do kolejnego kroku: scraper post-match może otwierać flash_url bezpośrednio

Przykład:
python scripts/scraping/flashscore/link_fixtures_flashscore.py --sezon 2026/27 --kolejka 1 --dry-run
python scripts/scraping/flashscore/link_fixtures_flashscore.py --sezon 2026/27 --kolejka 1
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright


DB_PATH = Path("db/ekstraklasa.db")
DEBUG_DIR = Path("data/processed/flashscore_link_debug")

FLASH_URL_DEFAULT = "https://www.flashscore.pl/pilka-nozna/polska/ekstraklasa/terminarz/"

POLISH_L_MAP = str.maketrans({
    "ł": "l",
    "Ł": "L",
})

TEAM_ALIASES = {
    "Cracovia": ["cracovia"],
    "GKS Katowice": ["gks katowice", "katowice"],
    "Górnik Zabrze": ["gornik zabrze", "gornik zabrze"],
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
    "Wisła Kraków": ["wisla krakow", "wisla"],
    "Wisła Płock": ["wisla plock", "plock"],
    "Wieczysta Kraków": ["wieczysta krakow", "ks wieczysta krakow", "wieczysta"],
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
    raw = TEAM_ALIASES.get(team_name, [team_name])
    out = []
    seen = set()
    for x in raw:
        nx = normalize_text(x)
        if nx and nx not in seen:
            out.append(nx)
            seen.add(nx)
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


def click_show_more(page, max_clicks: int = 5) -> int:
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


def make_absolute_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return f"https://www.flashscore.pl{href}"
    return f"https://www.flashscore.pl/{href}"


def fetch_schedule_anchors(url: str, headless: bool = True) -> List[Dict[str, Any]]:
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
        page.wait_for_timeout(3000)

        accepted = accept_cookies(page)
        print(f"Cookies: {'zaakceptowane' if accepted else 'brak / juz zaakceptowane'}")

        page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.35)")
        page.wait_for_timeout(1500)

        clicks = click_show_more(page, max_clicks=5)
        print(f'Klikniecia "pokaz wiecej": {clicks}')

        page.wait_for_timeout(2000)

        anchors = page.evaluate(
            """
            () => {
                const els = Array.from(document.querySelectorAll('a[href*="/mecz/"]'));
                return els.map(a => ({
                    href: a.getAttribute('href') || '',
                    text: (a.innerText || '').replace(/\\s+/g, ' ').trim(),
                    className: a.className || ''
                }));
            }
            """
        )

        browser.close()

    # filtracja i deduplikacja
    seen = set()
    for a in anchors:
        href = (a.get("href") or "").strip()
        text = (a.get("text") or "").strip()

        if not href:
            continue
        if "/mecz/" not in href:
            continue
        if " - " not in text:
            continue

        key = href
        if key in seen:
            continue
        seen.add(key)

        rows_out.append({
            "flash_id": None,
            "flash_url": make_absolute_url(href),
            "href": href,
            "href_norm": normalize_text(href),
            "anchor_text": text,
            "anchor_text_norm": normalize_text(text),
            "class_name": a.get("className", ""),
        })

    return rows_out


def score_team_aliases_in_text(team_name: str, combined_norm: str) -> int:
    aliases = get_aliases(team_name)
    best = 0
    for idx, alias in enumerate(aliases):
        if alias and alias in combined_norm:
            if idx == 0:
                best = max(best, 100)
            else:
                best = max(best, 75)
    return best


def split_anchor_text(anchor_text: str) -> Tuple[str, str]:
    if " - " not in anchor_text:
        return anchor_text.strip(), ""
    left, right = anchor_text.split(" - ", 1)
    return left.strip(), right.strip()


def score_anchor_for_fixture(anchor: Dict[str, Any], home_team: str, away_team: str) -> int:
    combined_norm = f"{anchor['anchor_text_norm']} | {anchor['href_norm']}"
    base_home = score_team_aliases_in_text(home_team, combined_norm)
    base_away = score_team_aliases_in_text(away_team, combined_norm)

    if base_home == 0 or base_away == 0:
        return 0

    score = base_home + base_away

    # bonus za poprawny układ HOME - AWAY w tekście anchora
    left, right = split_anchor_text(anchor["anchor_text"])
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)

    home_aliases = get_aliases(home_team)
    away_aliases = get_aliases(away_team)

    left_home = any(alias in left_norm for alias in home_aliases)
    right_away = any(alias in right_norm for alias in away_aliases)
    left_away = any(alias in left_norm for alias in away_aliases)
    right_home = any(alias in right_norm for alias in home_aliases)

    if left_home and right_away:
        score += 80

    if left_away and right_home:
        score -= 60

    return score


def match_fixtures_to_anchors(
    fixtures: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results = []
    used_hrefs = set()

    for fx in fixtures:
        home = fx["gospodarz"]
        away = fx["gosc"]

        candidates = []
        for a in anchors:
            if a["href"] in used_hrefs:
                continue
            score = score_anchor_for_fixture(a, home, away)
            if score > 0:
                candidates.append((score, a))

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
                "anchor_text": None,
                "href": None,
            })
            continue

        best_score, best_anchor = candidates[0]
        used_hrefs.add(best_anchor["href"])

        results.append({
            "fixture_id": fx["fixture_id"],
            "gospodarz": home,
            "gosc": away,
            "matched": True,
            "flash_id": None,
            "flash_url": best_anchor["flash_url"],
            "score": best_score,
            "anchor_text": best_anchor["anchor_text"],
            "href": best_anchor["href"],
        })

    return results


def save_debug_files(kolejka: int, anchors: List[Dict[str, Any]], matches: List[Dict[str, Any]]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    anchors_path = DEBUG_DIR / f"flashscore_anchors_K{kolejka:02d}.csv"
    matches_path = DEBUG_DIR / f"flashscore_matches_K{kolejka:02d}.csv"

    pd.DataFrame(anchors).to_csv(anchors_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(matches).to_csv(matches_path, index=False, encoding="utf-8-sig")

    print(f"DEBUG anchors: {anchors_path}")
    print(f"DEBUG matches: {matches_path}")


def update_db(conn: sqlite3.Connection, matched_rows: List[Dict[str, Any]], dry_run: bool = False) -> int:
    updated = 0

    for row in matched_rows:
        if not row["matched"]:
            continue

        if not dry_run:
            conn.execute(
                """
                UPDATE fixtures_upcoming
                SET flash_id = ?,
                    flash_url = ?,
                    updated_at = CURRENT_TIMESTAMP
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
    parser.add_argument("--headed", action="store_true")
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

        anchors = fetch_schedule_anchors(args.url, headless=not args.headed)

        print(f"Anchorow /mecz/: {len(anchors)}")
        print("")
        print("PODGLAD ANCHOROW")
        print("-" * 78)
        for i, a in enumerate(anchors[:20], start=1):
            print(f"[{i:02d}] {a['anchor_text']}")
            print(f"     {a['href']}")

        matched_rows = match_fixtures_to_anchors(fixtures, anchors)

        print("")
        print("WYNIK MATCHOWANIA")
        print("-" * 78)
        ok = 0
        miss = 0

        for row in matched_rows:
            if row["matched"]:
                ok += 1
                print(f"OK   | {row['gospodarz']} vs {row['gosc']} | score={row['score']}")
                print(f"     | {row['anchor_text']}")
                print(f"     | {row['href']}")
            else:
                miss += 1
                print(f"BRAK | {row['gospodarz']} vs {row['gosc']}")

        updated = update_db(conn, matched_rows, dry_run=args.dry_run)
        save_debug_files(args.kolejka, anchors, matched_rows)

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