"""
link_and_update_results.py
==========================
Etap A: Linkowanie i update statusu meczów z Flashscore.

Wchodzi na /wyniki/ Ekstraklasy, zbiera mid + wynik + data + drużyny,
matchuje do fixtures_upcoming, aktualizuje flash_id, flash_url, status.

Parser tekstu: 'DD.MM. HH:MM HOME AWAY G_H G_A' z pełnym słownikiem drużyn.

NIE pobiera statystyk meczowych - to robi osobny skrypt scrape_match_stats.py.

Użycie:
python scripts/scraping/flashscore/link_and_update_results.py --sezon 2026/27 --dry-run
python scripts/scraping/flashscore/link_and_update_results.py --sezon 2026/27
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright


DB_PATH = Path("db/ekstraklasa.db")
DEBUG_DIR = Path("data/processed/flashscore_debug")

FLASH_RESULTS_URL = "https://www.flashscore.pl/pilka-nozna/polska/ekstraklasa/wyniki/"

# Wszystkie drużyny Ekstraklasy 2026/27 (nazwy z Flashscore)
TEAMS = [
    "Cracovia",
    "GKS Katowice",
    "Górnik Zabrze",
    "Jagiellonia Białystok",
    "Korona Kielce",
    "Lech Poznań",
    "Legia Warszawa",
    "Motor Lublin",
    "Piast Gliwice",
    "Pogoń Szczecin",
    "Radomiak Radom",
    "Raków Częstochowa",
    "Widzew Łódź",
    "Wisła Kraków",
    "Wisła Płock",
    "Wieczysta Kraków",
    "Zagłębie Lubin",
    "Śląsk Wrocław",
]

# Sortuj po długości descending - żeby najpierw dopasować "Wisła Kraków" a nie "Wisła"
TEAMS_SORTED = sorted(TEAMS, key=len, reverse=True)


def parse_match_line(text: str) -> Optional[Dict]:
    """
    Parsuje: '02.08. 20:15 GKS Katowice Radomiak Radom 3 1'
    Zwraca: {home, away, gole_home, gole_away, date_dd, date_mm, godzina}
    """
    m = re.match(r"^(\d{2})\.(\d{2})\.\s+(\d{2}:\d{2})\s+(.+?)\s+(\d+)\s+(\d+)$", text)
    if not m:
        return None
    
    dd, mm, godzina, teams_str, g1, g2 = m.groups()
    g1, g2 = int(g1), int(g2)
    
    home = None
    remaining = None
    for team in TEAMS_SORTED:
        if teams_str.startswith(team):
            home = team
            remaining = teams_str[len(team):].strip()
            break
    
    if home is None:
        return None
    
    away = None
    for team in TEAMS_SORTED:
        if remaining == team:
            away = team
            break
    
    if away is None:
        return None
    
    return {
        "home": home,
        "away": away,
        "gole_home": g1,
        "gole_away": g2,
        "date_dd": dd,
        "date_mm": mm,
        "godzina": godzina,
    }


def accept_cookies(page) -> bool:
    for sel in ["button:has-text('Akceptuję')", "#onetrust-accept-btn-handler"]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=2500)
                page.wait_for_timeout(1500)
                return True
        except Exception:
            pass
    return False


def click_show_more(page, max_clicks=50) -> int:
    clicks = 0
    for _ in range(max_clicks):
        clicked = False
        for sel in [
            "button:has-text('Pokaż więcej meczów')",
            "a:has-text('Pokaż więcej')",
        ]:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    loc.first.click(timeout=2500)
                    page.wait_for_timeout(1500)
                    clicks += 1
                    clicked = True
                    break
            except Exception:
                pass
        if not clicked:
            break
    return clicks


def fetch_flash_results(url: str, headless: bool = True) -> List[Dict]:
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
        print(f"Cookies: {'zaakceptowane' if accepted else 'brak'}")
        
        page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.5)")
        page.wait_for_timeout(1500)
        
        clicks = click_show_more(page, max_clicks=30)
        print(f'Klikniecia "pokaz wiecej": {clicks}')
        page.wait_for_timeout(2000)
        
        data = page.evaluate("""
            () => {
                const items = [];
                const rows = document.querySelectorAll('[id^="g_1_"]');
                rows.forEach(el => {
                    const id = el.id.replace('g_1_', '');
                    const text = (el.innerText || '').replace(/\\s+/g, ' ').trim();
                    const anchor = el.querySelector('a[href*="/mecz/"]');
                    const href = anchor ? anchor.getAttribute('href') : null;
                    items.push({mid: id, text: text, href: href});
                });
                return items;
            }
        """)
        
        browser.close()
        return data


def build_full_url(href: str, mid: str) -> str:
    if not href:
        return f"https://www.flashscore.pl/?mid={mid}"
    if href.startswith("/"):
        href = f"https://www.flashscore.pl{href}"
    if "?" in href:
        return f"{href}&mid={mid}"
    return f"{href}?mid={mid}"


def year_from_season(sezon: str, mm: str) -> int:
    """Sezon 2026/27: mecze lipiec-grudzień = 2026, styczeń-czerwiec = 2027"""
    year = int(sezon.split("/")[0])
    if int(mm) < 7:
        year += 1
    return year


def match_flash_to_fixtures(
    flash_matches: List[Dict],
    fixtures: pd.DataFrame,
    sezon: str,
) -> List[Dict]:
    parsed_flash = []
    for f in flash_matches:
        parsed = parse_match_line(f["text"])
        if not parsed:
            continue
        
        year = year_from_season(sezon, parsed["date_mm"])
        data_meczu = f"{year}-{parsed['date_mm']}-{parsed['date_dd']}"
        
        parsed_flash.append({
            "mid": f["mid"],
            "href": f["href"],
            "data_meczu": data_meczu,
            "godzina": parsed["godzina"],
            "home": parsed["home"],
            "away": parsed["away"],
            "gole_home": parsed["gole_home"],
            "gole_away": parsed["gole_away"],
            "text": f["text"],
        })
    
    results = []
    for _, fx in fixtures.iterrows():
        home_fx = fx["gospodarz"]
        away_fx = fx["gosc"]
        data_fx = fx["data_planowana"]
        
        # UWAGA: matchowanie IGNORUJE datę, bo Flashscore ma rzeczywiste daty
        # a fixtures_upcoming ma daty planowane (mecze się często przesuwają).
        # W jednej kolejce nie ma duplikatów par drużyn, więc to bezpieczne.
        found = None
        for pf in parsed_flash:
            if pf["home"] == home_fx and pf["away"] == away_fx:
                found = pf
                break
        
        results.append({
            "fixture_id": fx["fixture_id"],
            "sezon": fx["sezon"],
            "kolejka": fx["kolejka"],
            "gospodarz": home_fx,
            "gosc": away_fx,
            "data_planowana": data_fx,
            "matched": found is not None,
            "flash_mid": found["mid"] if found else None,
            "flash_url": build_full_url(found["href"], found["mid"]) if found else None,
            "gole_home": found["gole_home"] if found else None,
            "gole_away": found["gole_away"] if found else None,
            "current_status": fx.get("status"),
        })
    
    return results


def get_fixtures(conn: sqlite3.Connection, sezon: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM fixtures_upcoming WHERE sezon=? ORDER BY kolejka, data_planowana",
        conn,
        params=(sezon,)
    )


def update_fixtures(conn: sqlite3.Connection, results: List[Dict], dry_run: bool) -> int:
    updated = 0
    for r in results:
        if not r["matched"]:
            continue
        
        if not dry_run:
            conn.execute("""
                UPDATE fixtures_upcoming
                SET flash_id = ?,
                    flash_url = ?,
                    status = 'played',
                    gole_home = ?,
                    gole_away = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE fixture_id = ?
            """, (r["flash_mid"], r["flash_url"], r["gole_home"], r["gole_away"], r["fixture_id"]))
        updated += 1
    
    if not dry_run:
        conn.commit()
    
    return updated


def save_debug(results: List[Dict], flash_matches: List[Dict]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(results).to_csv(DEBUG_DIR / f"link_results_{ts}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flash_matches).to_csv(DEBUG_DIR / f"link_flash_raw_{ts}.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    
    conn = sqlite3.connect(DB_PATH)
    
    print("=" * 78)
    print("LINK & UPDATE RESULTS - FLASHSCORE")
    print("=" * 78)
    print(f"Sezon:   {args.sezon}")
    print(f"Dry-run: {args.dry_run}")
    print()
    
    flash_matches = fetch_flash_results(FLASH_RESULTS_URL, headless=not args.headed)
    print(f"Pobrano z Flashscore: {len(flash_matches)}")
    
    fixtures = get_fixtures(conn, args.sezon)
    print(f"Fixtures w bazie:     {len(fixtures)}")
    print()
    
    results = match_flash_to_fixtures(flash_matches, fixtures, args.sezon)
    matched = [r for r in results if r["matched"]]
    
    print(f"Zmatchowano: {len(matched)}")
    print()
    print("MATCHED:")
    print("-" * 78)
    for r in matched:
        marker = " [już played]" if r["current_status"] == "played" else ""
        print(f"K{r['kolejka']:02d} {r['gospodarz']:<25} {r['gole_home']}-{r['gole_away']} {r['gosc']:<25} mid={r['flash_mid']}{marker}")
    
    updated = update_fixtures(conn, results, args.dry_run)
    save_debug(results, flash_matches)
    
    print()
    print("=" * 78)
    print(f"Zaktualizowano: {updated}")
    print(f"Dry-run:        {args.dry_run}")
    print("=" * 78)
    
    conn.close()


if __name__ == "__main__":
    main()