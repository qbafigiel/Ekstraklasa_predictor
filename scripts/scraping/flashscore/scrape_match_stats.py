"""
scrape_match_stats.py
=====================
Etap B: Scrapowanie statystyk meczowych z Flashscore.

Bierze mecze status='played' z fixtures_upcoming które nie mają jeszcze
rekordu w matches. Wchodzi na każdy flash_url, parsuje statystyki
(MAPA_STAT z scraper_flashscore.py) i wpisuje do matches.

Aktualizuje fixtures_upcoming.played_match_id na nowy match_id.

Użycie:
python scripts/scraping/flashscore/scrape_match_stats.py --sezon 2026/27 --dry-run
python scripts/scraping/flashscore/scrape_match_stats.py --sezon 2026/27
python scripts/scraping/flashscore/scrape_match_stats.py --sezon 2026/27 --kolejka 1
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from playwright.sync_api import sync_playwright


DB_PATH = Path("db/ekstraklasa.db")
DEBUG_DIR = Path("data/processed/flashscore_debug")

WAGI_SEZONOW = {
    "2023/24": 0.4,
    "2024/25": 0.7,
    "2025/26": 1.0,
    "2026/27": 1.0,
}

# Mapowanie statystyk Flashscore -> kolumny bazy
MAPA_STAT = {
    "Oczekiwane gole (xG)":               ("xg_gosp", "xg_gosc"),
    "Posiadanie piłki":                    ("posiadanie_gosp", "posiadanie_gosc"),
    "Strzały łącznie":                     ("strzaly_gosp", "strzaly_gosc"),
    "Strzały na bramkę":                   ("celne_gosp", "celne_gosc"),
    "Strzały niecelne":                    ("strzaly_niecelne_gosp", "strzaly_niecelne_gosc"),
    "Strzały zablokowane":                 ("strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc"),
    "Rzuty rożne":                         ("rozne_gosp", "rozne_gosc"),
    "Spalone":                             ("spalone_gosp", "spalone_gosc"),
    "Podania":                             ("podania_gosp", "podania_gosc"),
    "Dośrodkowania":                       ("dosrodkowania_gosp", "dosrodkowania_gosc"),
    "Faule":                               ("faule_gosp", "faule_gosc"),
    "Próby odbioru piłki":                 ("odbiory_gosp", "odbiory_gosc"),
    "Żółte kartki":                        ("zk_gosp", "zk_gosc"),
    "Czerwone kartki":                     ("czk_gosp", "czk_gosc"),
}

NAGLOWKI = {
    "TOP STATYSTYKI", "STRZAŁY", "ATAK", "PODANIA", "OBRONA",
    "STATYSTYKI BRAMKARZA", "KURSY", "MECZ", "1. POŁOWA", "2. POŁOWA",
    "SZCZEGÓŁY", "STATYSTYKI", "SKŁADY", "STATYSTYKI ZAWODNIKÓW",
}

DOMYSLNE_ZERO = {
    "czk_gosp", "czk_gosc",
    "zk_gosp", "zk_gosc",
}


def wyciagnij_liczbe(tekst: str) -> Optional[str]:
    tekst = tekst.strip()
    m = re.search(r"\((\d+)/\d+\)", tekst)
    if m:
        return m.group(1)
    m = re.match(r"^-?[\d.]+$", tekst)
    if m:
        return tekst
    return None


def znajdz_wstecz(linie: List[str], od: int) -> Optional[str]:
    j = od - 1
    while j >= max(0, od - 6):
        k = linie[j].strip()
        if k and k not in MAPA_STAT and k not in NAGLOWKI:
            if re.match(r"^\d+%$", k):
                prev = linie[j - 1].strip() if j > 0 else ""
                m = re.search(r"\((\d+)/\d+\)", prev)
                if m:
                    return m.group(1)
                return k.replace("%", "")
            wynik = wyciagnij_liczbe(k)
            if wynik is not None:
                return wynik
        j -= 1
    return None


def znajdz_wprzod(linie: List[str], od: int) -> Optional[str]:
    j = od + 1
    while j < min(len(linie), od + 6):
        k = linie[j].strip()
        if k and k not in MAPA_STAT and k not in NAGLOWKI:
            if re.match(r"^\d+%$", k):
                nast = linie[j + 1].strip() if j + 1 < len(linie) else ""
                m = re.search(r"\((\d+)/\d+\)", nast)
                if m:
                    return m.group(1)
                return k.replace("%", "")
            wynik = wyciagnij_liczbe(k)
            if wynik is not None:
                return wynik
        j += 1
    return None


def parsuj_statystyki(linie: List[str]) -> Dict[str, str]:
    statystyki = {}
    przetworzone = set()
    
    for i, linia in enumerate(linie):
        nazwa = linia.strip()
        if nazwa not in MAPA_STAT or nazwa in przetworzone:
            continue
        
        klucz_gosp, klucz_gosc = MAPA_STAT[nazwa]
        val_gosp = znajdz_wstecz(linie, i)
        val_gosc = znajdz_wprzod(linie, i)
        
        if val_gosp is not None and val_gosc is not None:
            statystyki[klucz_gosp] = val_gosp
            statystyki[klucz_gosc] = val_gosc
            przetworzone.add(nazwa)
    
    for klucz in DOMYSLNE_ZERO:
        if klucz not in statystyki:
            statystyki[klucz] = "0"
    
    return statystyki


def build_stats_url(flash_url: str) -> str:
    """Konwertuje URL meczu na URL statystyk.
    
    Wejscie: https://www.flashscore.pl/mecz/pilka-nozna/xxx/yyy/?mid=ABC
    Wyjscie: https://www.flashscore.pl/mecz/pilka-nozna/xxx/yyy/szczegoly/statystyki/?mid=ABC
    """
    if "?" not in flash_url:
        return flash_url + "szczegoly/statystyki/"
    
    base, query = flash_url.split("?", 1)
    base = base.rstrip("/")
    
    # Usun ewentualne duplikaty mid= w query
    import re as _re
    parts = query.split("&")
    seen = set()
    unique = []
    for p in parts:
        key = p.split("=")[0]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    query = "&".join(unique)
    
    return f"{base}/szczegoly/statystyki/?{query}"


def scrape_one_match(page, flash_url: str) -> Dict:
    """Scrapuje jeden mecz - zwraca dict ze statystykami."""
    stats_url = build_stats_url(flash_url)
    
    page.goto(stats_url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(4000)
    
    try:
        page.click("button#onetrust-accept-btn-handler", timeout=2000)
        page.wait_for_timeout(1500)
    except Exception:
        pass
    
    # Czekaj dodatkowo aż JS załaduje statystyki
    page.wait_for_timeout(3000)
    
    tekst = page.inner_text("body")
    linie = [l.strip() for l in tekst.split("\n") if l.strip()]
    
    return parsuj_statystyki(linie)


def get_fixtures_to_scrape(conn: sqlite3.Connection, sezon: str, kolejka: Optional[int]) -> pd.DataFrame:
    """Bierze mecze status='played' bez played_match_id."""
    query = """
        SELECT * FROM fixtures_upcoming 
        WHERE sezon=? 
          AND status='played' 
          AND played_match_id IS NULL
          AND flash_url IS NOT NULL
    """
    params = [sezon]
    
    if kolejka is not None:
        query += " AND kolejka=?"
        params.append(kolejka)
    
    query += " ORDER BY kolejka, data_planowana"
    
    return pd.read_sql_query(query, conn, params=params)


def next_match_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(match_id), 0) + 1 FROM matches").fetchone()
    return int(row[0])


def get_gole_from_fixture(conn: sqlite3.Connection, fixture_id: str) -> tuple:
    """
    Wynik meczu jest w fixtures_upcoming? Nie - został tylko w debug CSV.
    Musimy pobrać go ze strony samego meczu. Ale mamy alternatywę - z fixtures
    nie mamy wyniku, więc parsujemy ze scrapera osobno.
    """
    return None, None


def parse_score_from_page(page) -> tuple:
    """Wyciąga wynik ze strony meczu."""
    tekst = page.inner_text("body")
    # Format wyniku na Flashscore: często "1 - 2" lub "1:2"
    # Szukamy w kontekście: nazwy drużyn + wynik
    # Prostsza metoda: bierzemy z tekstu meczów w /wyniki/ - ale musimy to zrobić inaczej
    
    # Sprawdź w tekście linii format "X - Y" gdzie X, Y to małe liczby
    linie = [l.strip() for l in tekst.split("\n") if l.strip()]
    
    # Wynik jest zazwyczaj wcześnie na stronie
    for i, l in enumerate(linie[:50]):
        m = re.match(r"^(\d{1,2})\s*[-:]\s*(\d{1,2})$", l)
        if m:
            g1, g2 = int(m.group(1)), int(m.group(2))
            if 0 <= g1 <= 15 and 0 <= g2 <= 15:
                return g1, g2
    return None, None


def build_match_row(fixture: pd.Series, stats: Dict, gole_home: int, gole_away: int) -> Dict:
    """Buduje wiersz do wstawienia do matches."""
    row = {
        "sezon": fixture["sezon"],
        "waga_sezonu": WAGI_SEZONOW.get(fixture["sezon"], 1.0),
        "kolejka": fixture["kolejka"],
        "data_meczu": fixture["data_planowana"],
        "gospodarz": fixture["gospodarz"],
        "gosc": fixture["gosc"],
        "gole_gosp": gole_home,
        "gole_gosc": gole_away,
        "flash_id": fixture["flash_id"],
        "flash_url": fixture["flash_url"],
    }
    
    # Dodaj wszystkie statystyki
    for key, val in stats.items():
        try:
            row[key] = float(val)
        except (ValueError, TypeError):
            row[key] = None
    
    return row


def insert_match(conn: sqlite3.Connection, row: Dict) -> int:
    """Wstawia wiersz do matches - zwraca match_id."""
    match_id = next_match_id(conn)
    row["match_id"] = match_id
    
    cols_in_matches = [r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()]
    valid_cols = [c for c in row.keys() if c in cols_in_matches]
    
    cols_str = ", ".join(valid_cols)
    placeholders = ", ".join(["?"] * len(valid_cols))
    values = [row.get(c) for c in valid_cols]
    
    conn.execute(f"INSERT INTO matches ({cols_str}) VALUES ({placeholders})", values)
    return match_id


def update_fixture_match_id(conn: sqlite3.Connection, fixture_id: str, match_id: int) -> None:
    conn.execute("""
        UPDATE fixtures_upcoming 
        SET played_match_id=?, updated_at=CURRENT_TIMESTAMP 
        WHERE fixture_id=?
    """, (match_id, fixture_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True)
    parser.add_argument("--kolejka", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    
    conn = sqlite3.connect(DB_PATH)
    
    print("=" * 78)
    print("SCRAPE MATCH STATS - FLASHSCORE")
    print("=" * 78)
    print(f"Sezon:   {args.sezon}")
    print(f"Kolejka: {args.kolejka}")
    print(f"Dry-run: {args.dry_run}")
    print()
    
    fixtures = get_fixtures_to_scrape(conn, args.sezon, args.kolejka)
    print(f"Meczów do zescrapowania: {len(fixtures)}")
    
    if len(fixtures) == 0:
        print("Nic do roboty.")
        conn.close()
        return
    
    print()
    
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=not args.headed)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1200},
            locale="pl-PL",
        )
        
        ok = 0
        failed = 0
        
        for idx, fx in fixtures.iterrows():
            home = fx["gospodarz"]
            away = fx["gosc"]
            kolejka = fx["kolejka"]
            url = fx["flash_url"]
            
            print(f"[{idx+1}/{len(fixtures)}] K{kolejka:02d} {home} vs {away}")
            print(f"  URL: {url}")
            
            try:
                page = context.new_page()
                stats = scrape_one_match(page, url)
                page.close()
                
                # Wynik czytamy z fixtures_upcoming (etap A go zapisal)
                gole_home = int(fx["gole_home"]) if pd.notna(fx.get("gole_home")) else None
                gole_away = int(fx["gole_away"]) if pd.notna(fx.get("gole_away")) else None
                
                if not stats:
                    print("  FAIL: brak statystyk")
                    failed += 1
                    continue
                
                if gole_home is None or gole_away is None:
                    print(f"  WARN: brak wyniku (znaleziono {len(stats)} statystyk)")
                    # Nie przerywamy - może wynik jest w stats
                
                print(f"  Wynik: {gole_home}-{gole_away}")
                print(f"  Statystyki: {len(stats)} znalezionych")
                print(f"    xG: {stats.get('xg_gosp', '?')}/{stats.get('xg_gosc', '?')}")
                print(f"    Kornery: {stats.get('rozne_gosp', '?')}/{stats.get('rozne_gosc', '?')}")
                print(f"    Faule: {stats.get('faule_gosp', '?')}/{stats.get('faule_gosc', '?')}")
                print(f"    ŻK: {stats.get('zk_gosp', '?')}/{stats.get('zk_gosc', '?')}")
                
                if not args.dry_run:
                    row = build_match_row(fx, stats, gole_home, gole_away)
                    match_id = insert_match(conn, row)
                    update_fixture_match_id(conn, fx["fixture_id"], match_id)
                    conn.commit()
                    print(f"  DB: zapisano match_id={match_id}")
                    ok += 1
                else:
                    print("  DRY-RUN: nie zapisuje")
                    ok += 1
                
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1
                try:
                    page.close()
                except Exception:
                    pass
            
            print()
            time.sleep(1)
        
        browser.close()
    
    print("=" * 78)
    print(f"OK: {ok}")
    print(f"FAILED: {failed}")
    print("=" * 78)
    
    conn.close()


if __name__ == "__main__":
    main()