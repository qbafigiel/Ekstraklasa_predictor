"""
scrape_referees.py
==================
Scrapuje sędziów dla wszystkich meczów Ekstraklasy z Flashscore.

Źródło: zakładka statystyki (już otwarta w URL), klik Szczegóły, scroll.
Format: SĘDZIA: Nazwisko I. (Pol)

Tworzy tabelę match_referees w bazie:
  match_id, sezon, referee_raw, referee_name, referee_country, scraped_at

Checkpoint: pomija już zescrapowane mecze.
Rate limit: 2-4 sek przerwy między meczami.
"""

import sqlite3
import re
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

DB_PATH = Path("db/ekstraklasa.db")
REPORT_DIR = Path("data/reports/model")
REPORT_TXT = REPORT_DIR / "scrape_referees_report.txt"

DELAY_MIN = 2.0
DELAY_MAX = 4.0


# =============================================================================
# BAZA
# =============================================================================

def create_referee_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_referees (
            match_id    INTEGER PRIMARY KEY,
            sezon       TEXT,
            referee_raw TEXT,
            referee_name TEXT,
            referee_country TEXT,
            scraped_at  TEXT
        )
    """)
    conn.commit()


def get_already_scraped(conn):
    rows = conn.execute("SELECT match_id FROM match_referees").fetchall()
    return {r[0] for r in rows}


def get_matches_to_scrape(conn, season_filter=None):
    if season_filter:
        rows = conn.execute(
            "SELECT match_id, sezon, kolejka, gospodarz, gosc, flash_url, flash_id FROM matches WHERE sezon=? ORDER BY sezon, kolejka, match_id",
            (season_filter,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT match_id, sezon, kolejka, gospodarz, gosc, flash_url, flash_id FROM matches ORDER BY sezon, kolejka, match_id"
        ).fetchall()

    cols = ["match_id", "sezon", "kolejka", "gospodarz", "gosc", "flash_url", "flash_id"]
    return [dict(zip(cols, r)) for r in rows]


def save_referee(conn, match_id, sezon, referee_raw, referee_name, referee_country):
    conn.execute("""
        INSERT OR REPLACE INTO match_referees
            (match_id, sezon, referee_raw, referee_name, referee_country, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (match_id, sezon, referee_raw, referee_name, referee_country,
          datetime.now().isoformat()))
    conn.commit()


def save_no_referee(conn, match_id, sezon):
    conn.execute("""
        INSERT OR REPLACE INTO match_referees
            (match_id, sezon, referee_raw, referee_name, referee_country, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (match_id, sezon, "NOT_FOUND", None, None,
          datetime.now().isoformat()))
    conn.commit()


# =============================================================================
# PARSING
# =============================================================================

def parse_referee(text: str):
    """
    Szuka: SĘDZIA: Nazwisko I. (Pol)
    Zwraca: (referee_raw, referee_name, referee_country)
    """
    if not text:
        return None, None, None

    text_one = re.sub(r"\s+", " ", text).strip()

    # wzorzec główny
    patterns = [
        r"SĘDZIA:\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-\']+\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.)\s*\(([A-Za-z]+)\)",
        r"Sędzia:\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-\']+\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.)\s*\(([A-Za-z]+)\)",
        # bez kraju
        r"SĘDZIA:\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-\']+\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.)",
        r"Sędzia:\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-\']+\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.)",
    ]

    for pat in patterns:
        m = re.search(pat, text_one, flags=re.IGNORECASE)
        if m:
            groups = m.groups()
            referee_raw = m.group(0)
            referee_name = groups[0].strip()
            referee_country = groups[1].strip() if len(groups) > 1 else "Pol"
            return referee_raw, referee_name, referee_country

    return None, None, None


# =============================================================================
# PLAYWRIGHT HELPERS
# =============================================================================

_cookies_accepted = False


def accept_cookies(page):
    global _cookies_accepted
    if _cookies_accepted:
        return False

    candidates = [
        "button:has-text('Akceptuję')",
        "button:has-text('Akceptuj')",
        "button:has-text('Accept')",
        "#onetrust-accept-btn-handler",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=2000)
                time.sleep(1)
                _cookies_accepted = True
                return True
        except Exception:
            pass
    return False


def click_szczegoly_tab(page):
    selectors = [
        "a:has-text('Szczegóły')",
        "button:has-text('Szczegóły')",
        "a:has-text('Match details')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=3000)
                time.sleep(2)
                return True
        except Exception:
            pass
    return False


def scroll_to_bottom(page):
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)
        return True
    except Exception:
        return False


def get_body_text(page):
    try:
        return page.locator("body").inner_text(timeout=5000)
    except Exception:
        return ""


# =============================================================================
# MAIN SCRAPING LOOP
# =============================================================================

def scrape_all(season_filter=None, dry_run=False):
    global _cookies_accepted

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    create_referee_table(conn)

    already = get_already_scraped(conn)
    all_matches = get_matches_to_scrape(conn, season_filter)
    to_scrape = [m for m in all_matches if m["match_id"] not in already]

    print(f"Łącznie meczów: {len(all_matches)}")
    print(f"Już zescrapowanych: {len(already)}")
    print(f"Do zescrapowania: {len(to_scrape)}")

    if dry_run:
        print("\nDRY RUN — nie scrapujemy, tylko pokazujemy listę.")
        for m in to_scrape[:10]:
            print(f"  {m['sezon']} K{m['kolejka']:02d} | {m['gospodarz']} vs {m['gosc']} | {m['flash_id']}")
        conn.close()
        return

    stats = {
        "ok": 0,
        "not_found": 0,
        "error": 0,
        "skipped": len(already),
    }

    log_lines = []
    log_lines.append(f"SCRAPE START: {datetime.now().isoformat()}")
    log_lines.append(f"Season filter: {season_filter or 'ALL'}")
    log_lines.append(f"Total: {len(all_matches)} | Already: {len(already)} | To scrape: {len(to_scrape)}")
    log_lines.append("")

    _cookies_accepted = False

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(
            locale="pl-PL",
            viewport={"width": 1400, "height": 2400}
        )
        page = context.new_page()

        for i, m in enumerate(to_scrape, start=1):
            mid = m["match_id"]
            url = str(m["flash_url"]).strip()

            prefix = f"[{i:4d}/{len(to_scrape)}] {m['sezon']} K{m['kolejka']:02d} | {m['gospodarz']} vs {m['gosc']}"

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(2)

                accept_cookies(page)

                try:
                    page.wait_for_load_state("networkidle", timeout=6000)
                except Exception:
                    pass
                time.sleep(1.5)

                click_szczegoly_tab(page)
                scroll_to_bottom(page)

                body = get_body_text(page)
                referee_raw, referee_name, referee_country = parse_referee(body)

                if referee_name:
                    save_referee(conn, mid, m["sezon"], referee_raw, referee_name, referee_country)
                    stats["ok"] += 1
                    msg = f"OK  {referee_name} ({referee_country})"
                else:
                    save_no_referee(conn, mid, m["sezon"])
                    stats["not_found"] += 1
                    msg = "NOT_FOUND"

            except Exception as e:
                save_no_referee(conn, mid, m["sezon"])
                stats["error"] += 1
                msg = f"ERROR: {e}"

            print(f"{prefix} | {msg}")
            log_lines.append(f"{prefix} | {msg}")

            # zapisujemy raport co 50 meczów
            if i % 50 == 0:
                REPORT_TXT.write_text(
                    "\n".join(log_lines) + f"\n\nProgress: {i}/{len(to_scrape)}",
                    encoding="utf-8"
                )

            # rate limit
            delay = random.uniform(DELAY_MIN, DELAY_MAX)
            time.sleep(delay)

        browser.close()

    conn.close()

    # finalny raport
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append(f"SCRAPE END: {datetime.now().isoformat()}")
    log_lines.append(f"OK:        {stats['ok']}")
    log_lines.append(f"NOT_FOUND: {stats['not_found']}")
    log_lines.append(f"ERROR:     {stats['error']}")
    log_lines.append(f"SKIPPED:   {stats['skipped']}")
    log_lines.append(f"Coverage:  {stats['ok']}/{len(all_matches)} ({stats['ok']/max(len(all_matches),1):.1%})")

    report = "\n".join(log_lines)
    print("\n" + "=" * 60)
    print(report)
    REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\nZapisano raport: {REPORT_TXT}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scrape sędziów z Flashscore")
    parser.add_argument(
        "--season",
        help="Sezon do zescrapowania np. '2024/25'. Bez flagi = wszystkie.",
        default=None
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko pokaż co by było scrapowane, nie scrapuj."
    )
    args = parser.parse_args()

    scrape_all(season_filter=args.season, dry_run=args.dry_run)


if __name__ == "__main__":
    main()