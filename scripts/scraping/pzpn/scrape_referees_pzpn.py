"""
scrape_referees_pzpn.py
=======================
Scraper obsad sedziowskich Ekstraklasy z PZPN.

Iteruje po podstronach listy artykulow, filtruje po sezonie,
pobiera PDF z obsada i parsuje tabele (mecz, sedzia glowny).

Zapisuje do fixtures_upcoming.referee_full_name.

Uzycie:
    python scripts/scraping/pzpn/scrape_referees_pzpn.py --sezon 2026/27
    python scripts/scraping/pzpn/scrape_referees_pzpn.py --sezon 2025/26 --dry-run
    python scripts/scraping/pzpn/scrape_referees_pzpn.py --sezon 2026/27 --max-pages 5
"""

import argparse
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "db" / "ekstraklasa.db"
REPORT_PATH = ROOT / "data" / "reports" / "model" / "scrape_referees_pzpn_report.txt"

BASE_URL = "https://pzpn.pl/federacja/sedziowie/obsada-sedziowska/ekstraklasa"
POLISH_L_MAP = str.maketrans({"ł": "l", "Ł": "L"})

# Aliasy nazw klubow z PDF PZPN -> nazwy w naszej bazie
CLUB_ALIASES_PZPN = {
    "rks rakow czestochowa": "Raków Częstochowa",
    "rks rakow czestochowa s.a.": "Raków Częstochowa",
    "rakow czestochowa": "Raków Częstochowa",
    "kks lech poznan": "Lech Poznań",
    "lech poznan": "Lech Poznań",
    "gornik zabrze": "Górnik Zabrze",
    "gornik zabrze s.a.": "Górnik Zabrze",
    "gks piast gliwice": "Piast Gliwice",
    "gks piast gliwice s.a.": "Piast Gliwice",
    "piast gliwice": "Piast Gliwice",
    "gks gieksa katowice": "GKS Katowice",
    "gks gieksa katowice s.a.": "GKS Katowice",
    "gks katowice": "GKS Katowice",
    "legia warszawa": "Legia Warszawa",
    "legia warszawa s.a.": "Legia Warszawa",
    "arka gdynia": "Arka Gdynia",
    "arka gdynia sa": "Arka Gdynia",
    "arka gdynia s.a.": "Arka Gdynia",
    "wisla plock": "Wisła Płock",
    "wisla plock s.a.": "Wisła Płock",
    "wisla krakow": "Wisła Kraków",
    "wieczysta krakow": "Wieczysta Kraków",
    "bruk-bet termalica nieciecza": "Bruk-Bet Termalica Nieciecza",
    "bruk-bet termalica": "Bruk-Bet Termalica Nieciecza",
    "lechia gdansk": "Lechia Gdańsk",
    "jagiellonia bialystok": "Jagiellonia Białystok",
    "jagiellonia bialystok ssa": "Jagiellonia Białystok",
    "zaglebie lubin": "Zagłębie Lubin",
    "radomiak radom": "Radomiak Radom",
    "radomiak s.a.": "Radomiak Radom",
    "radomiak sa": "Radomiak Radom",
    "motor lublin": "Motor Lublin",
    "motor lublin s.a.": "Motor Lublin",
    "pogon szczecin": "Pogoń Szczecin",
    "widzew lodz": "Widzew Łódź",
    "widzew lodz sa": "Widzew Łódź",
    "widzew lodz s.a.": "Widzew Łódź",
    "cracovia": "Cracovia",
    "ks cracovia sa": "Cracovia",
    "ks cracovia": "Cracovia",
    "korona sa kielce": "Korona Kielce",
    "korona kielce": "Korona Kielce",
    "slask wroclaw": "Śląsk Wrocław",
    "puszcza niepolomice": "Puszcza Niepołomice",
    "stal mielec": "Stal Mielec",
    "ruch chorzow": "Ruch Chorzów",
    "warta poznan": "Warta Poznań",
    "lks lodz": "ŁKS Łódź",
}

def normalize_ascii(text: str) -> str:
    text = str(text).strip().lower()
    text = text.translate(POLISH_L_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())

def resolve_club(raw: str) -> str | None:
    """Mapuje nazwe klubu z PDF na nasza nazwe."""
    if not raw:
        return None
    # PDF czasem ma \n w srodku nazwy (np. 'RKS RAKÓW\nCZĘSTOCHOWA S.A.')
    cleaned = raw.replace("\n", " ").replace("\r", " ")
    norm = normalize_ascii(cleaned)
    if norm in CLUB_ALIASES_PZPN:
        return CLUB_ALIASES_PZPN[norm]
    # Fallback: usun sufiksy S.A., SA, SSA
    cleaned2 = re.sub(r"\s+(s\.?a\.?|ssa|sa)$", "", norm)
    if cleaned2 in CLUB_ALIASES_PZPN:
        return CLUB_ALIASES_PZPN[cleaned2]
    return None


def matches_target_season(article_title_or_url: str, target_sezon: str) -> bool:
    text_norm = normalize_ascii(article_title_or_url)
    left, right = target_sezon.split("/")
    right_full = right if len(right) == 4 else str(int(left[:2] + right))
    patterns = [
        f"{left}-{right_full}",
        f"{left}/{right_full}",
        f"{left} {right_full}",
        f"{left}_{right_full}",
    ]
    return any(p in text_norm for p in patterns)


def extract_kolejka_from_title(title: str) -> int | None:
    m = re.search(r"(\d{1,2})\.\s*kolejki", title.lower())
    if m:
        return int(m.group(1))
    return None


def fetch_html(url: str, playwright_ctx) -> str:
    page = playwright_ctx.new_page()
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)
        return page.content()
    finally:
        page.close()


def collect_article_links(playwright_ctx, target_sezon: str, max_pages: int = 30) -> list[dict]:
    articles = []
    seen_urls = set()
    empty_pages_in_row = 0

    for page_num in range(1, max_pages + 1):
        url = f"{BASE_URL}?p={page_num}"
        print(f"  Strona {page_num}: {url}")

        html = fetch_html(url, playwright_ctx)
        soup = BeautifulSoup(html, "html.parser")

        found_on_page = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "obsada" not in href.lower():
                continue
            if "kolejki" not in href.lower():
                continue

            full_url = href if href.startswith("http") else urljoin(BASE_URL, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            if not matches_target_season(full_url, target_sezon):
                continue

            articles.append({"url": full_url, "page_num": page_num})
            found_on_page += 1

        print(f"    Znaleziono {found_on_page} nowych artykulow sezonu {target_sezon}")

        if found_on_page == 0:
            empty_pages_in_row += 1
        else:
            empty_pages_in_row = 0

        if empty_pages_in_row >= 3:
            print(f"    3 puste strony z rzedu - przerywam.")
            break

        time.sleep(0.5)

    return articles


def enrich_article(article_url: str, playwright_ctx) -> dict | None:
    html = fetch_html(article_url, playwright_ctx)
    soup = BeautifulSoup(html, "html.parser")

    title = None
    for tag_name in ["h1", "h2"]:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(strip=True)
            if text and "obsada" in text.lower() and "kolejki" in text.lower():
                title = text
                break
        if title:
            break

    if not title:
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", "")

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    kolejka = extract_kolejka_from_title(title or "")

    pdf_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" not in href.lower():
            continue
        text = a.get_text(strip=True).lower()
        if text == "pobierz" or "komunikat" in href.lower() or "system/files/articles" in href.lower():
            pdf_url = href if href.startswith("http") else urljoin(article_url, href)
            break

    is_change = "zmiana" in (title or "").lower() or "zmiana" in article_url.lower()
    is_postponed = "przelozon" in (title or "").lower() or "przelozon" in article_url.lower()

    return {
        "url": article_url,
        "title": title or "",
        "kolejka": kolejka,
        "pdf_url": pdf_url,
        "is_change": is_change,
        "is_postponed": is_postponed,
    }


def parse_referee_pdf(pdf_bytes: bytes) -> tuple[list[dict], list[dict]]:
    """
    Parsuje PDF obsady wykorzystujac extract_tables().

    Zwraca (fixtures, unknown_clubs):
      fixtures: {gospodarz, gosc, data, godzina, referee_full_name, referee_raw}
      unknown_clubs: [{raw_home, raw_away, resolved_home, resolved_away}]
    """
    fixtures = []
    unknown = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if not row or len(row) < 5:
                        continue

                    # Wzorzec: [gospodarz, gosc, data, godzina, sedzia, ...]
                    home_raw = row[0]
                    away_raw = row[1]
                    data_val = row[2]
                    godzina_val = row[3]
                    referee_raw = row[4]

                    if not all([home_raw, away_raw, data_val, godzina_val, referee_raw]):
                        continue

                    # Filtr: data musi pasowac do YYYY-MM-DD
                    if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(data_val).strip()):
                        continue

                    # Godzina musi pasowac do HH:MM
                    if not re.match(r"^\d{1,2}:\d{2}$", str(godzina_val).strip()):
                        continue

                    home = resolve_club(home_raw)
                    away = resolve_club(away_raw)

                    if not home or not away:
                        unknown.append({
                            "raw_home": home_raw,
                            "raw_away": away_raw,
                            "resolved_home": home,
                            "resolved_away": away,
                        })
                        continue

                    # Sedzia glowny: format "Imie Nazwisko - Miasto\n" lub "Imie Nazwisko - Miasto"
                    ref_cleaned = str(referee_raw).replace("\n", " ").strip()
                    # Usun " - Miasto" na koncu
                    ref_main = re.split(r"\s*-\s*", ref_cleaned)[0].strip()

                    if not ref_main:
                        continue

                    fixtures.append({
                        "gospodarz": home,
                        "gosc": away,
                        "data": str(data_val).strip(),
                        "godzina": str(godzina_val).strip(),
                        "referee_full_name": ref_main,
                        "referee_raw": ref_cleaned,
                    })

    return fixtures, unknown


def load_fixtures_map(conn: sqlite3.Connection, sezon: str) -> dict:
    rows = conn.execute(
        "SELECT fixture_id, gospodarz, gosc, kolejka FROM fixtures_upcoming WHERE sezon = ?",
        (sezon,)
    ).fetchall()
    return {(g, a, k): fid for fid, g, a, k in rows}


def update_referee(conn: sqlite3.Connection, fixture_id: str, referee: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE fixtures_upcoming SET referee_full_name = ?, updated_at = ? WHERE fixture_id = ?",
        (referee, now, fixture_id)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True, help="np. 2026/27")
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Sezon:      {args.sezon}")
    print(f"Max pages:  {args.max_pages}")
    print(f"Dry run:    {args.dry_run}")
    print()

    if not DB_PATH.exists():
        print(f"Blad: baza {DB_PATH} nie istnieje.")
        sys.exit(1)

    stats = {
        "articles_found": 0,
        "articles_with_pdf": 0,
        "fixtures_parsed": 0,
        "fixtures_matched": 0,
        "fixtures_updated": 0,
        "unknown_clubs": [],
        "unmatched_fixtures": [],
    }

    conn = sqlite3.connect(DB_PATH)
    fixtures_map = load_fixtures_map(conn, args.sezon)
    print(f"Fixtures w bazie dla sezonu {args.sezon}: {len(fixtures_map)}")

    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/122.0"
            )

            print("\n[1] Zbieram linki artykulow obsad...")
            articles = collect_article_links(context, args.sezon, max_pages=args.max_pages)
            stats["articles_found"] = len(articles)
            print(f"    Znalazlem {len(articles)} artykulow sezonu {args.sezon}")

            if not articles:
                print("Brak artykulow.")
                return

            print(f"\n[2] Przetwarzam artykuly...")

            for i, article in enumerate(articles, 1):
                print(f"\n  [{i}/{len(articles)}] {article['url']}")

                enriched = enrich_article(article["url"], context)
                if not enriched or not enriched["pdf_url"]:
                    print(f"    BRAK PDF - pomijam")
                    continue

                if not enriched["kolejka"]:
                    print(f"    BRAK KOLEJKI - pomijam. Tytul: {enriched['title']}")
                    continue

                stats["articles_with_pdf"] += 1
                article_type = "ZMIANA" if enriched["is_change"] else ("PRZELOZONE" if enriched["is_postponed"] else "ZWYKLA")
                print(f"    Tytul: {enriched['title'][:80]}")
                print(f"    Kolejka: {enriched['kolejka']} | Typ: {article_type}")

                try:
                    r = requests.get(enriched["pdf_url"], timeout=30)
                    r.raise_for_status()
                except Exception as e:
                    print(f"    BLAD pobierania PDF: {e}")
                    continue

                try:
                    fixtures_pdf, unknown = parse_referee_pdf(r.content)
                except Exception as e:
                    print(f"    BLAD parsowania PDF: {e}")
                    continue

                stats["unknown_clubs"].extend(unknown)
                stats["fixtures_parsed"] += len(fixtures_pdf)
                print(f"    Sparsowanych meczy: {len(fixtures_pdf)}")

                for fpdf in fixtures_pdf:
                    key = (fpdf["gospodarz"], fpdf["gosc"], enriched["kolejka"])
                    fixture_id = fixtures_map.get(key)

                    if not fixture_id:
                        stats["unmatched_fixtures"].append({
                            "sezon": args.sezon,
                            "kolejka": enriched["kolejka"],
                            "gospodarz": fpdf["gospodarz"],
                            "gosc": fpdf["gosc"],
                            "referee": fpdf["referee_full_name"],
                        })
                        continue

                    stats["fixtures_matched"] += 1

                    if not args.dry_run:
                        update_referee(conn, fixture_id, fpdf["referee_full_name"])
                        stats["fixtures_updated"] += 1

                    print(f"      MATCH: K{enriched['kolejka']} {fpdf['gospodarz']} vs {fpdf['gosc']} -> {fpdf['referee_full_name']}")

                if not args.dry_run:
                    conn.commit()

                time.sleep(0.5)

            browser.close()

    finally:
        conn.close()

    # Raport
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("SCRAPE REFEREES PZPN REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {now}")
    lines.append(f"Sezon:     {args.sezon}")
    lines.append(f"Dry run:   {args.dry_run}")
    lines.append("")
    lines.append(f"Artykulow znalezionych:     {stats['articles_found']}")
    lines.append(f"Artykulow z PDF:             {stats['articles_with_pdf']}")
    lines.append(f"Fixtures sparsowanych:       {stats['fixtures_parsed']}")
    lines.append(f"Fixtures zmatchowanych:      {stats['fixtures_matched']}")
    lines.append(f"Fixtures zaktualizowanych:   {stats['fixtures_updated']}")
    lines.append(f"Nieznanych klubow (skips):   {len(stats['unknown_clubs'])}")
    lines.append("")

    if stats["unknown_clubs"]:
        lines.append("NIEZNANE KLUBY (pierwsze 20):")
        for u in stats["unknown_clubs"][:20]:
            lines.append(f"  home='{u['raw_home']}' away='{u['raw_away']}' -> ({u['resolved_home']}, {u['resolved_away']})")
        lines.append("")

    if stats["unmatched_fixtures"]:
        lines.append(f"NIEZMATCHOWANE Z FIXTURES_UPCOMING ({len(stats['unmatched_fixtures'])}) - pierwsze 30:")
        for u in stats["unmatched_fixtures"][:30]:
            lines.append(f"  K{u['kolejka']:02d} {u['gospodarz']} vs {u['gosc']} -> {u['referee']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nRaport: {REPORT_PATH}")


if __name__ == "__main__":
    main()