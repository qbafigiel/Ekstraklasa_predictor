import sqlite3
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

DB_PATH = Path("db/ekstraklasa.db")
REPORT_DIR = Path("data/reports/model")
REPORT_TXT = REPORT_DIR / "probe_flashscore_referee.txt"
REPORT_HTML_DIR = REPORT_DIR / "probe_flashscore_referee_html"

SAMPLE_QUERY = """
SELECT match_id, sezon, kolejka, gospodarz, gosc, flash_url, flash_id
FROM matches
WHERE sezon IN ('2024/25', '2025/26')
ORDER BY sezon, kolejka, match_id
LIMIT 5
"""


def load_sample_matches():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(SAMPLE_QUERY).fetchall()
    conn.close()

    cols = ["match_id", "sezon", "kolejka", "gospodarz", "gosc", "flash_url", "flash_id"]
    return [dict(zip(cols, row)) for row in rows]


def normalize_url(url: str) -> str:
    return str(url).strip()


def accept_cookies(page):
    candidates = [
        "button:has-text('Akceptuję')",
        "button:has-text('Akceptuj')",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "#onetrust-accept-btn-handler",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=1500)
                time.sleep(1)
                return True
        except Exception:
            pass
    return False


def click_match_details_tab(page):
    """
    Przełącza zakładkę na 'Szczegóły meczu' / podobną.
    Jeśli się nie uda, zostawia stronę jak jest.
    """
    selectors = [
        "a:has-text('Szczegóły meczu')",
        "button:has-text('Szczegóły meczu')",
        "a:has-text('Szczegóły')",
        "button:has-text('Szczegóły')",
        "a:has-text('Match details')",
        "button:has-text('Match details')",
    ]

    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(timeout=3000)
                time.sleep(2)
                return f"clicked: {sel}"
        except Exception:
            pass

    return "tab_not_found"


def extract_referee_from_text(text: str):
    """
    Szuka:
      Sędzia: Nazwisko I.
    i zwraca dopasowanie + kontekst.
    """
    if not text:
        return None, None

    text_one = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"Sędzia:\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-\']+(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.)?)",
        r"Sedzia:\s*([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\-\']+(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.)?)",
        r"Referee:\s*([A-Z][A-Za-z\-\']+(?:\s+[A-Z]\.)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text_one, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start() - 180)
            end = min(len(text_one), m.end() + 220)
            snippet = text_one[start:end]
            return m.group(1).strip(), snippet

    return None, None


def extract_referee_from_html(html: str):
    if not html:
        return None, None

    html_one = re.sub(r"\s+", " ", html).strip()

    patterns = [
        r"Sędzia:\s*([^<]{2,80})",
        r"Sedzia:\s*([^<]{2,80})",
        r"Referee:\s*([^<]{2,80})",
    ]

    for pat in patterns:
        m = re.search(pat, html_one, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            candidate = re.split(r"(Stadion:|Pojemność:|Frekwencja:|Venue:|Attendance:)", candidate)[0].strip()
            start = max(0, m.start() - 180)
            end = min(len(html_one), m.end() + 220)
            snippet = html_one[start:end]
            return candidate, snippet

    return None, None


def scroll_to_bottom(page):
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        return True
    except Exception:
        return False


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_HTML_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_sample_matches()
    if not matches:
        raise RuntimeError("Brak meczów do próby.")

    lines = []
    lines.append("=" * 100)
    lines.append("PROBE FLASHSCORE — SĘDZIA Z ZAKŁADKI 'SZCZEGÓŁY MECZU'")
    lines.append("=" * 100)
    lines.append("")

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(locale="pl-PL", viewport={"width": 1400, "height": 2400})
        page = context.new_page()

        for i, m in enumerate(matches, start=1):
            url = normalize_url(m["flash_url"])

            lines.append("-" * 100)
            lines.append(
                f"[{i}] {m['sezon']} K{m['kolejka']:02d} | "
                f"{m['gospodarz']} vs {m['gosc']} | match_id={m['match_id']} | flash_id={m['flash_id']}"
            )
            lines.append(f"URL: {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(3)

                cookies_clicked = accept_cookies(page)
                lines.append(f"Cookies clicked: {cookies_clicked}")

                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
                time.sleep(2)

                tab_info = click_match_details_tab(page)
                lines.append(f"Tab switch: {tab_info}")

                scroll_ok = scroll_to_bottom(page)
                lines.append(f"Scrolled to bottom: {scroll_ok}")

                time.sleep(2)

                body_text = ""
                try:
                    body_text = page.locator("body").inner_text(timeout=5000)
                except Exception:
                    pass

                html = page.content()

                html_path = REPORT_HTML_DIR / f"{i}_{m['sezon'].replace('/', '_')}_{m['match_id']}.html"
                png_path = REPORT_HTML_DIR / f"{i}_{m['sezon'].replace('/', '_')}_{m['match_id']}.png"

                html_path.write_text(html, encoding="utf-8")
                try:
                    page.screenshot(path=str(png_path), full_page=True)
                except Exception as e:
                    lines.append(f"Screenshot error: {e}")

                lines.append(f"Body text len: {len(body_text)}")
                lines.append(f"HTML saved: {html_path}")
                lines.append(f"PNG saved:  {png_path}")

                ref_text, snippet_text = extract_referee_from_text(body_text)
                ref_html, snippet_html = extract_referee_from_html(html)

                lines.append(f"Referee from text: {ref_text}")
                if snippet_text:
                    lines.append(f"  snippet_text: {snippet_text}")

                lines.append(f"Referee from html: {ref_html}")
                if snippet_html:
                    lines.append(f"  snippet_html: {snippet_html}")

                # dodatkowo szukamy surowo keywordów
                text_one = re.sub(r"\s+", " ", body_text).strip()
                keyword_hits = []
                for kw in ["Sędzia", "Sedzia", "Referee", "Stadion", "Pojemność", "Frekwencja"]:
                    for hit in re.finditer(re.escape(kw), text_one, flags=re.IGNORECASE):
                        start = max(0, hit.start() - 120)
                        end = min(len(text_one), hit.end() + 180)
                        keyword_hits.append((kw, text_one[start:end]))

                if keyword_hits:
                    lines.append(f"Keyword hits in text: {len(keyword_hits)}")
                    for kw, snip in keyword_hits[:10]:
                        lines.append(f"  [{kw}] {snip}")
                else:
                    lines.append("Keyword hits in text: 0")

            except Exception as e:
                lines.append(f"BŁĄD: {e}")

            lines.append("")

        browser.close()

    report_text = "\n".join(lines)
    print(report_text)
    REPORT_TXT.write_text(report_text, encoding="utf-8")
    print(f"\nZapisano: {REPORT_TXT}")


if __name__ == "__main__":
    main()