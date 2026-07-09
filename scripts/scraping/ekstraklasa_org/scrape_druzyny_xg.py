import asyncio
import csv
import time
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "raw" / "ekstraklasa_org" / "druzyny"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEZONY = ["2023-2024", "2024-2025", "2025-2026"]
RANKINGI = ["druzynowe-xg", "druzynowe-xga"]
BASE = "https://ekstraklasa.org/statystyki/"


async def scroll_do_konca(page):
    prev = 0
    for _ in range(15):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.8)
        new = await page.evaluate("document.body.scrollHeight")
        if new == prev:
            return
        prev = new


def parsuj_druzyny(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    # Kontenery pełnej listy (min-h-20), tak jak w rankingach zawodników
    kontenery = soup.find_all("div", class_=lambda c: c and "grid" in c and "min-h-20" in c)
    
    wyniki = []
    widziane = set()
    for kont in kontenery:
        # Kluby: link /kluby/{slug}/ (bez /zawodnik/)
        link = kont.find("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" not in h)
        if not link:
            continue
        href = link["href"]
        parts = href.strip("/").split("/")
        klub_slug = parts[1] if len(parts) > 1 else ""
        if not klub_slug or klub_slug in widziane:
            continue
        widziane.add(klub_slug)
        
        spany = kont.find_all("span", recursive=False)
        pozycja = spany[0].get_text(strip=True) if spany else ""
        wartosc = spany[-1].get_text(strip=True) if len(spany) >= 2 else ""
        
        span_w_linku = link.find("span")
        nazwa = span_w_linku.get_text(strip=True) if span_w_linku else link.get_text(strip=True)
        
        wyniki.append({
            "pozycja": pozycja, "klub_slug": klub_slug,
            "nazwa": nazwa, "wartosc": wartosc,
        })
    return wyniki


async def scrapuj(browser, url: str, out_path: Path) -> tuple[int, float]:
    t_start = time.time()
    context = await browser.new_context(viewport={"width": 1400, "height": 900})
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        await scroll_do_konca(page)
        html = await page.content()
        wyniki = parsuj_druzyny(html)
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pozycja", "klub_slug", "nazwa", "wartosc"])
            w.writeheader()
            w.writerows(wyniki)
        return len(wyniki), round(time.time() - t_start, 1)
    finally:
        await context.close()


async def main():
    zadania = []
    for sezon in SEZONY:
        for ranking in RANKINGI:
            url = f"{BASE}?tab=team&season={sezon}&ranking={ranking}"
            out_path = OUT_DIR / f"{sezon}_{ranking}.csv"
            zadania.append((url, out_path, f"{sezon} / {ranking}"))
    
    print(f"Do zescrapowania: {len(zadania)} rankingów drużynowych")
    print(f"Wyniki: {OUT_DIR}\n")
    
    t_total = time.time()
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        for i, (url, out_path, opis) in enumerate(zadania, 1):
            print(f"[{i}/{len(zadania)}] {opis:<40}", end=" ", flush=True)
            try:
                count, czas = await scrapuj(browser, url, out_path)
                print(f"✓ {count} klubów ({czas}s)")
            except Exception as e:
                print(f"✗ BŁĄD: {e}")
            await asyncio.sleep(0.5)
        await browser.close()
    
    print(f"\nKoniec. Czas: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())