import argparse
import asyncio
import csv
import time
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
BASE = "https://ekstraklasa.org/statystyki/"

RANKINGI = [
    ("gole", ""), ("asysty", ""), ("kanadyjska", ""), ("strzaly", ""),
    ("kluczowe-podania", ""), ("odbiory", ""), ("xg", ""), ("xgot", ""),
    ("konwersja-strzalow", ""), ("gole-z-gry", ""), ("gole-z-pola-karnego", ""),
    ("gole-z-rzutow-karnych", ""), ("karne-wykonane", ""), ("celne-strzaly", ""),
    ("celnosc-strzalow", ""), ("dogodne-szanse", ""),
    ("pojedynki-obronne-wygrane", "defensive"), ("przechwyty", "defensive"),
    ("wybicia", "defensive"), ("zablokowane-strzaly", "defensive"),
    ("podania", "passing"), ("podania-celne", "passing"),
    ("podania-w-polowie-przeciwnika", "passing"),
    ("podania-celne-w-polowie-przeciwnika", "passing"),
    ("stworzone-dogodne-szanse", "passing"), ("xa", "passing"),
    ("dosrodkowania", "passing"), ("dosrodkowania-celne", "passing"),
    ("rzuty-rozne", "passing"),
    ("dryblingi", "other"), ("pojedynki-wygrane", "other"), ("minuty", "other"),
    ("spalone", "other"), ("faule", "other"), ("faule-wywalczone", "other"),
]

BRAMKARZE = [
    "czyste-konta", "obronione-rzuty-karne", "obronione-strzaly",
    "gk-piastkowania", "bramki-stracone", "gk-podania",
    "gk-mecze-rozegrane", "gk-minuty-rozegrane", "gk-rzuty-karne-przeciwko",
    "gk-podania-celne", "gk-dlugie-podania-celne",
]


def buduj_url(sezon: str, ranking: str, category: str = "", typ_bramkarze: bool = False) -> str:
    params = [f"season={sezon}"]
    if typ_bramkarze:
        params.append("typ=bramkarze")
    elif category:
        params.append(f"category={category}")
    params.append(f"ranking={ranking}")
    return BASE + "?" + "&".join(params)


async def scroll_do_konca(page):
    prev = 0
    for _ in range(30):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.2)
        new = await page.evaluate("document.body.scrollHeight")
        if new == prev:
            return
        prev = new


async def klik_pokaz_wiecej(page):
    btns = await page.locator("button").all()
    for btn in btns:
        try:
            txt = (await btn.inner_text()).strip().lower()
            cls = await btn.get_attribute("class") or ""
            if "pokaż" in txt and "cky" not in cls:
                await btn.click(timeout=5000)
                await asyncio.sleep(2)
                return True
        except Exception:
            continue
    return False


def parsuj_ranking(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    kontenery = soup.find_all("div", class_=lambda c: c and "grid" in c and "min-h-20" in c)
    
    wyniki = []
    widziane = set()
    for kont in kontenery:
        link = kont.find("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" in h)
        if not link:
            continue
        href = link["href"]
        parts = href.strip("/").split("/")
        klub_slug = parts[1] if len(parts) > 1 else ""
        player_slug = parts[3] if len(parts) > 3 else ""
        if player_slug in widziane:
            continue
        widziane.add(player_slug)
        
        spany = kont.find_all("span", recursive=False)
        pozycja = spany[0].get_text(strip=True) if spany else ""
        wartosc = spany[-1].get_text(strip=True) if len(spany) >= 2 else ""
        
        span_w_linku = link.find("span")
        nazwa = span_w_linku.get_text(strip=True) if span_w_linku else link.get_text(strip=True)
        
        wyniki.append({
            "pozycja": pozycja, "player_slug": player_slug,
            "klub_slug": klub_slug, "nazwa": nazwa, "wartosc": wartosc,
        })
    return wyniki


async def scrapuj_ranking(browser, url: str, out_path: Path) -> tuple[int, float]:
    t_start = time.time()
    context = await browser.new_context(viewport={"width": 1400, "height": 900})
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        await klik_pokaz_wiecej(page)
        await scroll_do_konca(page)
        html = await page.content()
        wyniki = parsuj_ranking(html)
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pozycja", "player_slug", "klub_slug", "nazwa", "wartosc"])
            w.writeheader()
            w.writerows(wyniki)
        return len(wyniki), round(time.time() - t_start, 1)
    finally:
        await context.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, help='np. "2025-2026"')
    args = parser.parse_args()
    
    sezon = args.season
    out_dir = ROOT / "data" / "raw" / "ekstraklasa_org" / sezon
    out_dir.mkdir(parents=True, exist_ok=True)
    
    zadania = []
    for ranking, category in RANKINGI:
        url = buduj_url(sezon, ranking, category=category)
        zadania.append((url, out_dir / f"pole_{ranking}.csv", ranking))
    for ranking in BRAMKARZE:
        url = buduj_url(sezon, ranking, typ_bramkarze=True)
        zadania.append((url, out_dir / f"gk_{ranking}.csv", ranking))
    
    print(f"Sezon: {sezon}")
    print(f"Do zescrapowania: {len(zadania)} rankingów")
    print(f"Wyniki: {out_dir}\n")
    
    t_total = time.time()
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        for i, (url, out_path, ranking) in enumerate(zadania, 1):
            print(f"[{i}/{len(zadania)}] {ranking:<40}", end=" ", flush=True)
            try:
                count, czas = await scrapuj_ranking(browser, url, out_path)
                print(f"✓ {count} rekordów ({czas}s)")
            except Exception as e:
                print(f"✗ BŁĄD: {e}")
            await asyncio.sleep(0.5)
        await browser.close()
    
    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"KONIEC {sezon}. Czas: {total_min:.1f} min")
    print(f"Pliki: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())