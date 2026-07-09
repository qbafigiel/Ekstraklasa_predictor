import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Slugi z JSON'a Bobcka - hipoteza
SLUGI_TEST = [
    "druzynowe-gole",
    "druzynowe-strzaly",
    "druzynowe-xg",
    "druzynowe-xga",
    "druzynowe-celnosc-podan",
    "druzynowe-posiadanie",
]

SEZON = "2025-2026"
BASE = "https://ekstraklasa.org/statystyki/"


async def sprawdz_slug(page, slug: str) -> tuple[bool, int, str]:
    """Otwiera URL i sprawdza czy jest tabela z drużynami."""
    url = f"{BASE}?season={SEZON}&ranking={slug}"
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(3)
    
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    
    # Kluby: linki /kluby/{slug}/ (bez /zawodnik/)
    linki_klubow = soup.find_all("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" not in h)
    unikalne_kluby = set()
    for a in linki_klubow:
        parts = a["href"].strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "kluby":
            unikalne_kluby.add(parts[1])
    
    # Sprawdź czy jest tekst "brak" gdzieś
    txt = soup.get_text().lower()
    brak_danych = "brak" in txt and "danych" in txt
    
    return len(unikalne_kluby) >= 10, len(unikalne_kluby), "BRAK DANYCH" if brak_danych else "OK"


async def main():
    print(f"Testuję {len(SLUGI_TEST)} slugów dla sezonu {SEZON}\n")
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()
        
        for slug in SLUGI_TEST:
            print(f"  {slug:<35}", end=" ", flush=True)
            try:
                ok, ile_klubow, status = await sprawdz_slug(page, slug)
                znak = "✓" if ok else "✗"
                print(f"{znak} klubów: {ile_klubow:<3} {status}")
            except Exception as e:
                print(f"✗ BŁĄD: {e}")
            await asyncio.sleep(1)
        
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())