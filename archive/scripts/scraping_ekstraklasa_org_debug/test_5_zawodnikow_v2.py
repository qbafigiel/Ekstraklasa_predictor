import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

DEBUG_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"
LISTA_JSON = DEBUG_DIR / "zawodnicy_2025_26.json"

ZAKLADKI = ["Defensywa", "Podania", "Pozostałe"]
TEST_INDEKSY = [0, 50, 200, 400, 447]


async def scroll_do_dolu(page):
    for _ in range(8):
        await page.evaluate("window.scrollBy(0, 500)")
        await asyncio.sleep(0.3)
    await asyncio.sleep(1)


def parsuj_statystyki(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    labels = soup.find_all(
        "p", 
        class_=lambda c: c and "label-medium-bold" in c and "uppercase" in c
    )
    wyniki = {}
    for label in labels:
        parent = label.parent
        span = parent.find("span", class_=lambda c: c and "heading-small" in c) if parent else None
        if span:
            nazwa = label.get_text(strip=True)
            wartosc = span.get_text(strip=True)
            wyniki[nazwa] = wartosc
    return wyniki


async def scrapuj_zawodnika(browser, url: str, sezon: str = "2025/2026") -> dict:
    """Każdy zawodnik = NOWY CONTEXT (crash-safe, świeży stan)."""
    t_start = time.time()
    context = await browser.new_context(viewport={"width": 1400, "height": 900})
    page = await context.new_page()
    
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        
        # Zmiana sezonu z retry
        zmieniono_sezon = False
        for proba in range(3):
            try:
                await page.get_by_text("2026/2027").first.click(timeout=8000)
                await asyncio.sleep(1)
                await page.get_by_text(sezon).first.click(timeout=8000)
                await asyncio.sleep(3)
                zmieniono_sezon = True
                break
            except Exception as e:
                print(f"    Próba {proba+1}/3 zmiany sezonu nie udana: {e}")
                await asyncio.sleep(2)
        
        if not zmieniono_sezon:
            return {"_url": url, "error": "Nie udało się zmienić sezonu (3 próby)"}
        
        # Nazwa zawodnika
        soup = BeautifulSoup(await page.content(), "html.parser")
        h1 = soup.find("h1")
        nazwa = h1.get_text(strip=True) if h1 else "?"
        
        wszystkie_staty = {"_url": url, "_nazwa": nazwa, "_sezon": sezon}
        
        # OFENSYWA (domyślna)
        await scroll_do_dolu(page)
        html = await page.content()
        wszystkie_staty.update(parsuj_statystyki(html))
        
        # Pozostałe zakładki z retry
        for zakladka in ZAKLADKI:
            klikniete = False
            for proba in range(3):
                try:
                    btn = page.locator("button").filter(has_text=zakladka).first
                    await btn.click(timeout=8000)
                    await asyncio.sleep(1.5)
                    await scroll_do_dolu(page)
                    html = await page.content()
                    nowe_staty = parsuj_statystyki(html)
                    wszystkie_staty.update(nowe_staty)
                    klikniete = True
                    break
                except Exception as e:
                    print(f"    Próba {proba+1}/3 {zakladka}: {e}")
                    await asyncio.sleep(1)
            if not klikniete:
                wszystkie_staty[f"_err_{zakladka}"] = "Nie kliknięto po 3 próbach"
        
        wszystkie_staty["_czas_s"] = round(time.time() - t_start, 1)
        return wszystkie_staty
    
    finally:
        await context.close()


async def main():
    zawodnicy = json.loads(LISTA_JSON.read_text(encoding="utf-8"))
    testowi = [zawodnicy[i] for i in TEST_INDEKSY]
    
    print(f"Testuję {len(testowi)} zawodników (NOWY CONTEXT per zawodnik):")
    for z in testowi:
        print(f"  - {z['player_slug']} ({z['klub_slug']})")
    print()
    
    wyniki = []
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        
        for i, z in enumerate(testowi, 1):
            print(f"\n[{i}/{len(testowi)}] {z['player_slug']}")
            try:
                staty = await scrapuj_zawodnika(browser, z["url"])
                wyniki.append(staty)
                
                if "error" in staty:
                    print(f"  ✗ BŁĄD: {staty['error']}")
                else:
                    ile = len([k for k in staty if not k.startswith("_")])
                    errs = [k for k in staty if k.startswith("_err_")]
                    print(f"  ✓ {ile} statystyk, czas: {staty.get('_czas_s')}s")
                    if errs:
                        print(f"    BŁĘDY zakładek: {errs}")
                    for kluczowa in ["xG", "xA", "Minuty"]:
                        if kluczowa in staty:
                            print(f"    {kluczowa}: {staty[kluczowa]}")
            except Exception as e:
                print(f"  ✗ CRASH: {e}")
                wyniki.append({"_url": z["url"], "error": str(e)})
            
            # Krótka przerwa między zawodnikami (anti-block)
            await asyncio.sleep(1)
        
        await browser.close()
    
    out_json = DEBUG_DIR / "test_5_zawodnikow_v2.json"
    out_json.write_text(json.dumps(wyniki, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    udane = [w for w in wyniki if "error" not in w]
    kompletne = [w for w in udane if not any(k.startswith("_err_") for k in w)]
    print(f"Udane (jakikolwiek wynik): {len(udane)}/{len(wyniki)}")
    print(f"Kompletne (wszystkie 4 zakładki): {len(kompletne)}/{len(wyniki)}")
    if udane:
        avg = sum(w.get("_czas_s", 0) for w in udane) / len(udane)
        print(f"Średni czas: {avg:.1f}s/zawodnik")
        print(f"Szacowany czas dla 448 zawodników: {avg * 448 / 60:.1f} min")
    print(f"\nSzczegóły: {out_json}")


if __name__ == "__main__":
    asyncio.run(main())