import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

DEBUG_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"
LISTA_JSON = DEBUG_DIR / "zawodnicy_2025_26.json"

ZAKLADKI = ["Defensywa", "Podania", "Pozostałe"]

# Wybieramy 5 zawodników do testu: top, środek, koniec, bramkarz, mało minut
TEST_INDEKSY = [0, 50, 200, 400, 447]


async def scroll_do_dolu(page):
    for _ in range(10):
        await page.evaluate("window.scrollBy(0, 500)")
        await asyncio.sleep(0.3)
    await asyncio.sleep(1)


def parsuj_statystyki(html: str, zakladka: str) -> dict:
    """Wyciąga pary (nazwa: wartość) z jednej zakładki."""
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


async def scrapuj_zawodnika(page, url: str, sezon: str = "2025/2026") -> dict:
    """Scrapuje jednego zawodnika — wszystkie 4 zakładki."""
    t_start = time.time()
    
    await page.goto(url, wait_until="networkidle", timeout=60000)
    await asyncio.sleep(2)
    
    # Zmiana sezonu
    try:
        await page.get_by_text("2026/2027").first.click(timeout=5000)
        await asyncio.sleep(0.5)
        await page.get_by_text(sezon).first.click(timeout=5000)
        await asyncio.sleep(2)
    except Exception as e:
        return {"error": f"Zmiana sezonu: {e}", "url": url}
    
    # Nazwa zawodnika (z <h1> lub podobne)
    soup = BeautifulSoup(await page.content(), "html.parser")
    h1 = soup.find("h1")
    nazwa = h1.get_text(strip=True) if h1 else "?"
    
    wszystkie_staty = {"_url": url, "_nazwa": nazwa, "_sezon": sezon}
    
    # OFENSYWA (domyślna)
    await scroll_do_dolu(page)
    html = await page.content()
    wszystkie_staty.update(parsuj_statystyki(html, "OFENSYWA"))
    
    # Pozostałe zakładki
    for zakladka in ZAKLADKI:
        try:
            btn = page.locator("button").filter(has_text=zakladka).first
            await btn.click(timeout=5000)
            await asyncio.sleep(1)
            await scroll_do_dolu(page)
            html = await page.content()
            wszystkie_staty.update(parsuj_statystyki(html, zakladka))
        except Exception as e:
            wszystkie_staty[f"_err_{zakladka}"] = str(e)
    
    wszystkie_staty["_czas_s"] = round(time.time() - t_start, 1)
    return wszystkie_staty


async def main():
    zawodnicy = json.loads(LISTA_JSON.read_text(encoding="utf-8"))
    testowi = [zawodnicy[i] for i in TEST_INDEKSY]
    
    print(f"Testuję {len(testowi)} zawodników:")
    for z in testowi:
        print(f"  - {z['player_slug']} ({z['klub_slug']})")
    print()
    
    wyniki = []
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()
        
        for i, z in enumerate(testowi, 1):
            print(f"\n[{i}/{len(testowi)}] {z['player_slug']}")
            try:
                staty = await scrapuj_zawodnika(page, z["url"])
                wyniki.append(staty)
                
                if "error" in staty:
                    print(f"  ✗ BŁĄD: {staty['error']}")
                else:
                    ile = len([k for k in staty if not k.startswith("_")])
                    print(f"  ✓ {ile} statystyk, czas: {staty.get('_czas_s')}s")
                    # Pokaż kluczowe pola
                    for kluczowa in ["xG", "xA", "Minuty", "Podania"]:
                        if kluczowa in staty:
                            print(f"    {kluczowa}: {staty[kluczowa]}")
            except Exception as e:
                print(f"  ✗ CRASH: {e}")
                wyniki.append({"_url": z["url"], "error": str(e)})
        
        await browser.close()
    
    # Zapisz wyniki
    out_json = DEBUG_DIR / "test_5_zawodnikow.json"
    out_json.write_text(json.dumps(wyniki, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    udane = [w for w in wyniki if "error" not in w]
    print(f"Udane: {len(udane)}/{len(wyniki)}")
    if udane:
        avg_czas = sum(w.get("_czas_s", 0) for w in udane) / len(udane)
        print(f"Średni czas: {avg_czas:.1f}s/zawodnik")
        print(f"Szacowany czas dla 448 zawodników: {avg_czas * 448 / 60:.1f} min")
    print(f"\nSzczegóły w: {out_json}")


if __name__ == "__main__":
    asyncio.run(main())