import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

URL = "https://ekstraklasa.org/statystyki/?season=2025-2026&ranking=xg"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def scroll_do_konca(page):
    prev_height = 0
    for i in range(30):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.5)
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == prev_height:
            return i + 1
        prev_height = new_height
    return 30


async def klik_pokaz_wiecej(page):
    """Szuka przycisku 'Pokaż więcej' (nie cookies) i klika."""
    btns = await page.locator("button").all()
    for btn in btns:
        try:
            txt = (await btn.inner_text()).strip().lower()
            cls = await btn.get_attribute("class") or ""
            if "pokaż" in txt and "cky" not in cls:
                print(f"  Klikam: '{txt}'")
                await btn.click(timeout=5000)
                await asyncio.sleep(2)
                return True
        except Exception:
            continue
    return False


async def main():
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        print(f"Otwieram: {URL}")
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)

        print("Szukam przycisku 'Pokaż więcej'...")
        klik = await klik_pokaz_wiecej(page)
        if not klik:
            print("  Nie znaleziono odpowiedniego przycisku")

        print("Scrolluję do końca...")
        scrolli = await scroll_do_konca(page)
        print(f"  Wykonano {scrolli} scrolli")

        html = await page.content()
        (OUT_DIR / "ranking_xg_2025_26.html").write_text(html, encoding="utf-8")
        print(f"✓ Zapisano HTML ({len(html)} znaków)")

        soup = BeautifulSoup(html, "html.parser")
        linki = soup.find_all("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" in h)
        # deduplikacja
        widziane = set()
        unikalne = []
        for a in linki:
            if a["href"] not in widziane:
                widziane.add(a["href"])
                unikalne.append(a)
        print(f"\n✓ Znaleziono {len(unikalne)} unikalnych zawodników")

        # Test parsera wartości — pierwsze 10
        print(f"\nPRÓBKA (pierwsze 10):")
        for a in unikalne[:10]:
            href = a["href"]
            parts = href.strip("/").split("/")
            player_slug = parts[3] if len(parts) > 3 else "?"

            # Idź w górę drzewa aż znajdziesz liczbę
            wartosc = None
            elem = a
            for _ in range(6):
                elem = elem.parent
                if elem is None:
                    break
                for child in elem.find_all(string=True):
                    txt = child.strip()
                    if txt and txt.replace(".", "").replace(",", "").replace("%", "").isdigit():
                        wartosc = txt
                        break
                if wartosc:
                    break

            print(f"  {player_slug:<40} {wartosc}")

        print(f"\nHTML: {OUT_DIR / 'ranking_xg_2025_26.html'}")
        await asyncio.sleep(3)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())