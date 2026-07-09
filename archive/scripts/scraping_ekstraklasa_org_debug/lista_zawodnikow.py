import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

URL = "https://ekstraklasa.org/statystyki/?season=2025-2026&ranking=xg"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        print(f"Otwieram: {URL}")
        await page.goto(URL, wait_until="networkidle", timeout=60000)
        await asyncio.sleep(3)

        # Kliknij "Pokaż więcej" (jeśli jest)
        print("Szukam przycisku 'Pokaż więcej'...")
        try:
            btn = page.locator("button").filter(has_text="Pokaż więcej").first
            await btn.click(timeout=10000)
            await asyncio.sleep(3)
            print("✓ Kliknięto 'Pokaż więcej'")
        except Exception as e:
            print(f"⚠ Brak przycisku lub błąd: {e}")

        # Scrolluj do samego dołu
        print("Scrolluję do końca strony...")
        prev_height = 0
        for i in range(30):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                print(f"✓ Osiągnięto koniec strony po {i+1} scrollach")
                break
            prev_height = new_height

        # Zapisz HTML dla debugu
        html = await page.content()
        (OUT_DIR / "lista_xg_2025_26.html").write_text(html, encoding="utf-8")
        print(f"✓ Zapisano HTML ({len(html)} znaków)")

        # Parsuj linki do zawodników
        soup = BeautifulSoup(html, "html.parser")
        linki = soup.find_all("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" in h)
        
        zawodnicy = []
        widziane_url = set()
        for a in linki:
            href = a["href"]
            if href in widziane_url:
                continue
            widziane_url.add(href)
            
            # Pełny URL
            if href.startswith("/"):
                full_url = "https://ekstraklasa.org" + href
            else:
                full_url = href
            
            # Wyciągnij nazwiska i klub z URL: /kluby/{klub}/zawodnik/{gracz}/
            parts = href.strip("/").split("/")
            klub_slug = parts[1] if len(parts) > 1 else ""
            player_slug = parts[3] if len(parts) > 3 else ""
            
            # Nazwa gracza z tekstu linku
            nazwa = a.get_text(strip=True)
            
            zawodnicy.append({
                "player_slug": player_slug,
                "klub_slug": klub_slug,
                "nazwa_z_linku": nazwa,
                "url": full_url,
            })

        print(f"\n✓ Znaleziono {len(zawodnicy)} unikalnych zawodników")
        
        # Pokaż 5 pierwszych i 5 ostatnich
        print("\nPIERWSI 5:")
        for z in zawodnicy[:5]:
            print(f"  {z['nazwa_z_linku']:<30} {z['klub_slug']:<25} {z['url']}")
        
        print("\nOSTATNI 5:")
        for z in zawodnicy[-5:]:
            print(f"  {z['nazwa_z_linku']:<30} {z['klub_slug']:<25} {z['url']}")

        # Zapisz do JSON
        out_json = OUT_DIR / "zawodnicy_2025_26.json"
        out_json.write_text(json.dumps(zawodnicy, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ Zapisano listę: {out_json}")

        await asyncio.sleep(3)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())