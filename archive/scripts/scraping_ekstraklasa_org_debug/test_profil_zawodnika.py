import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

URL = "https://ekstraklasa.org/kluby/lechia-gdansk/zawodnik/tomas-bobcek/"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# UWAGA: tekst z małych/wielkich liter — w HTML jest "Defensywa" a nie "DEFENSYWA"
ZAKLADKI = ["Defensywa", "Podania", "Pozostałe"]


async def scroll_do_dolu(page):
    for _ in range(10):
        await page.evaluate("window.scrollBy(0, 500)")
        await asyncio.sleep(0.3)
    await asyncio.sleep(1)


async def main():
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        print(f"Otwieram: {URL}")
        await page.goto(URL, wait_until="networkidle", timeout=60000)
        await asyncio.sleep(3)

        # Zmiana sezonu
        print("Zmieniam sezon na 2025/2026...")
        await page.get_by_text("2026/2027").first.click()
        await asyncio.sleep(1)
        await page.get_by_text("2025/2026").first.click()
        await asyncio.sleep(3)
        print("✓ Sezon zmieniony")

        # Scroll i zapis OFENSYWA (domyślna)
        await scroll_do_dolu(page)
        html = await page.content()
        (OUT_DIR / "bobcek_ofensywa.html").write_text(html, encoding="utf-8")
        print(f"✓ Zapisano OFENSYWA ({len(html)} znaków)")

        # Klikanie zakładek (właściwe nazwy!)
        for nazwa in ZAKLADKI:
            print(f"\nKlikam: {nazwa}")
            try:
                # Szukamy button z tekstem
                btn = page.locator("button").filter(has_text=nazwa).first
                await btn.click(timeout=10000)
                await asyncio.sleep(2)
                await scroll_do_dolu(page)
                html = await page.content()
                fname = f"bobcek_{nazwa.lower().replace('ł', 'l').replace('ż','z').replace('ę','e').replace('ó','o').replace('ą','a')}.html"
                (OUT_DIR / fname).write_text(html, encoding="utf-8")
                print(f"✓ Zapisano {fname} ({len(html)} znaków)")
            except Exception as e:
                print(f"✗ Błąd przy {nazwa}: {e}")

        print(f"\nPliki w: {OUT_DIR}")
        await asyncio.sleep(5)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())