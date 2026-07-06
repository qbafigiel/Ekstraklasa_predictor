"""
test_lineups.py v4
==================
Diagnoza struktury składów Flashscore:
- szuka nazw po fragmencie klasy
- wypisuje nagłówki sekcji
- wypisuje tekst z bloków lineup
"""

import asyncio
import sqlite3
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

DB_PATH = "db/ekstraklasa.db"


async def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT flash_id, flash_url, gospodarz, gosc, sezon
        FROM matches
        WHERE flash_id IS NOT NULL
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    flash_id, flash_url, home, away, season = row
    url = flash_url.replace("statystyki", "sklady")

    print(f"Mecz: {home} vs {away} ({season})")
    print(f"URL: {url}")

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
        except Exception as e:
            print(f"Błąd goto: {e}")

        await asyncio.sleep(5)

        content = await page.content()
        with open("debug_lineups.html", "w", encoding="utf-8") as f:
            f.write(content)

        soup = BeautifulSoup(content, "html.parser")

        print("\n" + "=" * 60)
        print("1. NAGŁÓWKI / HEADERS")
        print("=" * 60)
        headers = soup.select('[class*="formationHeader"]')
        print(f"Znaleziono headers: {len(headers)}")
        for i, h in enumerate(headers, 1):
            txt = " ".join(h.get_text(" ", strip=True).split())
            print(f"[{i}] {txt}")

        print("\n" + "=" * 60)
        print("2. NAZWY ZAWODNIKÓW PO FRAGMENCIE KLASY")
        print("=" * 60)
        name_nodes = soup.select('[class*="lineupsParticipantName_"]')
        print(f"Znaleziono name_nodes: {len(name_nodes)}")
        for i, n in enumerate(name_nodes[:80], 1):
            txt = " ".join(n.get_text(" ", strip=True).split())
            print(f"[{i:02d}] {txt}")

        print("\n" + "=" * 60)
        print("3. BLOKI .lf__lineUp")
        print("=" * 60)
        blocks = soup.select(".lf__lineUp")
        print(f"Znaleziono bloków .lf__lineUp: {len(blocks)}")
        for i, b in enumerate(blocks, 1):
            txt = " ".join(b.get_text(" ", strip=True).split())
            print(f"\n--- BLOCK {i} ---")
            print(txt[:1000] if txt else "[PUSTY]")

        print("\n" + "=" * 60)
        print("4. PIERWSZE 10 .lf__participantNew — pełny tekst")
        print("=" * 60)
        participants = soup.select(".lf__participantNew")
        print(f"Znaleziono participants: {len(participants)}")
        for i, pnode in enumerate(participants[:10], 1):
            txt = " ".join(pnode.get_text(" ", strip=True).split())
            classes = " ".join(pnode.get("class", []))
            print(f"\n[{i}] classes={classes}")
            print(f"text={txt if txt else '[PUSTY]'}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())