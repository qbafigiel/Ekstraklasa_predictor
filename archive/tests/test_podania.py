from playwright.sync_api import sync_playwright

url = "https://www.flashscore.pl/mecz/pilka-nozna/gornik-zabrze-2LH3Ywq4/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=6ou8D5jS"

with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    page.goto(url, wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(3000)
    tekst = page.inner_text("body")
    linie = [l.strip() for l in tekst.split("\n") if l.strip()]

    szukane = ["Podania", "odbioru", "rodkowania", "Długie"]
    for i, l in enumerate(linie):
        if any(s in l for s in szukane):
            for j in range(max(0, i-4), min(len(linie), i+5)):
                marker = ">>>" if j == i else "   "
                print(f"[{j:3d}] {marker} {linie[j]}")
            print()

    browser.close()