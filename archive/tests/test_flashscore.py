from playwright.sync_api import sync_playwright

URL = "https://www.flashscore.pl/mecz/pilka-nozna/cracovia-KvXSf2A6/lechia-gdansk-GGLmkiK8/szczegoly/statystyki/"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # widoczna przeglądarka
    page = browser.new_page()
    
    page.set_extra_http_headers({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "pl-PL,pl;q=0.9"
    })
    
    print("Wchodzę na stronę...")
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(5000)
    
    # Spróbuj zamknąć cookie banner
    try:
        page.click("button#onetrust-accept-btn-handler", timeout=3000)
        page.wait_for_timeout(1000)
        print("Zamknięto cookies")
    except:
        print("Brak bannera cookies")
    
    page.wait_for_timeout(2000)
    
    tekst = page.inner_text("body")
    linie = [l.strip() for l in tekst.split("\n") if l.strip()]
    
    print(f"\nLiczba linii: {len(linie)}")
    print("\n=== WSZYSTKIE LINIE ===")
    for i, l in enumerate(linie):
        print(f"[{i}] {l}")
    
    browser.close()