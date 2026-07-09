from pathlib import Path
from bs4 import BeautifulSoup

DEBUG_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"

pliki = {
    "OFENSYWA":  "bobcek_ofensywa.html",
    "DEFENSYWA": "bobcek_defensywa.html",
    "PODANIA":   "bobcek_podania.html",
    "POZOSTALE": "bobcek_pozostale.html",
}

for nazwa, plik in pliki.items():
    path = DEBUG_DIR / plik
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    
    # Szukamy wszystkich <p> z klasą "label-medium-bold ... uppercase"
    # (nazwa statystyki) — sąsiadujące <span> z klasą "heading-small" to wartość
    
    print(f"\n{'='*60}")
    print(f"ZAKŁADKA: {nazwa}")
    print(f"{'='*60}")
    
    labels = soup.find_all("p", class_=lambda c: c and "label-medium-bold" in c and "uppercase" in c)
    
    znalezione = 0
    for label in labels:
        # Szukamy najbliższego <span> w tym samym kontenerze
        parent = label.parent
        span = parent.find("span", class_=lambda c: c and "heading-small" in c) if parent else None
        
        if span:
            nazwa_stat = label.get_text(strip=True)
            wartosc = span.get_text(strip=True)
            print(f"  {nazwa_stat:<40} {wartosc}")
            znalezione += 1
    
    print(f"\nRAZEM w {nazwa}: {znalezione} statystyk")