from pathlib import Path
from bs4 import BeautifulSoup

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "ranking_xg_2025_26.html"
html = HTML_FILE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

# Znajdź pierwszy link do Bobcka
link = soup.find("a", href=lambda h: h and "tomas-bobcek" in h)
print("HTML LINKA DO BOBCKA:")
print(link)
print()

# Idź w górę i wypisz strukturę do 5 poziomów
elem = link
for poziom in range(6):
    if elem.parent is None:
        break
    elem = elem.parent
    print(f"\n=== POZIOM +{poziom+1} ({elem.name}, class={elem.get('class', [])[:3]}...) ===")
    text = elem.get_text(" | ", strip=True)
    print(f"TEKST: {text[:200]}")
    if "19.8" in text:
        print("🎯 TU JEST WARTOŚĆ 19.8!")
        print(f"\nCAŁY HTML tego elementu (pierwsze 1500 znaków):")
        print(str(elem)[:1500])
        break