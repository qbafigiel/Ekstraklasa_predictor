from pathlib import Path
from bs4 import BeautifulSoup

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "ranking_xg_2025_26.html"
html = HTML_FILE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

# Wszystkie linki do zawodników
linki = soup.find_all("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" in h)
print(f"Wszystkich linków do zawodników: {len(linki)}\n")

# Sprawdź kilka: pierwszy (Bobcek), 50-ty, 200-ty, ostatni
indeksy = [0, 3, 4, 5, 50, 200, 447]

for idx in indeksy:
    if idx >= len(linki):
        continue
    link = linki[idx]
    nazwa = link.get_text(strip=True)
    
    print(f"\n{'='*70}")
    print(f"REKORD #{idx}: {nazwa}")
    print(f"{'='*70}")
    
    # Rodzic bezpośredni
    parent = link.parent
    if parent:
        print(f"POZIOM +1: <{parent.name}> class={parent.get('class', [])[:5]}")
        text = parent.get_text(" | ", strip=True)
        print(f"  TEXT: {text[:150]}")
    
    # Dziadek
    if parent and parent.parent:
        grandparent = parent.parent
        print(f"POZIOM +2: <{grandparent.name}> class={grandparent.get('class', [])[:5]}")
        text = grandparent.get_text(" | ", strip=True)
        print(f"  TEXT: {text[:150]}")