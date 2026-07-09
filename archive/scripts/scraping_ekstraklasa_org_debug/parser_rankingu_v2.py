from pathlib import Path
from bs4 import BeautifulSoup

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "ranking_xg_2025_26.html"
html = HTML_FILE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

# Kontenery pełnej listy: min-h-20 (nie min-h-14 które jest podglądem)
kontenery = soup.find_all("div", class_=lambda c: c and "grid" in c and "min-h-20" in c)
print(f"Kontenerów min-h-20: {len(kontenery)}\n")

wyniki = []
widziane = set()

for kont in kontenery:
    link = kont.find("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" in h)
    if not link:
        continue
    
    href = link["href"]
    parts = href.strip("/").split("/")
    klub_slug = parts[1] if len(parts) > 1 else ""
    player_slug = parts[3] if len(parts) > 3 else ""
    
    if player_slug in widziane:
        continue
    widziane.add(player_slug)
    
    nazwa = link.get_text(strip=True)
    
    # Pozycja: pierwszy <span> w kontenerze (bezpośredni)
    spany_bezposr = kont.find_all("span", recursive=False)
    pozycja = spany_bezposr[0].get_text(strip=True) if spany_bezposr else "?"
    wartosc = spany_bezposr[-1].get_text(strip=True) if len(spany_bezposr) >= 2 else "?"
    
    wyniki.append({
        "pozycja": pozycja,
        "player_slug": player_slug,
        "klub_slug": klub_slug,
        "nazwa": nazwa,
        "wartosc": wartosc,
    })

print(f"Unikalnych zawodników: {len(wyniki)}\n")

# Pokaż 10 pierwszych
print("PIERWSI 10:")
print(f"{'POZ':<5} {'NAZWA':<30} {'KLUB':<25} {'WARTOŚĆ'}")
print("-" * 75)
for w in wyniki[:10]:
    print(f"{w['pozycja']:<5} {w['nazwa']:<30} {w['klub_slug']:<25} {w['wartosc']}")

# 5 ostatnich
print("\nOSTATNI 5:")
for w in wyniki[-5:]:
    print(f"{w['pozycja']:<5} {w['nazwa']:<30} {w['klub_slug']:<25} {w['wartosc']}")