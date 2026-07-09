from pathlib import Path
from bs4 import BeautifulSoup

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "ranking_xg_2025_26.html"
html = HTML_FILE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

# Kontenery rekordów mają klasę "grid" z "grid-cols-[40px_1fr_44px]" 
# (albo podobny wariant z inną szerokością wartości)
kontenery = soup.find_all("div", class_=lambda c: c and "grid" in c and "min-h-14" in c)
print(f"Znaleziono {len(kontenery)} kontenerów\n")

wyniki = []
for kont in kontenery:
    # Link do zawodnika
    link = kont.find("a", href=lambda h: h and "/kluby/" in h and "/zawodnik/" in h)
    if not link:
        continue
    
    href = link["href"]
    parts = href.strip("/").split("/")
    klub_slug = parts[1] if len(parts) > 1 else ""
    player_slug = parts[3] if len(parts) > 3 else ""
    nazwa = link.get_text(strip=True)
    
    # Pozycja: pierwszy <span> w kontenerze
    spany = kont.find_all("span", recursive=False)
    pozycja = spany[0].get_text(strip=True) if spany else "?"
    
    # Wartość: ostatni <span> w kontenerze (a raczej ostatni bezpośredni)
    wartosc = spany[-1].get_text(strip=True) if len(spany) > 1 else "?"
    
    wyniki.append({
        "pozycja": pozycja,
        "player_slug": player_slug,
        "klub_slug": klub_slug,
        "nazwa": nazwa,
        "wartosc": wartosc,
    })

# Pokaż 10 pierwszych i 5 ostatnich
print(f"PIERWSI 10:")
print(f"{'POZ':<5} {'NAZWA':<30} {'KLUB':<25} {'WARTOŚĆ':<10}")
print("-" * 75)
for w in wyniki[:10]:
    print(f"{w['pozycja']:<5} {w['nazwa']:<30} {w['klub_slug']:<25} {w['wartosc']:<10}")

print(f"\nOSTATNI 5:")
for w in wyniki[-5:]:
    print(f"{w['pozycja']:<5} {w['nazwa']:<30} {w['klub_slug']:<25} {w['wartosc']:<10}")

print(f"\nRAZEM: {len(wyniki)} rekordów")