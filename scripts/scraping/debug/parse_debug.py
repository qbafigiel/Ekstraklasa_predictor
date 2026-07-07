"""
parse_debug.py
==============
Analizuje zapisany debug_lineups.html i wypisuje strukturę
sekcji: zmiany, absencje, trenerzy.
"""

from bs4 import BeautifulSoup
import re

with open("debug_lineups.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# ============================================================
# 1. PEŁNY TEKST BLOKU lf__lineUp (żeby zobaczyć strukturę)
# ============================================================
print("=" * 70)
print("1. PEŁNY TEKST .lf__lineUp")
print("=" * 70)
block = soup.select_one(".lf__lineUp")
if block:
    txt = block.get_text("\n", strip=True)
    print(txt[:5000])
else:
    print("Nie znaleziono .lf__lineUp")

# ============================================================
# 2. WSZYSTKIE UNIKALNE NAGŁÓWKI SEKCJI
# ============================================================
print("\n" + "=" * 70)
print("2. NAGŁÓWKI SEKCJI (szukamy separatorów)")
print("=" * 70)

# Szukamy elementów które mogą być nagłówkami sekcji
for cls in [
    "lf__title", "lf__header", "lf__label",
    "lf__sectionTitle", "lf__section",
    "lf__formation--extended",
]:
    els = soup.select(f".{cls}")
    if els:
        print(f"\n.{cls} ({len(els)} elementów):")
        for el in els:
            print(f"  '{el.get_text(strip=True)}'")

# ============================================================
# 3. STRUKTURA participantNew — wszystkie typy
# ============================================================
print("\n" + "=" * 70)
print("3. WSZYSTKIE .lf__participantNew — klasy i tekst")
print("=" * 70)
nodes = soup.select(".lf__participantNew")
print(f"Łącznie: {len(nodes)}")
for i, n in enumerate(nodes, 1):
    klasy = " ".join(n.get("class", []))
    txt = " ".join(n.get_text(" ", strip=True).split())
    print(f"[{i:02d}] {klasy}")
    print(f"      {txt[:120]}")

# ============================================================
# 4. SZUKAMY SEKCJI PO TEKŚCIE
# ============================================================
print("\n" + "=" * 70)
print("4. ELEMENTY ZAWIERAJĄCE KLUCZOWE SŁOWA")
print("=" * 70)

slowa = [
    "Zmienieni", "Składy wyjściowe", "Rezerwowi",
    "Wykluczeni", "Kontuzja", "Czerwona",
    "Trener", "Trenerzy",
]

for slowo in slowa:
    els = soup.find_all(string=re.compile(slowo, re.IGNORECASE))
    if els:
        print(f"\n'{slowo}' — znaleziono {len(els)}x:")
        for el in els[:3]:
            parent = el.parent
            print(f"  tag={parent.name} klasa={parent.get('class')} tekst='{el.strip()[:80]}'")

# ============================================================
# 5. STRUKTURA .lf__sides — zagnieżdżenie
# ============================================================
print("\n" + "=" * 70)
print("5. STRUKTURA .lf__sides")
print("=" * 70)
sides = soup.select_one(".lf__sides")
if sides:
    def print_tree(el, depth=0):
        if depth > 4:
            return
        cls = " ".join(el.get("class", []))
        txt = el.get_text(strip=True)[:60].replace("\n", " ")
        if cls or el.name != "div":
            print("  " * depth + f"<{el.name} class='{cls}'> {txt}")
        for child in el.find_all(recursive=False):
            print_tree(child, depth + 1)
    print_tree(sides)
else:
    print("Nie znaleziono .lf__sides")