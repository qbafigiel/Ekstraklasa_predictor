from pathlib import Path
from bs4 import BeautifulSoup

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "bobcek_ofensywa.html"
html = HTML_FILE.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

# Szukamy elementów z tekstem "2025/2026" lub "2026/2027"
print("=" * 70)
print("ELEMENTY Z TEKSTEM SEZONÓW")
print("=" * 70)

for sezon in ["2026/2027", "2025/2026", "2024/2025", "2023/2024"]:
    print(f"\n--- Szukam: '{sezon}' ---")
    elementy = soup.find_all(string=sezon)
    for i, el in enumerate(elementy[:3]):
        parent = el.parent
        parent_parent = parent.parent if parent else None
        print(f"  [{i}] Element: <{parent.name}> class='{parent.get('class', [])}'")
        if parent_parent:
            print(f"      Rodzic: <{parent_parent.name}> class='{parent_parent.get('class', [])}'")

# Szukamy też zakładek (buttony)
print("\n\n" + "=" * 70)
print("PRZYCISKI ZAKŁADEK (Ofensywa/Defensywa/Podania/Pozostałe)")
print("=" * 70)

for zakladka in ["Ofensywa", "Defensywa", "Podania", "Pozostałe"]:
    elementy = soup.find_all(string=zakladka)
    print(f"\n--- '{zakladka}' ---")
    for i, el in enumerate(elementy[:2]):
        parent = el.parent
        print(f"  [{i}] <{parent.name}> class='{parent.get('class', [])}'")