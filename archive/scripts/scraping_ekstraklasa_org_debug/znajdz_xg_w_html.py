from pathlib import Path
import re

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "bobcek_ofensywa.html"
html = HTML_FILE.read_text(encoding="utf-8")

# Szukamy fragmentów wokół "XG" (nagłówek statystyki)
print("=" * 70)
print("FRAGMENTY HTML wokół 'XG' (±400 znaków)")
print("=" * 70)

# Znajdź "XG" jako osobne słowo (nie część XGOT)
pattern = re.compile(r'XG(?!OT)', re.IGNORECASE)

count = 0
for match in pattern.finditer(html):
    idx = match.start()
    # Pomiń jeśli to część większego słowa
    if idx > 0 and html[idx-1].isalpha():
        continue
    
    start = max(0, idx - 200)
    end = min(len(html), idx + 400)
    fragment = html[start:end]
    
    count += 1
    print(f"\n--- Wystąpienie {count} (pozycja {idx}) ---")
    print(fragment)
    print("-" * 70)
    
    if count >= 5:  # max 5 pierwszych żeby nie zaśmiecać
        break

print(f"\n\nZnaleziono {count} wystąpień 'XG' (pokazano max 5)")