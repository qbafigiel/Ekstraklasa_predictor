from pathlib import Path
import re

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "bobcek_ofensywa.html"
html = HTML_FILE.read_text(encoding="utf-8")

# Szukamy wzorca liczby 19.8 (xG Bobcka) w różnych kontekstach
print("=" * 70)
print("SZUKAM LICZBOWEJ WARTOŚCI xG (19.8) w kodzie")
print("=" * 70)

# Wzorce jakie mogą się pojawić w JSON
wzorce = [
    r'"xG"\s*:\s*[\d.]+',           # "xG": 19.8
    r'"xG"\s*:\s*"[\d.]+',           # "xG": "19.8"
    r'"value"\s*:\s*19\.8',          # "value": 19.8
    r'19\.8[^0-9]',                  # 19.8 gdziekolwiek
    r'"key"\s*:\s*"xG"',             # "key": "xG"
]

for wzor in wzorce:
    matches = list(re.finditer(wzor, html))
    print(f"\nWzór: {wzor}")
    print(f"Znaleziono: {len(matches)} wystąpień")
    for m in matches[:3]:
        idx = m.start()
        fragment = html[max(0,idx-150):idx+200]
        print(f"  Pozycja {idx}:")
        print(f"  ...{fragment}...")
        print()

# Sprawdź też czy jest __NEXT_DATA__ (typowe dla Next.js)
print("\n" + "=" * 70)
print("SPRAWDZAM CZY JEST __NEXT_DATA__ (kluczowy JSON dla Next.js)")
print("=" * 70)

if "__NEXT_DATA__" in html:
    idx = html.find("__NEXT_DATA__")
    print(f"✓ ZNALEZIONO na pozycji {idx}")
    print(f"Fragment:")
    print(html[idx:idx+500])
else:
    print("✗ BRAK __NEXT_DATA__")

# Szukaj też innych typowych struktur Next.js/React
for wzor in ['"pageProps"', '"buildId"', 'self.__next_f']:
    if wzor in html:
        idx = html.find(wzor)
        print(f"\n✓ Znaleziono '{wzor}' na pozycji {idx}")