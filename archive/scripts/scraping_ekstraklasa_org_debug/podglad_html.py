from pathlib import Path

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "bobcek_01_sezon_2025_26.html"
html = HTML_FILE.read_text(encoding="utf-8")

# Znajdź wszystkie wystąpienia "19.8" i pokaż kontekst
print("=" * 70)
print("WSZYSTKIE WYSTĄPIENIA '19.8' Z KONTEKSTEM (±300 znaków)")
print("=" * 70)

idx = 0
nr = 1
while True:
    idx = html.find("19.8", idx)
    if idx == -1:
        break
    start = max(0, idx - 300)
    end = min(len(html), idx + 100)
    fragment = html[start:end]
    print(f"\n--- Wystąpienie {nr} (pozycja {idx}) ---")
    print(fragment)
    print("-" * 70)
    idx += 1
    nr += 1

# Sprawdź też "20.1" (xGOT)
print("\n\n" + "=" * 70)
print("WYSTĄPIENIA '20.1' Z KONTEKSTEM")
print("=" * 70)
idx = 0
while True:
    idx = html.find("20.1", idx)
    if idx == -1:
        break
    start = max(0, idx - 200)
    end = min(len(html), idx + 100)
    print(f"\n--- Pozycja {idx} ---")
    print(html[start:end])
    print("-" * 70)
    idx += 1