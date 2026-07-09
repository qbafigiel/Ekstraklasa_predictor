from pathlib import Path

HTML_FILE = Path(__file__).resolve().parents[3] / "data" / "debug" / "bobcek_01_sezon_2025_26.html"

html = HTML_FILE.read_text(encoding="utf-8")

szukane = [
    # Sekcja główna
    ("Rozegrane mecze (30/34)", "30/34"),
    ("Gol co 123 min", "123"),
    ("Asysta co 411 min", "411"),
    # OFENSYWA (widzieliśmy na screenie)
    ("xG (19.8)", "19.8"),
    ("xGOT (20.1)", "20.1"),
    ("Konwersja (21%)", "21%"),
    ("Gole z gry (14)", ">14<"),
    # DEFENSYWA (nie widzieliśmy jeszcze - test)
    ("Tekst 'PRZECHWYT'", "PRZECHWYT"),
    ("Tekst 'POJEDYN'", "POJEDYN"),
    ("Tekst 'WYBICIA'", "WYBICIA"),
    # PODANIA
    ("Tekst 'XA'", "XA"),
    ("Tekst 'PODANIA CELNE'", "PODANIA CELNE"),
    ("Tekst 'KLUCZOWE'", "KLUCZOWE"),
    ("Tekst 'DOŚRODK'", "DOŚRODK"),
    # POZOSTAŁE
    ("Tekst 'DRYBLING'", "DRYBLING"),
    ("Tekst 'SPALON'", "SPALON"),
    ("Tekst 'FAULE'", "FAULE"),
    # Zakładki (nazwy)
    ("Zakładka OFENSYWA", "OFENSYWA"),
    ("Zakładka DEFENSYWA", "DEFENSYWA"),
    ("Zakładka PODANIA", "PODANIA"),
    ("Zakładka POZOSTAŁE", "POZOSTAŁE"),
]

print(f"Plik: {HTML_FILE.name}")
print(f"Rozmiar: {len(html)} znaków\n")
print(f"{'CO SZUKAM':<35} {'STATUS':<10} {'RAZY'}")
print("-" * 55)

wszystko_ok = True
for opis, wzor in szukane:
    ile = html.count(wzor)
    status = "✓" if ile > 0 else "✗"
    if ile == 0:
        wszystko_ok = False
    print(f"{opis:<35} {status:<10} {ile}")

print("\n" + "=" * 55)
if wszystko_ok:
    print("✅ WSZYSTKO JEST W JEDNYM PLIKU — nie trzeba klikać zakładek!")
else:
    print("⚠️  Część danych brakuje — trzeba klikać zakładki w Playwright")