from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"

pliki = {
    "OFENSYWA":  "bobcek_ofensywa.html",
    "DEFENSYWA": "bobcek_defensywa.html",
    "PODANIA":   "bobcek_podania.html",
    "POZOSTALE": "bobcek_pozostale.html",
}

# Kluczowe słowa dla każdej zakładki (spodziewane statystyki)
oczekiwane = {
    "OFENSYWA":  ["XG", "XGOT", "KONWERSJA", "GOLE", "STRZAŁY", "KARN"],
    "DEFENSYWA": ["PRZECHWYT", "POJEDYN", "WYBICI", "ZABLOKOWA"],
    "PODANIA":   ["XA", "PODANIA", "KLUCZOWE", "DOŚRODK", "ROŻNE"],
    "POZOSTALE": ["DRYBLING", "SPALON", "FAULE", "MINUTY"],
}

print(f"{'ZAKŁADKA':<12} {'ROZMIAR':<10} {'ZNALEZIONE STATYSTYKI'}")
print("=" * 80)

for nazwa, plik in pliki.items():
    path = DEBUG_DIR / plik
    if not path.exists():
        print(f"{nazwa:<12} BRAK PLIKU")
        continue
    
    html = path.read_text(encoding="utf-8")
    rozmiar = len(html)
    
    znalezione = []
    brakuje = []
    for slowo in oczekiwane[nazwa]:
        if slowo in html.upper():
            znalezione.append(f"✓{slowo}")
        else:
            brakuje.append(f"✗{slowo}")
    
    print(f"{nazwa:<12} {rozmiar:<10} {' '.join(znalezione)}")
    if brakuje:
        print(f"{'':12} {'':10} BRAK: {' '.join(brakuje)}")
    print()