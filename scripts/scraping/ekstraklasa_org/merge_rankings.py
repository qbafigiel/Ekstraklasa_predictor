import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
IN_DIR = ROOT / "data" / "raw" / "ekstraklasa_org"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parsuj_wartosc(v: str):
    """Zamień wartość ze stringa na float/int. '19.8' -> 19.8, '21%' -> 21, '–' -> 0"""
    v = v.strip()
    if not v or v in ("–", "-", "—"):
        return 0
    v = v.replace("%", "").replace(",", ".")
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except ValueError:
        return v  # zostaw jak jest jeśli nie liczba


def main():
    # Zbierz dane per zawodnik
    # klucz: player_slug -> dict {nazwa, klub_slug, staty...}
    zawodnicy = {}
    kolumny_staty = []
    
    csv_files = sorted(IN_DIR.glob("*.csv"))
    print(f"Znaleziono {len(csv_files)} plików CSV\n")
    
    for csv_file in csv_files:
        # Nazwa statystyki = nazwa pliku bez rozszerzenia
        nazwa_stat = csv_file.stem  # np. "pole_xg" albo "gk_czyste-konta"
        kolumny_staty.append(nazwa_stat)
        
        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slug = row["player_slug"]
                if not slug:
                    continue
                
                if slug not in zawodnicy:
                    zawodnicy[slug] = {
                        "player_slug": slug,
                        "nazwa": row["nazwa"],
                        "klub_slug": row["klub_slug"],
                    }
                # Zapisz wartość dla tej statystyki
                zawodnicy[slug][nazwa_stat] = parsuj_wartosc(row["wartosc"])
    
    print(f"Unikalnych zawodników: {len(zawodnicy)}")
    print(f"Kolumn statystyk: {len(kolumny_staty)}\n")
    
    # Uzupełnij brakujące wartości zerami
    for slug, dane in zawodnicy.items():
        for kol in kolumny_staty:
            if kol not in dane:
                dane[kol] = 0
    
    # Zapisz do CSV
    out_path = OUT_DIR / "zawodnicy_ekstraklasa_org_2025_26.csv"
    fieldnames = ["player_slug", "nazwa", "klub_slug"] + kolumny_staty
    
    # Sortuj po klubie i nazwie
    sorted_zaw = sorted(zawodnicy.values(), key=lambda z: (z["klub_slug"], z["nazwa"]))
    
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_zaw)
    
    print(f"✓ Zapisano: {out_path}")
    print(f"  Wiersze: {len(sorted_zaw)}")
    print(f"  Kolumny: {len(fieldnames)}")
    
    # Podgląd — top 5 wg xG
    print("\nTOP 5 wg xG:")
    top_xg = sorted(zawodnicy.values(), key=lambda z: z.get("pole_xg", 0), reverse=True)[:5]
    for z in top_xg:
        print(f"  {z['nazwa']:<30} {z['klub_slug']:<25} xG={z.get('pole_xg', 0)} min={z.get('pole_minuty', 0) or z.get('gk-minuty-rozegrane', 0)}")
    
    # Ilu zawodników per klub
    print("\nZawodnicy per klub:")
    per_klub = defaultdict(int)
    for z in zawodnicy.values():
        per_klub[z["klub_slug"]] += 1
    for klub, ile in sorted(per_klub.items()):
        print(f"  {klub:<30} {ile}")


if __name__ == "__main__":
    main()