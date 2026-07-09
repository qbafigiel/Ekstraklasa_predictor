"""
Merge rankingów zawodników z ekstraklasa.org dla wybranego sezonu.

Użycie:
    python scripts/scraping/ekstraklasa_org/merge_rankings.py --season 2025-2026
    python scripts/scraping/ekstraklasa_org/merge_rankings.py --season 2024-2025
    python scripts/scraping/ekstraklasa_org/merge_rankings.py --season 2023-2024
    python scripts/scraping/ekstraklasa_org/merge_rankings.py --all
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

ROOT    = Path(__file__).resolve().parents[3]
IN_ROOT = ROOT / "data" / "raw" / "ekstraklasa_org"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SEASONS = ["2023-2024", "2024-2025", "2025-2026"]

SEASON_LABEL = {
    "2023-2024": "2023_24",
    "2024-2025": "2024_25",
    "2025-2026": "2025_26",
}


def parsuj_wartosc(v: str):
    """'19.8' -> 19.8 | '21%' -> 21 | '–' -> 0"""
    v = v.strip()
    if not v or v in ("–", "-", "—"):
        return 0
    v = v.replace("%", "").replace(",", ".")
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except ValueError:
        return v


def merge_season(season: str) -> None:
    in_dir = IN_ROOT / season

    if not in_dir.exists():
        print(f"[SKIP] Brak folderu: {in_dir}")
        return

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"[SKIP] Brak plików CSV w: {in_dir}")
        return

    label      = SEASON_LABEL[season]
    out_path   = OUT_DIR / f"zawodnicy_ekstraklasa_org_{label}.csv"

    print(f"\n=== {season} ===")
    print(f"Pliki CSV: {len(csv_files)}")

    zawodnicy    = {}
    kolumny_stat = []

    for csv_file in csv_files:
        nazwa_stat = csv_file.stem          # np. "pole_xg", "gk_czyste-konta"
        kolumny_stat.append(nazwa_stat)

        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slug = row.get("player_slug", "").strip()
                if not slug:
                    continue

                if slug not in zawodnicy:
                    zawodnicy[slug] = {
                        "player_slug": slug,
                        "nazwa":       row.get("nazwa", ""),
                        "klub_slug":   row.get("klub_slug", ""),
                    }

                zawodnicy[slug][nazwa_stat] = parsuj_wartosc(row.get("wartosc", ""))

    # Uzupełnij brakujące statystyki zerami
    for dane in zawodnicy.values():
        for kol in kolumny_stat:
            if kol not in dane:
                dane[kol] = 0

    # Zapis
    fieldnames   = ["player_slug", "nazwa", "klub_slug"] + kolumny_stat
    sorted_zaw   = sorted(zawodnicy.values(), key=lambda z: (z["klub_slug"], z["nazwa"]))

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_zaw)

    print(f"Zawodnicy:          {len(sorted_zaw)}")
    print(f"Kolumny statystyk:  {len(kolumny_stat)}")
    print(f"Zapisano:           {out_path}")

    # Podgląd top 5 xG
    xg_kol = "pole_xg" if "pole_xg" in kolumny_stat else None
    if xg_kol:
        print("\nTop 5 wg xG:")
        top = sorted(zawodnicy.values(), key=lambda z: z.get(xg_kol, 0), reverse=True)[:5]
        for z in top:
            print(f"  {z['nazwa']:<35} {z['klub_slug']:<25} xG={z.get(xg_kol, 0)}")

    # Per klub
    per_klub = defaultdict(int)
    for z in zawodnicy.values():
        per_klub[z["klub_slug"]] += 1
    print("\nZawodnicy per klub:")
    for klub, ile in sorted(per_klub.items()):
        print(f"  {klub:<30} {ile}")


def main():
    parser = argparse.ArgumentParser(description="Merge rankingów ekstraklasa.org per sezon")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--season",
        choices=ALL_SEASONS,
        help="Konkretny sezon: 2023-2024 | 2024-2025 | 2025-2026",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Wykonaj merge dla wszystkich 3 sezonów",
    )
    args = parser.parse_args()

    seasons = ALL_SEASONS if args.all else [args.season]

    for season in seasons:
        merge_season(season)

    print("\nGotowe.")


if __name__ == "__main__":
    main()