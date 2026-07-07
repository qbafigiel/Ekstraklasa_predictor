"""
backfill_flash_2023_24_v2.py
============================
Poprawny backfill flash_id i flash_url dla sezonu 2023/24.

WAŻNE:
- czyści poprzedni błędny backfill dla 2023/24
- NIE zakłada kolejności home/away w URL Flashscore
- używa ręcznego mapowania slug -> nazwa drużyny
- matchuje po: sezon + data + kolejka + para drużyn (unordered)
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path("db/ekstraklasa.db")
RAW_CSV = Path("data/raw/flash/flash_2023_24.csv")
SEASON = "2023/24"

SLUG_MAP = {
    "jagiellonia-bialystok": "Jagiellonia Białystok",
    "warta-poznan": "Warta Poznań",
    "pogon-szczecin": "Pogoń Szczecin",
    "legia-warszawa": "Legia Warszawa",
    "lks-lodz": "ŁKS Łódź",
    "cracovia": "Cracovia",
    "piast-gliwice": "Piast Gliwice",
    "lech-poznan": "Lech Poznań",
    "gornik-zabrze": "Górnik Zabrze",
    "korona-kielce": "Korona Kielce",
    "puszcza": "Puszcza Niepołomice",
    "rakow-czestochowa": "Raków Częstochowa",
    "radomiak-radom": "Radomiak Radom",
    "ruch-chorzow": "Ruch Chorzów",
    "slask-wroclaw": "Śląsk Wrocław",
    "stal-mielec": "Stal Mielec",
    "widzew-lodz": "Widzew Łódź",
    "zaglebie-lubin": "Zagłębie Lubin",
}


def extract_team_segments(url: str):
    """
    Z URL wyciąga dwa segmenty drużynowe.

    Przykład:
    https://www.flashscore.pl/mecz/pilka-nozna/jagiellonia-bialystok-lIDaZJTc/warta-poznan-CrVUWAl8/szczegoly/statystyki/?mid=phWZm4I5
    """
    if not url or pd.isna(url):
        return None, None

    m = re.search(r"/mecz/pilka-nozna/([^/]+)/([^/]+)/szczegoly/", str(url))
    if not m:
        return None, None

    return m.group(1), m.group(2)


def slug_prefix(segment: str):
    """
    Odcina końcowy identyfikator Flashscore:
    'jagiellonia-bialystok-lIDaZJTc' -> 'jagiellonia-bialystok'
    'warta-poznan-CrVUWAl8' -> 'warta-poznan'
    """
    if not segment:
        return None

    m = re.match(r"^(.*)-[A-Za-z0-9]{6,}$", segment)
    if m:
        return m.group(1)
    return segment


def resolve_team(segment: str):
    if not segment:
        return None
    return SLUG_MAP.get(slug_prefix(segment))


def normalize_date(x):
    if pd.isna(x):
        return None
    return str(x).strip()[:10]


def normalize_round(x):
    if pd.isna(x):
        return None
    return int(float(x))


def main():
    if not DB_PATH.exists():
        print(f"Brak bazy: {DB_PATH}")
        return

    if not RAW_CSV.exists():
        print(f"Brak pliku CSV: {RAW_CSV}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=" * 80)
    print("BACKFILL FLASH URL/ID DLA 2023/24 — WERSJA POPRAWIONA")
    print("=" * 80)

    # 1. Czyścimy sezon 2023/24 z błędnego partial backfillu
    cur.execute("""
        UPDATE matches
        SET flash_id = NULL,
            flash_url = NULL
        WHERE sezon = ?
    """, (SEASON,))
    conn.commit()

    print("Wyczyszczono flash_id / flash_url dla sezonu 2023/24")
    print()

    df_raw = pd.read_csv(RAW_CSV)

    updated = 0
    unresolved = []
    not_found = []
    duplicates = []

    for _, row in df_raw.iterrows():
        url = row.get("url")
        flash_id = row.get("flash_id")
        data_meczu = normalize_date(row.get("data_meczu_flash"))
        kolejka = normalize_round(row.get("kolejka_flash"))

        if pd.isna(url) or pd.isna(flash_id):
            continue

        url = str(url).strip()
        flash_id = str(flash_id).strip()

        seg1, seg2 = extract_team_segments(url)
        team1 = resolve_team(seg1)
        team2 = resolve_team(seg2)

        if not team1 or not team2:
            unresolved.append({
                "url": url,
                "seg1": seg1,
                "seg2": seg2,
                "team1": team1,
                "team2": team2,
            })
            continue

        # Matchujemy po dacie, kolejce i parze drużyn bez względu na kolejność
        cur.execute("""
            SELECT match_id, gospodarz, gosc
            FROM matches
            WHERE sezon = ?
              AND data_meczu = ?
              AND kolejka = ?
              AND (
                    (gospodarz = ? AND gosc = ?)
                 OR (gospodarz = ? AND gosc = ?)
              )
        """, (
            SEASON,
            data_meczu,
            kolejka,
            team1, team2,
            team2, team1
        ))

        rows = cur.fetchall()

        if len(rows) == 1:
            match_id = rows[0][0]
            cur.execute("""
                UPDATE matches
                SET flash_id = ?, flash_url = ?
                WHERE match_id = ? AND sezon = ?
            """, (flash_id, url, match_id, SEASON))
            updated += 1

        elif len(rows) == 0:
            not_found.append({
                "data_meczu": data_meczu,
                "kolejka": kolejka,
                "team1": team1,
                "team2": team2,
                "flash_id": flash_id,
                "url": url,
            })

        else:
            duplicates.append({
                "data_meczu": data_meczu,
                "kolejka": kolejka,
                "team1": team1,
                "team2": team2,
                "flash_id": flash_id,
                "url": url,
                "rows": rows,
            })

    conn.commit()

    cur.execute("""
        SELECT COUNT(*)
        FROM matches
        WHERE sezon = '2023/24' AND flash_id IS NOT NULL
    """)
    season_count = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*)
        FROM matches
        WHERE flash_id IS NOT NULL
    """)
    total_count = cur.fetchone()[0]

    conn.close()

    print("=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)
    print(f"Updated rows        : {updated}")
    print(f"2023/24 with flash  : {season_count} / 306")
    print(f"Total with flash    : {total_count} / 918")
    print(f"Unresolved          : {len(unresolved)}")
    print(f"Not found in DB     : {len(not_found)}")
    print(f"Duplicate matches   : {len(duplicates)}")

    if unresolved:
        print("\n--- UNRESOLVED (max 20) ---")
        for x in unresolved[:20]:
            print(x)

    if not_found:
        print("\n--- NOT FOUND IN DB (max 20) ---")
        for x in not_found[:20]:
            print(x)

    if duplicates:
        print("\n--- DUPLICATES (max 20) ---")
        for x in duplicates[:20]:
            print(x)

    print("=" * 80)


if __name__ == "__main__":
    main()