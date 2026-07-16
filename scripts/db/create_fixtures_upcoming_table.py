"""
create_fixtures_upcoming_table.py
==================================
Tworzy tabele fixtures_upcoming w bazie ekstraklasa.db.

Ta tabela trzyma mecze:
- zaplanowane, ale jeszcze nierozegrane
- przelozone
- z niepelnymi metadanymi (brak daty/godziny/sedziego)

Po rozegraniu meczu dane trafiaja do 'matches',
a rekord w 'fixtures_upcoming' dostaje status='played' + played_match_id.
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "ekstraklasa.db"


DDL = """
CREATE TABLE IF NOT EXISTS fixtures_upcoming (
    fixture_id           TEXT PRIMARY KEY,
    sezon                TEXT NOT NULL,
    kolejka              INTEGER NOT NULL,
    gospodarz            TEXT NOT NULL,
    gosc                 TEXT NOT NULL,
    data_planowana       TEXT,
    godzina              TEXT,
    stadion              TEXT,
    referee_full_name    TEXT,
    source_url           TEXT,
    ekstraklasa_uuid     TEXT,
    status               TEXT NOT NULL DEFAULT 'scheduled',
    played_match_id      INTEGER,
    created_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sezon, kolejka, gospodarz, gosc)
);

CREATE INDEX IF NOT EXISTS idx_fixtures_sezon_kolejka
    ON fixtures_upcoming(sezon, kolejka);

CREATE INDEX IF NOT EXISTS idx_fixtures_status
    ON fixtures_upcoming(status);

CREATE INDEX IF NOT EXISTS idx_fixtures_data
    ON fixtures_upcoming(data_planowana);
"""


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Baza {DB_PATH} nie istnieje.")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(DDL)
        conn.commit()

        cols = conn.execute("PRAGMA table_info(fixtures_upcoming)").fetchall()
        print("OK: tabela fixtures_upcoming utworzona/gotowa.")
        print("\nKolumny:")
        for c in cols:
            print(f"  {c[1]:<24} {c[2]}")

        n = conn.execute("SELECT COUNT(*) FROM fixtures_upcoming").fetchone()[0]
        print(f"\nAktualnych rekordow: {n}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()