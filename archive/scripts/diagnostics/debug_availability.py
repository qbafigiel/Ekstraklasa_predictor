"""
debug_availability.py
=====================
Pokazuje CO siedzi w core roster dla podejrzanego meczu.
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")

TARGET_TEAM = "Cracovia"
TARGET_SEZON = "2025/26"
TARGET_KOLEJKA = 19
CORE_SIZE = 14


def main():
    conn = sqlite3.connect(DB_PATH)

    # Znajdź match_id
    match = pd.read_sql("""
        SELECT match_id, data_meczu, gospodarz, gosc
        FROM matches
        WHERE sezon = ? AND kolejka = ?
          AND (gospodarz = ? OR gosc = ?)
    """, conn, params=(TARGET_SEZON, TARGET_KOLEJKA, TARGET_TEAM, TARGET_TEAM))
    print("MECZ TARGETOWY:")
    print(match.to_string(index=False))

    match_id = int(match["match_id"].iloc[0])
    data_meczu = match["data_meczu"].iloc[0]

    # Pobierz wszystkie mecze Cracovii PRZED tym meczem
    hist = pd.read_sql("""
        SELECT m.match_id, m.sezon, m.data_meczu, m.kolejka
        FROM matches m
        WHERE (m.gospodarz = ? OR m.gosc = ?)
          AND m.data_meczu < ?
        ORDER BY m.data_meczu ASC
    """, conn, params=(TARGET_TEAM, TARGET_TEAM, data_meczu))
    print(f"\nHISTORIA CRACOVII PRZED MECZEM: {len(hist)} meczów")
    print(f"Sezony: {hist['sezon'].unique()}")

    # Akumuluj minuty
    hist_match_ids = list(zip(hist["sezon"], hist["match_id"]))

    if not hist_match_ids:
        print("Brak historii")
        return

    placeholders = ",".join(["(?,?)"] * len(hist_match_ids))
    params = [x for pair in hist_match_ids for x in pair]

    minutes = pd.read_sql(f"""
        SELECT player_name, SUM(minutes_played) AS total_min,
               COUNT(*) AS matches_in_squad
        FROM player_minutes
        WHERE team_name = ?
          AND (sezon, match_id) IN ({placeholders})
        GROUP BY player_name
        ORDER BY total_min DESC
    """, conn, params=(TARGET_TEAM, *params))

    hist_count = len(hist)
    minutes["avg_min"] = minutes["total_min"] / hist_count
    minutes["played_pct"] = (minutes["matches_in_squad"] / hist_count * 100).round(1)

    print(f"\nTOP 20 ZAWODNIKÓW CRACOVII WG SUMY MINUT (przed meczem):")
    print(minutes.head(20).to_string(index=False))

    # Core roster wg naszej metryki (avg_min)
    core = minutes.head(CORE_SIZE)
    print(f"\nCORE ROSTER ({CORE_SIZE} osób) — użyty do liczenia availability:")
    print(core.to_string(index=False))

    # Kto był w kadrze meczu targetowego
    squad = pd.read_sql("""
        SELECT player_name, is_starter
        FROM lineups
        WHERE sezon = ? AND match_id = ? AND team_name = ?
    """, conn, params=(TARGET_SEZON, match_id, TARGET_TEAM))
    print(f"\nKADRA MECZOWA CRACOVII W TYM MECZU: {len(squad)} osób")
    print(squad.to_string(index=False))

    # Kto z core JEST i kogo NIE MA
    squad_players = set(squad["player_name"])
    core_players = set(core["player_name"])

    available = core_players & squad_players
    missing = core_players - squad_players

    print(f"\nCORE DOSTĘPNI ({len(available)}):")
    for p in sorted(available):
        print(f"  ✓ {p}")

    print(f"\nCORE BRAKUJĄCY ({len(missing)}):")
    for p in sorted(missing):
        # Kiedy ostatnio grał?
        last = pd.read_sql("""
            SELECT data_meczu, kolejka, minutes_played
            FROM player_minutes pm
            JOIN matches m ON pm.sezon = m.sezon AND pm.match_id = m.match_id
            WHERE pm.team_name = ? AND pm.player_name = ?
              AND m.data_meczu < ?
            ORDER BY m.data_meczu DESC LIMIT 1
        """, conn, params=(TARGET_TEAM, p, data_meczu))
        if len(last) > 0:
            r = last.iloc[0]
            print(f"  ✗ {p} — ostatni mecz: {r['data_meczu']} kolejka {r['kolejka']} ({r['minutes_played']} min)")
        else:
            print(f"  ✗ {p} — brak danych")

    conn.close()


if __name__ == "__main__":
    main()