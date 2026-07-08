"""
build_player_minutes.py
=======================
Buduje tabelę player_minutes z lineups + substitutions.

Zasady:
- starter: gra od minuty 0
- rezerwowy który wszedł: gra od minuty zmiany
- starter który zszedł: gra do minuty zmiany
- rezerwowy który nie wszedł: 0 minut
- domyślny koniec meczu: 90 minut (bez doliczonego czasu)

Efekt: tabela player_minutes z kolumnami:
- sezon, match_id, team_side, team_name, player_name
- minutes_played, started, came_on, went_off
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")
MATCH_LENGTH_MINUTES = 90


def ensure_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS player_minutes (
            sezon TEXT NOT NULL,
            match_id INTEGER NOT NULL,
            team_side TEXT NOT NULL,
            team_name TEXT NOT NULL,
            player_name TEXT NOT NULL,
            minutes_played INTEGER NOT NULL,
            started INTEGER NOT NULL,
            came_on INTEGER NOT NULL,
            went_off INTEGER NOT NULL,
            came_on_minute INTEGER,
            went_off_minute INTEGER,
            PRIMARY KEY (sezon, match_id, team_side, player_name)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pm_match ON player_minutes (sezon, match_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pm_player ON player_minutes (team_name, player_name)
    """)
    conn.commit()


def build_minutes(conn):
    lineups = pd.read_sql("""
        SELECT sezon, match_id, team_side, team_name, player_name, is_starter
        FROM lineups
    """, conn)

    subs = pd.read_sql("""
        SELECT sezon, match_id, team_side, team_name,
               player_in, player_out, minute_num
        FROM match_substitutions
        WHERE minute_num IS NOT NULL
    """, conn)

    # Wyczyść starą tabelę
    cur = conn.cursor()
    cur.execute("DELETE FROM player_minutes")
    conn.commit()

    # Grupujemy zmiany per (match, team_side)
    subs_by_match = {}
    for _, r in subs.iterrows():
        key = (r["sezon"], r["match_id"], r["team_side"])
        subs_by_match.setdefault(key, []).append(r)

    rows_to_insert = []

    grouped = lineups.groupby(["sezon", "match_id", "team_side"])

    for (sezon, match_id, side), grp in grouped:
        team_name = grp["team_name"].iloc[0]
        starters = set(grp[grp["is_starter"] == 1]["player_name"])
        bench = set(grp[grp["is_starter"] == 0]["player_name"])

        match_subs = subs_by_match.get((sezon, match_id, side), [])

        # Buduj minuty
        player_stats = {}  # player -> dict

        # Wszyscy starterzy zaczynają od 0
        for p in starters:
            player_stats[p] = {
                "minutes_played": MATCH_LENGTH_MINUTES,
                "started": 1,
                "came_on": 0,
                "went_off": 0,
                "came_on_minute": None,
                "went_off_minute": None,
            }

        # Wszyscy z ławki startowo mają 0 minut
        for p in bench:
            player_stats[p] = {
                "minutes_played": 0,
                "started": 0,
                "came_on": 0,
                "went_off": 0,
                "came_on_minute": None,
                "went_off_minute": None,
            }

        # Aplikuj zmiany
        for sub in match_subs:
            p_in = sub["player_in"]
            p_out = sub["player_out"]
            minute = int(sub["minute_num"])

            # Zawodnik schodzący
            if p_out in player_stats:
                if player_stats[p_out]["started"] == 1:
                    player_stats[p_out]["minutes_played"] = minute
                    player_stats[p_out]["went_off"] = 1
                    player_stats[p_out]["went_off_minute"] = minute

            # Zawodnik wchodzący
            if p_in in player_stats:
                if player_stats[p_in]["started"] == 0:
                    player_stats[p_in]["minutes_played"] = max(0, MATCH_LENGTH_MINUTES - minute)
                    player_stats[p_in]["came_on"] = 1
                    player_stats[p_in]["came_on_minute"] = minute

        # Dodaj do batch insert
        for player_name, stats in player_stats.items():
            rows_to_insert.append((
                sezon, match_id, side, team_name, player_name,
                stats["minutes_played"],
                stats["started"],
                stats["came_on"],
                stats["went_off"],
                stats["came_on_minute"],
                stats["went_off_minute"],
            ))

    cur.executemany("""
        INSERT INTO player_minutes
        (sezon, match_id, team_side, team_name, player_name,
         minutes_played, started, came_on, went_off,
         came_on_minute, went_off_minute)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)

    conn.commit()
    return len(rows_to_insert)


def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    print("Buduję tabelę player_minutes...")
    n = build_minutes(conn)
    print(f"Wstawiono {n} rekordów")

    # Weryfikacja
    print("\n" + "=" * 70)
    print("WERYFIKACJA")
    print("=" * 70)

    df = pd.read_sql("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT sezon || '_' || match_id) AS matches,
            SUM(started) AS total_starters,
            SUM(came_on) AS total_came_on,
            AVG(minutes_played) AS avg_minutes
        FROM player_minutes
    """, conn)
    print(df.to_string(index=False))

    # Top zawodnicy per minuty
    print("\nTOP 15 ZAWODNIKÓW WG SUMY MINUT:")
    print("-" * 70)
    df2 = pd.read_sql("""
        SELECT team_name, player_name,
               COUNT(*) AS matches_in_squad,
               SUM(started) AS starts,
               SUM(came_on) AS sub_ins,
               SUM(minutes_played) AS total_minutes
        FROM player_minutes
        GROUP BY team_name, player_name
        ORDER BY total_minutes DESC
        LIMIT 15
    """, conn)
    print(df2.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()