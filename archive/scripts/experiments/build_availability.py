"""
build_availability.py
=====================
Buduje ważone availability_score — sezon-aware wersja.

Kluczowa zmiana:
- historia liczona TYLKO z bieżącego sezonu drużyny
- pierwsze MIN_HISTORY meczów sezonu = neutralny score (1.0)
- eliminuje problem letnich transferów i budowy nowego składu

Zasady:
1. Dla każdego meczu M drużyny T w sezonie S:
   - bierzemy ostatnie RECENT_WINDOW meczów T z sezonu S PRZED M
   - jeśli mniej niż MIN_HISTORY -> score = 1.0
2. availability_score = suma_avg_min_dostępnych / suma_avg_min_core
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")

CORE_SIZE = 14
RECENT_WINDOW = 12       # ile ostatnich meczów z BIEŻĄCEGO sezonu
MIN_HISTORY = 6          # minimum meczów sezonowych żeby liczyć score


def ensure_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_availability (
            sezon TEXT NOT NULL,
            match_id INTEGER NOT NULL,
            team_side TEXT NOT NULL,
            team_name TEXT NOT NULL,
            core_size INTEGER NOT NULL,
            core_available INTEGER NOT NULL,
            availability_score REAL NOT NULL,
            history_matches INTEGER NOT NULL,
            PRIMARY KEY (sezon, match_id, team_side)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_avail_match
        ON match_availability (sezon, match_id)
    """)
    conn.commit()


def build_availability(conn):
    matches = pd.read_sql("""
        SELECT match_id, sezon, data_meczu, kolejka, gospodarz, gosc
        FROM matches
        ORDER BY data_meczu ASC, match_id ASC
    """, conn)

    minutes = pd.read_sql("""
        SELECT sezon, match_id, team_name, player_name, minutes_played
        FROM player_minutes
    """, conn)

    squad = pd.read_sql("""
        SELECT sezon, match_id, team_name, player_name
        FROM lineups
    """, conn)
    squad_set = set(
        (r["sezon"], r["match_id"], r["team_name"], r["player_name"])
        for _, r in squad.iterrows()
    )

    min_index = {}
    for _, r in minutes.iterrows():
        key = (r["sezon"], r["match_id"], r["team_name"])
        min_index.setdefault(key, {})[r["player_name"]] = r["minutes_played"]

    cur = conn.cursor()
    cur.execute("DELETE FROM match_availability")
    conn.commit()

    # Historia SEZONOWA: (sezon, team_name) -> [(match_id), ...]
    team_season_history = {}

    rows_to_insert = []

    for m in matches.to_dict("records"):
        sezon = m["sezon"]
        match_id = m["match_id"]
        home = m["gospodarz"]
        away = m["gosc"]

        for side, team_name in [("home", home), ("away", away)]:
            key = (sezon, team_name)
            history = team_season_history.get(key, [])
            hist_count = len(history)

            if hist_count >= MIN_HISTORY:
                recent = history[-RECENT_WINDOW:]
                recent_count = len(recent)

                player_minutes_sum = {}
                for hm in recent:
                    match_min = min_index.get((sezon, hm, team_name), {})
                    for player, mins in match_min.items():
                        player_minutes_sum[player] = \
                            player_minutes_sum.get(player, 0) + mins

                avg_minutes = {
                    p: total / recent_count
                    for p, total in player_minutes_sum.items()
                }

                if len(avg_minutes) < CORE_SIZE:
                    rows_to_insert.append((
                        sezon, match_id, side, team_name,
                        CORE_SIZE, CORE_SIZE, 1.0, hist_count
                    ))
                else:
                    sorted_core = sorted(avg_minutes.items(), key=lambda x: x[1], reverse=True)[:CORE_SIZE]

                    total_weight = sum(w for _, w in sorted_core)
                    available_weight = 0.0
                    available_count = 0

                    for player, weight in sorted_core:
                        if (sezon, match_id, team_name, player) in squad_set:
                            available_weight += weight
                            available_count += 1

                    score = available_weight / total_weight if total_weight > 0 else 1.0
                    rows_to_insert.append((
                        sezon, match_id, side, team_name,
                        CORE_SIZE, available_count, round(score, 4), hist_count
                    ))
            else:
                rows_to_insert.append((
                    sezon, match_id, side, team_name,
                    CORE_SIZE, CORE_SIZE, 1.0, hist_count
                ))

            # Update PO meczu
            team_season_history.setdefault(key, []).append(match_id)

    cur.executemany("""
        INSERT INTO match_availability
        (sezon, match_id, team_side, team_name,
         core_size, core_available, availability_score, history_matches)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    conn.commit()

    return len(rows_to_insert)


def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    print(f"Buduję availability_score (sezon-aware):")
    print(f"  CORE_SIZE={CORE_SIZE}")
    print(f"  RECENT_WINDOW={RECENT_WINDOW}")
    print(f"  MIN_HISTORY={MIN_HISTORY}")

    n = build_availability(conn)
    print(f"\nWstawiono {n} rekordów")

    print("\n" + "=" * 70)
    print("WERYFIKACJA")
    print("=" * 70)

    df = pd.read_sql("""
        SELECT
            COUNT(*) AS total,
            AVG(availability_score) AS avg_score,
            MIN(availability_score) AS min_score,
            MAX(availability_score) AS max_score
        FROM match_availability
        WHERE history_matches >= ?
    """, conn, params=(MIN_HISTORY,))
    print(f"\nStatystyki (history >= {MIN_HISTORY}):")
    print(df.to_string(index=False))

    print("\nROZKŁAD:")
    print("-" * 70)
    df_dist = pd.read_sql("""
        SELECT
            CASE
                WHEN availability_score >= 0.95 THEN '0.95-1.00 (pełny)'
                WHEN availability_score >= 0.90 THEN '0.90-0.95'
                WHEN availability_score >= 0.85 THEN '0.85-0.90'
                WHEN availability_score >= 0.80 THEN '0.80-0.85'
                WHEN availability_score >= 0.70 THEN '0.70-0.80'
                ELSE '<0.70 (osłabiony)'
            END AS bucket,
            COUNT(*) AS n
        FROM match_availability
        WHERE history_matches >= ?
        GROUP BY bucket
        ORDER BY bucket DESC
    """, conn, params=(MIN_HISTORY,))
    print(df_dist.to_string(index=False))

    print("\nTOP 15 NAJBARDZIEJ OSŁABIONYCH:")
    print("-" * 70)
    df2 = pd.read_sql("""
        SELECT a.sezon, a.team_name, a.availability_score,
               a.core_available, a.history_matches, m.kolejka, m.data_meczu
        FROM match_availability a
        JOIN matches m ON a.match_id = m.match_id AND a.sezon = m.sezon
        WHERE a.history_matches >= ?
        ORDER BY a.availability_score ASC
        LIMIT 15
    """, conn, params=(MIN_HISTORY,))
    print(df2.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()