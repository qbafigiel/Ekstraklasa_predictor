"""
check_player_consistency.py
===========================
Sprawdza spójność nazw zawodników w bazie.
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")

def main():
    conn = sqlite3.connect(DB_PATH)

    # Top 30 najczęściej występujących zawodników (per klub)
    df = pd.read_sql("""
        SELECT team_name, player_name, COUNT(*) AS n_matches
        FROM lineups
        GROUP BY team_name, player_name
        ORDER BY n_matches DESC
        LIMIT 30
    """, conn)

    print("TOP 30 ZAWODNIKÓW WG LICZBY WYSTĄPIEŃ:")
    print("-" * 70)
    print(df.to_string(index=False))

    # Sprawdź czy są dziwne warianty nazw
    print("\n\nPRZYKŁAD Z JEDNEJ DRUŻYNY (LEGIA WARSZAWA):")
    print("-" * 70)
    df2 = pd.read_sql("""
        SELECT player_name, COUNT(*) AS n
        FROM lineups
        WHERE team_name = 'Legia Warszawa'
        GROUP BY player_name
        ORDER BY n DESC
        LIMIT 30
    """, conn)
    print(df2.to_string(index=False))

    # Ilu unikalnych zawodników per klub
    print("\n\nUNIKALNI ZAWODNICY PER KLUB:")
    print("-" * 70)
    df3 = pd.read_sql("""
        SELECT team_name, COUNT(DISTINCT player_name) AS unique_players
        FROM lineups
        GROUP BY team_name
        ORDER BY unique_players DESC
    """, conn)
    print(df3.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    main()