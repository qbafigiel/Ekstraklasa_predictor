"""
audit_external_data.py
======================
Audyt danych zewnętrznych po scrapingu Flashscore.

Raportuje:
- coverage meczów z lineups
- coverage trenerów
- liczba absencji
- liczba zmian
- średnie per mecz
- rozbicie per sezon
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")


def main():
    if not DB_PATH.exists():
        print(f"Brak bazy: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    print("=" * 90)
    print("AUDYT DANYCH ZEWNĘTRZNYCH — FLASHSCORE")
    print("=" * 90)

    # ---------------------------------------------------------
    # 1. PODSTAWOWE LICZNIKI
    # ---------------------------------------------------------
    total_matches = pd.read_sql("SELECT COUNT(*) AS n FROM matches", conn)["n"].iloc[0]
    lineup_matches = pd.read_sql("SELECT COUNT(DISTINCT sezon || '_' || match_id) AS n FROM lineups", conn)["n"].iloc[0]
    coach_matches = pd.read_sql("SELECT COUNT(*) AS n FROM match_coaches", conn)["n"].iloc[0]
    abs_rows = pd.read_sql("SELECT COUNT(*) AS n FROM match_absences", conn)["n"].iloc[0]
    sub_rows = pd.read_sql("SELECT COUNT(*) AS n FROM match_substitutions", conn)["n"].iloc[0]

    print("\n[1] PODSTAWOWE LICZNIKI")
    print("-" * 90)
    print(f"Wszystkie mecze w bazie                : {total_matches}")
    print(f"Mecze ze składami (lineups)            : {lineup_matches}")
    print(f"Mecze z trenerami (match_coaches)      : {coach_matches}")
    print(f"Łączna liczba rekordów absencji        : {abs_rows}")
    print(f"Łączna liczba rekordów zmian           : {sub_rows}")

    # ---------------------------------------------------------
    # 2. COVERAGE PER SEZON
    # ---------------------------------------------------------
    df_season_matches = pd.read_sql("""
        SELECT sezon, COUNT(*) AS matches_total
        FROM matches
        GROUP BY sezon
        ORDER BY sezon
    """, conn)

    df_season_lineups = pd.read_sql("""
        SELECT sezon, COUNT(DISTINCT match_id) AS lineup_matches
        FROM lineups
        GROUP BY sezon
        ORDER BY sezon
    """, conn)

    df_season_coaches = pd.read_sql("""
        SELECT sezon, COUNT(*) AS coach_matches
        FROM match_coaches
        GROUP BY sezon
        ORDER BY sezon
    """, conn)

    df_season_abs = pd.read_sql("""
        SELECT sezon, COUNT(*) AS abs_rows
        FROM match_absences
        GROUP BY sezon
        ORDER BY sezon
    """, conn)

    df_season_subs = pd.read_sql("""
        SELECT sezon, COUNT(*) AS sub_rows
        FROM match_substitutions
        GROUP BY sezon
        ORDER BY sezon
    """, conn)

    df_cov = (
        df_season_matches
        .merge(df_season_lineups, on="sezon", how="left")
        .merge(df_season_coaches, on="sezon", how="left")
        .merge(df_season_abs, on="sezon", how="left")
        .merge(df_season_subs, on="sezon", how="left")
        .fillna(0)
    )

    df_cov["lineup_matches"] = df_cov["lineup_matches"].astype(int)
    df_cov["coach_matches"] = df_cov["coach_matches"].astype(int)
    df_cov["abs_rows"] = df_cov["abs_rows"].astype(int)
    df_cov["sub_rows"] = df_cov["sub_rows"].astype(int)

    df_cov["lineup_coverage"] = (df_cov["lineup_matches"] / df_cov["matches_total"]).round(4)
    df_cov["coach_coverage"] = (df_cov["coach_matches"] / df_cov["matches_total"]).round(4)
    df_cov["avg_abs_per_match"] = (df_cov["abs_rows"] / df_cov["matches_total"]).round(2)
    df_cov["avg_subs_per_match"] = (df_cov["sub_rows"] / df_cov["matches_total"]).round(2)

    print("\n[2] COVERAGE PER SEZON")
    print("-" * 90)
    print(df_cov.to_string(index=False))

    # ---------------------------------------------------------
    # 3. JAKOŚĆ LINEUPS
    # ---------------------------------------------------------
    df_lineups_per_match = pd.read_sql("""
        SELECT
            sezon,
            match_id,
            SUM(CASE WHEN is_starter = 1 THEN 1 ELSE 0 END) AS starters_total,
            SUM(CASE WHEN is_starter = 0 THEN 1 ELSE 0 END) AS bench_total,
            SUM(CASE WHEN is_starter = 1 AND team_side = 'home' THEN 1 ELSE 0 END) AS home_starters,
            SUM(CASE WHEN is_starter = 1 AND team_side = 'away' THEN 1 ELSE 0 END) AS away_starters,
            SUM(CASE WHEN is_starter = 0 AND team_side = 'home' THEN 1 ELSE 0 END) AS home_bench,
            SUM(CASE WHEN is_starter = 0 AND team_side = 'away' THEN 1 ELSE 0 END) AS away_bench
        FROM lineups
        GROUP BY sezon, match_id
        ORDER BY sezon, match_id
    """, conn)

    print("\n[3] JAKOŚĆ LINEUPS")
    print("-" * 90)
    if len(df_lineups_per_match) == 0:
        print("Brak danych w lineups")
    else:
        print(f"Mecze w lineups                         : {len(df_lineups_per_match)}")
        print(f"Śr. starterów / mecz                   : {df_lineups_per_match['starters_total'].mean():.2f}")
        print(f"Śr. ławka / mecz                       : {df_lineups_per_match['bench_total'].mean():.2f}")
        print(f"Min/max starterów / mecz               : {df_lineups_per_match['starters_total'].min()} / {df_lineups_per_match['starters_total'].max()}")
        print(f"Min/max ławki / mecz                   : {df_lineups_per_match['bench_total'].min()} / {df_lineups_per_match['bench_total'].max()}")

        bad_starters = df_lineups_per_match[
            (df_lineups_per_match["home_starters"] != 11) |
            (df_lineups_per_match["away_starters"] != 11)
        ]
        print(f"Mecze z !=11 starterami home/away      : {len(bad_starters)}")

    # ---------------------------------------------------------
    # 4. ABSENCJE PER MECZ
    # ---------------------------------------------------------
    df_abs_per_match = pd.read_sql("""
        SELECT sezon, match_id, COUNT(*) AS abs_count
        FROM match_absences
        GROUP BY sezon, match_id
    """, conn)

    print("\n[4] ABSENCJE")
    print("-" * 90)
    if len(df_abs_per_match) == 0:
        print("Brak absencji")
    else:
        print(f"Mecze z >=1 absencją                   : {len(df_abs_per_match)}")
        print(f"Śr. absencji w meczach z absencjami    : {df_abs_per_match['abs_count'].mean():.2f}")
        print(f"Max absencji w meczu                   : {df_abs_per_match['abs_count'].max()}")

    # ---------------------------------------------------------
    # 5. ZMIANY PER MECZ
    # ---------------------------------------------------------
    df_sub_per_match = pd.read_sql("""
        SELECT sezon, match_id, COUNT(*) AS sub_count
        FROM match_substitutions
        GROUP BY sezon, match_id
    """, conn)

    print("\n[5] ZMIANY")
    print("-" * 90)
    if len(df_sub_per_match) == 0:
        print("Brak zmian")
    else:
        print(f"Mecze z >=1 zmianą                     : {len(df_sub_per_match)}")
        print(f"Śr. zmian w meczach ze zmianami        : {df_sub_per_match['sub_count'].mean():.2f}")
        print(f"Max zmian w meczu                      : {df_sub_per_match['sub_count'].max()}")

    # ---------------------------------------------------------
    # 6. TOP POWODY ABSENCJI
    # ---------------------------------------------------------
    df_abs_reasons = pd.read_sql("""
        SELECT reason_raw, COUNT(*) AS n
        FROM match_absences
        GROUP BY reason_raw
        ORDER BY n DESC
        LIMIT 20
    """, conn)

    print("\n[6] TOP POWODY ABSENCJI")
    print("-" * 90)
    if len(df_abs_reasons) == 0:
        print("Brak danych")
    else:
        print(df_abs_reasons.to_string(index=False))

    # ---------------------------------------------------------
    # 7. TOP TRENERZY
    # ---------------------------------------------------------
    df_coaches = pd.read_sql("""
        SELECT coach_name, COUNT(*) AS n
        FROM (
            SELECT home_coach AS coach_name FROM match_coaches
            UNION ALL
            SELECT away_coach AS coach_name FROM match_coaches
        )
        WHERE coach_name IS NOT NULL AND TRIM(coach_name) <> ''
        GROUP BY coach_name
        ORDER BY n DESC
        LIMIT 20
    """, conn)

    print("\n[7] TOP TRENERZY")
    print("-" * 90)
    if len(df_coaches) == 0:
        print("Brak danych")
    else:
        print(df_coaches.to_string(index=False))

    conn.close()

    print("\n" + "=" * 90)
    print("GOTOWE")
    print("=" * 90)


if __name__ == "__main__":
    main()