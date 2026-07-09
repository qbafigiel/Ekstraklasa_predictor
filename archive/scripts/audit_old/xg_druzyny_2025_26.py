import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "db" / "ekstraklasa.db"

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("""
    SELECT team, ROUND(SUM(xg), 2) AS xg_total, COUNT(*) AS mecze
    FROM (
        SELECT gospodarz AS team, xg_gosp AS xg FROM matches WHERE sezon='2025/26'
        UNION ALL
        SELECT gosc AS team, xg_gosc AS xg FROM matches WHERE sezon='2025/26'
    )
    WHERE xg IS NOT NULL
    GROUP BY team
    ORDER BY xg_total DESC
""")

print(f"{'DRUŻYNA':<25} {'xG':>8} {'MECZE':>8}")
print("-" * 45)
for team, xg, mecze in cur.fetchall():
    print(f"{team:<25} {xg:>8} {mecze:>8}")

conn.close()