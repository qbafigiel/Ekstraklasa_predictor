import sqlite3
import pandas as pd

# --- FLASHSCORE lineups ---
conn = sqlite3.connect('db/ekstraklasa.db')

print("=== FLASHSCORE: przykładowe nazwiska (2025/26) ===")
rows = conn.execute(
    'SELECT DISTINCT player_name, team_name FROM lineups WHERE sezon = ? ORDER BY team_name, player_name LIMIT 40',
    ('2025/26',)
).fetchall()
for r in rows:
    print(f'{r[1]:30s} | {r[0]}')

print()
print("=== FLASHSCORE: liczba graczy per drużyna ===")
rows2 = conn.execute(
    'SELECT team_name, COUNT(DISTINCT player_name) as cnt FROM lineups WHERE sezon = ? GROUP BY team_name ORDER BY team_name',
    ('2025/26',)
).fetchall()
for r in rows2:
    print(f'{r[0]:30s} | {r[1]} graczy')

conn.close()

# --- EKSTRAKLASA.ORG ---
print()
print("=== EKSTRAKLASA.ORG: kolumny ===")
df = pd.read_csv('data/processed/zawodnicy_ekstraklasa_org_2025_26.csv')
print(list(df.columns[:8]))

print()
print("=== EKSTRAKLASA.ORG: kluby i przykładowi zawodnicy ===")
for klub in sorted(df['klub_slug'].unique()):
    graczy = df[df['klub_slug'] == klub]
    print(f'{klub:30s} | {len(graczy)} graczy')
    for _, row in graczy.head(3).iterrows():
        print(f'  slug: {row["player_slug"]:40s} | nazwa: {row["nazwa"]}')
    print()