import sqlite3
conn = sqlite3.connect("db/ekstraklasa.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM matches")
print("Wszystkich meczów:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM matches WHERE flash_id IS NOT NULL")
print("Z flash_id:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM matches WHERE flash_url IS NOT NULL")
print("Z flash_url:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM matches WHERE flash_id IS NULL")
print("BEZ flash_id:", cursor.fetchone()[0])

cursor.execute("""
    SELECT sezon, COUNT(*) as total,
    SUM(CASE WHEN flash_id IS NOT NULL THEN 1 ELSE 0 END) as z_flashid
    FROM matches
    GROUP BY sezon
    ORDER BY sezon
""")
print("\nPer sezon:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[2]}/{row[1]} meczów ma flash_id")

conn.close()