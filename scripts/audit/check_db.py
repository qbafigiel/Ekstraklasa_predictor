import sqlite3
conn = sqlite3.connect("db/ekstraklasa.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM matches WHERE flash_id IS NOT NULL")
print("Mecze z flash_id:", cursor.fetchone())
cursor.execute("SELECT flash_id, flash_url, sezon, gospodarz, gosc FROM matches WHERE flash_id IS NOT NULL LIMIT 5")
for row in cursor.fetchall():
    print(row)
conn.close()