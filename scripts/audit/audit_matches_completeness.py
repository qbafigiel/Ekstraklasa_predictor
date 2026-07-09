import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "db" / "ekstraklasa.db"

# Wszystkie kolumny statystyk (pary gosp/gosc)
KOLUMNY_PAR = [
    ("gole", "gole_gosp", "gole_gosc"),
    ("xg", "xg_gosp", "xg_gosc"),
    ("posiadanie", "posiadanie_gosp", "posiadanie_gosc"),
    ("strzaly", "strzaly_gosp", "strzaly_gosc"),
    ("strzaly_celne", "celne_gosp", "celne_gosc"),
    ("strzaly_zablokowane", "strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc"),
    ("strzaly_niecelne", "strzaly_niecelne_gosp", "strzaly_niecelne_gosc"),
    ("rozne", "rozne_gosp", "rozne_gosc"),
    ("faule", "faule_gosp", "faule_gosc"),
    ("spalone", "spalone_gosp", "spalone_gosc"),
    ("zk", "zk_gosp", "zk_gosc"),
    ("czk", "czk_gosp", "czk_gosc"),
    ("druga_zk", "druga_zk_gosp", "druga_zk_gosc"),
    ("dosrodkowania", "dosrodkowania_gosp", "dosrodkowania_gosc"),
    ("dosrodkowania_celne", "dosrodkowania_celne_gosp", "dosrodkowania_celne_gosc"),
    ("odbiory", "odbiory_gosp", "odbiory_gosc"),
    ("podania", "podania_gosp", "podania_gosc"),
    ("podania_celne", "podania_celne_gosp", "podania_celne_gosc"),
]

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Ile meczów per sezon
cur.execute("SELECT sezon, COUNT(*) FROM matches GROUP BY sezon ORDER BY sezon")
sezony = cur.fetchall()

print("=" * 90)
print("AUDYT KOMPLETNOŚCI DANYCH MECZOWYCH PER SEZON")
print("=" * 90)

print(f"\nMecze w bazie:")
for sezon, ile in sezony:
    print(f"  {sezon}: {ile} meczów")

# Nagłówek tabeli
print(f"\n{'STATYSTYKA':<25}", end="")
for sezon, _ in sezony:
    print(f"{sezon:>18}", end="")
print()
print("-" * (25 + 18 * len(sezony)))

# Wiersze
for nazwa, kol_g, kol_a in KOLUMNY_PAR:
    print(f"{nazwa:<25}", end="")
    for sezon, total in sezony:
        q = f"""
            SELECT COUNT(*) FROM matches
            WHERE sezon = ?
              AND {kol_g} IS NOT NULL
              AND {kol_a} IS NOT NULL
        """
        cur.execute(q, (sezon,))
        ile = cur.fetchone()[0]
        proc = round(100 * ile / total, 0) if total > 0 else 0

        if ile == total:
            status = f"{ile}/{total} ✓"
        elif ile == 0:
            status = f"{ile}/{total} ✗ (0%)"
        else:
            status = f"{ile}/{total} ({proc:.0f}%)"

        print(f"{status:>18}", end="")
    print()

print()
print("=" * 90)
print("Legenda:")
print("  ✓  = pełne pokrycie")
print("  ✗  = całkowity brak danych")
print("  %  = częściowe pokrycie")

conn.close()