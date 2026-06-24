from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "data" / "processed" / "mvp_merged_2023_26.csv"
DB_PATH = ROOT / "db" / "ekstraklasa.db"
TABLE_NAME = "matches"

REQUIRED_COLUMNS = [
    "match_id",
    "sezon",
    "waga_sezonu",
    "kolejka",
    "data_meczu",
    "gospodarz",
    "gosc",
    "gole_gosp",
    "gole_gosc",
    "posiadanie_gosp",
    "posiadanie_gosc",
    "strzaly_gosp",
    "strzaly_gosc",
    "celne_gosp",
    "celne_gosc",
    "strzaly_zablokowane_gosp",
    "strzaly_zablokowane_gosc",
    "strzaly_niecelne_gosp",
    "strzaly_niecelne_gosc",
    "rozne_gosp",
    "rozne_gosc",
    "faule_gosp",
    "faule_gosc",
    "spalone_gosp",
    "spalone_gosc",
    "zk_gosp",
    "zk_gosc",
    "czk_gosp",
    "czk_gosc",
    "druga_zk_gosp",
    "druga_zk_gosc",
    "dosrodkowania_gosp",
    "dosrodkowania_gosc",
    "dosrodkowania_celne_gosp",
    "dosrodkowania_celne_gosc",
    "odbiory_gosp",
    "odbiory_gosc",
    "podania_gosp",
    "podania_gosc",
    "podania_celne_gosp",
    "podania_celne_gosc",
    "xg_gosp",
    "xg_gosc",
    "flash_id",
    "flash_url",
]

NUMERIC_COLUMNS = [
    "match_id",
    "waga_sezonu",
    "kolejka",
    "gole_gosp",
    "gole_gosc",
    "posiadanie_gosp",
    "posiadanie_gosc",
    "strzaly_gosp",
    "strzaly_gosc",
    "celne_gosp",
    "celne_gosc",
    "strzaly_zablokowane_gosp",
    "strzaly_zablokowane_gosc",
    "strzaly_niecelne_gosp",
    "strzaly_niecelne_gosc",
    "rozne_gosp",
    "rozne_gosc",
    "faule_gosp",
    "faule_gosc",
    "spalone_gosp",
    "spalone_gosc",
    "zk_gosp",
    "zk_gosc",
    "czk_gosp",
    "czk_gosc",
    "druga_zk_gosp",
    "druga_zk_gosc",
    "dosrodkowania_gosp",
    "dosrodkowania_gosc",
    "dosrodkowania_celne_gosp",
    "dosrodkowania_celne_gosc",
    "odbiory_gosp",
    "odbiory_gosc",
    "podania_gosp",
    "podania_gosc",
    "podania_celne_gosp",
    "podania_celne_gosc",
    "xg_gosp",
    "xg_gosc",
]

TEXT_COLUMNS = [
    "sezon",
    "data_meczu",
    "gospodarz",
    "gosc",
    "flash_id",
    "flash_url",
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Nie udało się odczytać pliku: {path}")


def validate_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]

    if missing:
        raise RuntimeError(
            "Brakuje wymaganych kolumn w pliku CSV:\n- " + "\n- ".join(missing)
        )

    print("✅ Walidacja kolumn OK")
    print(f"Kolumn wymaganych: {len(REQUIRED_COLUMNS)}")
    print(f"Kolumn w pliku    : {len(df.columns)}")

    if extra:
        print("ℹ️ Dodatkowe kolumny w pliku (zostaną zignorowane):")
        for col in extra:
            print(f" - {col}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Zostaw tylko kolumny wymagane i w tej kolejności
    df = df[REQUIRED_COLUMNS]

    # Konwersja kolumn numerycznych
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Konwersja tekstów
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    # Puste stringi -> None
    for col in TEXT_COLUMNS:
        df[col] = df[col].replace({"": None, "nan": None, "<NA>": None})

    # Match ID i kolejka jako int tam gdzie się da
    if df["match_id"].isna().any():
        raise RuntimeError("Kolumna 'match_id' zawiera braki po konwersji numerycznej.")
    if df["kolejka"].isna().any():
        raise RuntimeError("Kolumna 'kolejka' zawiera braki po konwersji numerycznej.")

    df["match_id"] = df["match_id"].astype(int)
    df["kolejka"] = df["kolejka"].astype(int)

    # Podstawowe walidacje
    if df["sezon"].isna().any():
        raise RuntimeError("Kolumna 'sezon' zawiera puste wartości.")
    if df["gospodarz"].isna().any():
        raise RuntimeError("Kolumna 'gospodarz' zawiera puste wartości.")
    if df["gosc"].isna().any():
        raise RuntimeError("Kolumna 'gosc' zawiera puste wartości.")

    # Zamień pandasowe NA na None pod SQLite
    df = df.astype(object).where(pd.notna(df), None)

    return df


def create_table(conn: sqlite3.Connection):
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

    conn.execute(f"""
    CREATE TABLE {TABLE_NAME} (
        match_id INTEGER NOT NULL,
        sezon TEXT NOT NULL,
        waga_sezonu REAL NOT NULL,
        kolejka INTEGER NOT NULL,
        data_meczu TEXT NOT NULL,
        gospodarz TEXT NOT NULL,
        gosc TEXT NOT NULL,

        gole_gosp REAL,
        gole_gosc REAL,

        posiadanie_gosp REAL,
        posiadanie_gosc REAL,
        strzaly_gosp REAL,
        strzaly_gosc REAL,
        celne_gosp REAL,
        celne_gosc REAL,
        strzaly_zablokowane_gosp REAL,
        strzaly_zablokowane_gosc REAL,
        strzaly_niecelne_gosp REAL,
        strzaly_niecelne_gosc REAL,
        rozne_gosp REAL,
        rozne_gosc REAL,
        faule_gosp REAL,
        faule_gosc REAL,
        spalone_gosp REAL,
        spalone_gosc REAL,
        zk_gosp REAL,
        zk_gosc REAL,
        czk_gosp REAL,
        czk_gosc REAL,
        druga_zk_gosp REAL,
        druga_zk_gosc REAL,
        dosrodkowania_gosp REAL,
        dosrodkowania_gosc REAL,
        dosrodkowania_celne_gosp REAL,
        dosrodkowania_celne_gosc REAL,
        odbiory_gosp REAL,
        odbiory_gosc REAL,
        podania_gosp REAL,
        podania_gosc REAL,
        podania_celne_gosp REAL,
        podania_celne_gosc REAL,

        xg_gosp REAL,
        xg_gosc REAL,

        flash_id TEXT,
        flash_url TEXT,

        PRIMARY KEY (sezon, match_id)
    )
    """)

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_kolejka ON {TABLE_NAME}(kolejka)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_data_meczu ON {TABLE_NAME}(data_meczu)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_gospodarz ON {TABLE_NAME}(gospodarz)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_gosc ON {TABLE_NAME}(gosc)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_sezon_kolejka ON {TABLE_NAME}(sezon, kolejka)")


def insert_data(conn: sqlite3.Connection, df: pd.DataFrame):
    placeholders = ", ".join(["?"] * len(REQUIRED_COLUMNS))
    columns_sql = ", ".join(REQUIRED_COLUMNS)

    sql = f"""
    INSERT INTO {TABLE_NAME} ({columns_sql})
    VALUES ({placeholders})
    """

    rows = [tuple(row[col] for col in REQUIRED_COLUMNS) for _, row in df.iterrows()]
    conn.executemany(sql, rows)
    conn.commit()


def print_summary(conn: sqlite3.Connection):
    print("\n" + "=" * 100)
    print("PODSUMOWANIE BAZY")
    print("=" * 100)

    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"Liczba meczów w tabeli '{TABLE_NAME}': {total}")

    print("\nMecze per sezon:")
    rows = conn.execute(f"""
        SELECT sezon, COUNT(*) AS liczba
        FROM {TABLE_NAME}
        GROUP BY sezon
        ORDER BY sezon
    """).fetchall()
    for sezon, liczba in rows:
        print(f" - {sezon}: {liczba}")

    print("\nDostępność xG per sezon:")
    rows = conn.execute(f"""
        SELECT
            sezon,
            COUNT(*) AS wszystkie_mecze,
            SUM(CASE WHEN xg_gosp IS NOT NULL AND xg_gosc IS NOT NULL THEN 1 ELSE 0 END) AS mecze_z_xg,
            SUM(CASE WHEN xg_gosp IS NULL OR xg_gosc IS NULL THEN 1 ELSE 0 END) AS mecze_bez_xg
        FROM {TABLE_NAME}
        GROUP BY sezon
        ORDER BY sezon
    """).fetchall()
    for sezon, wszystkie, z_xg, bez_xg in rows:
        print(f" - {sezon}: z xG = {z_xg}/{wszystkie}, bez xG = {bez_xg}")

    print("\nPierwsze 5 meczów:")
    rows = conn.execute(f"""
        SELECT sezon, kolejka, gospodarz, gosc, gole_gosp, gole_gosc, xg_gosp, xg_gosc
        FROM {TABLE_NAME}
        ORDER BY sezon, kolejka, match_id
        LIMIT 5
    """).fetchall()
    for row in rows:
        print(" -", row)


def main():
    print("=" * 100)
    print("BUDOWA BAZY SQLITE")
    print("=" * 100)
    print(f"CSV wejściowy : {CSV_PATH}")
    print(f"Baza wyjściowa: {DB_PATH}")
    print(f"Tabela        : {TABLE_NAME}")
    print()

    if not CSV_PATH.exists():
        print(f"❌ Nie znaleziono pliku CSV:\n{CSV_PATH}")
        print("Najpierw uruchom: python scripts/cleaning/build_mvp_z_xg.py")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Wczytywanie CSV...")
    df = read_csv_flexible(CSV_PATH)
    print(f"   Wczytano: {len(df)} wierszy, {len(df.columns)} kolumn")

    print("\n2. Walidacja kolumn...")
    validate_columns(df)

    print("\n3. Czyszczenie danych...")
    df = clean_dataframe(df)
    print(f"   Dane po czyszczeniu: {len(df)} wierszy, {len(df.columns)} kolumn")

    print("\n4. Tworzenie bazy i tabeli...")
    conn = sqlite3.connect(DB_PATH)

    try:
        create_table(conn)
        print("   ✅ Tabela utworzona")

        print("\n5. Wstawianie danych do SQLite...")
        insert_data(conn, df)
        print(f"   ✅ Wstawiono {len(df)} rekordów")

        print_summary(conn)

    finally:
        conn.close()

    print("\n✅ Gotowe.")
    print(f"Baza zapisana w: {DB_PATH}")


if __name__ == "__main__":
    main()