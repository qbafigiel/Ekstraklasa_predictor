from __future__ import annotations

import re
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "db" / "ekstraklasa.db"
OUTPUT_CSV_PATH = ROOT / "data" / "processed" / "referee_dictionary.csv"
REPORT_PATH = ROOT / "data" / "reports" / "model" / "referee_full_names_report.txt"


REFEREE_FULL_NAMES = {
    "Marciniak S.": "Szymon Marciniak",
    "Lasyk P.": "Piotr Lasyk",
    "Raczkowski P.": "Pawel Raczkowski",
    "Przybyl J.": "Jaroslaw Przybyl",
    "Sylwestrzak D.": "Damian Sylwestrzak",
    "Arys K.": "Karol Arys",
    "Kuzma L.": "Lukasz Kuzma",
    "Kwiatkowski T.": "Tomasz Kwiatkowski",
    "Myc W.": "Wojciech Myc",
    "Frankowski B.": "Bartosz Frankowski",
    "Gryckiewicz P.": "Patryk Gryckiewicz",
    "Stefanski D.": "Daniel Stefanski",
    "Kos D.": "Damian Kos",
    "Malec P.": "Pawel Malec",
    "Szczerbowicz M.": "Marcin Szczerbowicz",
    "Musia T.": "Tomasz Musial",       # w bazie "Musiał" → "Musia" po ASCII strip
    "Kochanek M.": "Marcin Kochanek",
    "Krasny S.": "Sebastian Krasny",
    "Jakubik K.": "Krzysztof Jakubik",
    "Rzucidlo P.": "Piotr Rzucidlo",
    "Piszczelok M.": "Mateusz Piszczelok",
    "Nagamine K.": "Koki Nagamine",
    "Marciniak T.": "Tomasz Marciniak",
    "Araki Y.": "Yusuke Araki",
    "Karski L.": "Lukasz Karski",
    "Mucha S.": "Sebastian Mucha",
    "Malyszek J.": "Jacek Malyszek",
    "Listkiewicz T.": "Tomasz Listkiewicz",
    "Kawalko G.": "Grzegorz Kawalko",
    "Jarzebak S.": "Sebastian Jarzebak",
    "Al-Emara M.": "Mohammad Al-Emara",
}


def ascii_normalize(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_parent_dirs() -> None:
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def add_referee_full_name_column_if_missing(conn: sqlite3.Connection) -> bool:
    if column_exists(conn, "match_referees", "referee_full_name"):
        return False
    conn.execute("ALTER TABLE match_referees ADD COLUMN referee_full_name TEXT")
    return True


def load_distinct_referees(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT
            TRIM(referee_name) AS referee_name,
            COUNT(*) AS matches_count
        FROM match_referees
        WHERE referee_name IS NOT NULL
          AND TRIM(referee_name) <> ''
        GROUP BY TRIM(referee_name)
        ORDER BY matches_count DESC, referee_name
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        raise RuntimeError("Tabela match_referees nie zawiera żadnych nazw sędziów.")
    return df


def validate_mapping(df_referees: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    df = df_referees.copy()
    df["referee_key"] = df["referee_name"].apply(ascii_normalize)

    dup = df.groupby("referee_key")["referee_name"].nunique()
    dup = dup[dup > 1]
    if not dup.empty:
        conflict_rows = (
            df[df["referee_key"].isin(dup.index)]
            .sort_values(["referee_key", "referee_name"])
            .to_string(index=False)
        )
        raise RuntimeError(
            "Wykryto konflikt nazw sędziów po normalizacji ASCII.\n"
            "To trzeba najpierw ręcznie rozstrzygnąć.\n\n"
            f"{conflict_rows}"
        )

    db_keys = set(df["referee_key"])
    map_keys = set(REFEREE_FULL_NAMES.keys())

    missing_in_mapping = sorted(db_keys - map_keys)
    extra_in_mapping = sorted(map_keys - db_keys)

    return missing_in_mapping, extra_in_mapping, df


def update_database(conn: sqlite3.Connection, df_referees: pd.DataFrame) -> int:
    rows = []
    for _, row in df_referees.iterrows():
        referee_name_original = row["referee_name"]
        referee_key = row["referee_key"]
        referee_full_name = REFEREE_FULL_NAMES[referee_key]
        rows.append((referee_full_name, referee_name_original))

    conn.executemany(
        """
        UPDATE match_referees
        SET referee_full_name = ?
        WHERE TRIM(referee_name) = ?
        """,
        rows,
    )

    filled_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM match_referees
        WHERE referee_full_name IS NOT NULL
          AND TRIM(referee_full_name) <> ''
        """
    ).fetchone()[0]

    named_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM match_referees
        WHERE referee_name IS NOT NULL
          AND TRIM(referee_name) <> ''
        """
    ).fetchone()[0]

    if filled_count != named_count:
        raise RuntimeError(
            f"Po update niepełne pokrycie referee_full_name: "
            f"{filled_count}/{named_count} rekordów."
        )

    return filled_count


def export_dictionary_csv(df_referees: pd.DataFrame) -> pd.DataFrame:
    df_out = df_referees.copy()
    df_out["referee_full_name"] = df_out["referee_key"].map(REFEREE_FULL_NAMES)
    df_out = df_out[["referee_name", "referee_key", "referee_full_name", "matches_count"]]
    df_out = df_out.sort_values(["matches_count", "referee_name"], ascending=[False, True]).reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    return df_out


def build_report(
    column_added: bool,
    updated_rows: int,
    df_export: pd.DataFrame,
    extra_in_mapping: list[str],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("ENRICH REFEREE FULL NAMES REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {now}")
    lines.append(f"DB: {DB_PATH}")
    lines.append("")
    lines.append("Zalozenia:")
    lines.append("- referee_full_name zapisany w ASCII (bez polskich znakow)")
    lines.append("- slownik short_name -> full_name oparty na danych zweryfikowanych przez uzytkownika")
    lines.append("- klucze slownika to ASCII-normalize(referee_name z bazy)")
    lines.append("")
    lines.append(f"Dodano nowa kolumne referee_full_name: {'TAK' if column_added else 'NIE (juz istniala)'}")
    lines.append(f"Uzupelnione rekordy referee_full_name: {updated_rows}")
    lines.append(f"Liczba unikalnych sedziow: {len(df_export)}")
    lines.append("")

    if extra_in_mapping:
        lines.append("Pozycje obecne w slowniku, ale nieuzyte w aktualnej bazie:")
        for item in extra_in_mapping:
            lines.append(f"  - {item}")
        lines.append("")
    else:
        lines.append("Brak nadmiarowych pozycji w slowniku wzgledem aktualnej bazy.")
        lines.append("")

    lines.append("Slownik koncowy (referee_name -> referee_full_name):")
    for _, row in df_export.iterrows():
        lines.append(
            f"  {row['referee_name']:<20} -> {row['referee_full_name']:<25} "
            f"(key={row['referee_key']:<20}, matches={row['matches_count']})"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_parent_dirs()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono bazy: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        df_referees = load_distinct_referees(conn)
        missing_in_mapping, extra_in_mapping, df_validated = validate_mapping(df_referees)

        if missing_in_mapping:
            missing_text = "\n".join(f"  - {x}" for x in missing_in_mapping)
            raise RuntimeError(
                "Brakuje mapowania dla czesci sedziow obecnych w bazie.\n"
                "Skrypt przerwany bez zapisywania.\n\n"
                f"{missing_text}"
            )

        column_added = add_referee_full_name_column_if_missing(conn)
        updated_rows = update_database(conn, df_validated)
        conn.commit()

        df_export = export_dictionary_csv(df_validated)
        build_report(
            column_added=column_added,
            updated_rows=updated_rows,
            df_export=df_export,
            extra_in_mapping=extra_in_mapping,
        )

        print("OK: referee_full_name uzupelnione w tabeli match_referees")
        print(f"Rekordy uzupelnione:    {updated_rows}")
        print(f"Unikalni sedziowie:     {len(df_export)}")
        print(f"CSV:    {OUTPUT_CSV_PATH}")
        print(f"Raport: {REPORT_PATH}")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()