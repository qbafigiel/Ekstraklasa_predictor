from pathlib import Path
import pandas as pd

CSV_PATH = Path(r"D:\projects\Ekstraklasa_predictor\data\pelne_2025_26.csv")


def read_csv_flexible(path: Path):
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False), enc
        except Exception:
            continue
    raise RuntimeError("Nie udało się odczytać pliku CSV.")


def main():
    if not CSV_PATH.exists():
        print(f"❌ Nie znaleziono pliku:\n{CSV_PATH}")
        return

    df, enc = read_csv_flexible(CSV_PATH)

    print(f"Plik: {CSV_PATH}")
    print(f"Encoding: {enc}")
    print(f"Liczba kolumn: {len(df.columns)}\n")

    for i, col in enumerate(df.columns, start=1):
        print(f"{i:>3}. {col}")


if __name__ == "__main__":
    main()