from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "api_2023_24": ROOT / "data" / "raw" / "api" / "mecze_2023_24.csv",
    "api_2024_25": ROOT / "data" / "raw" / "api" / "mecze_2024_25.csv",
    "api_2025_26": ROOT / "data" / "raw" / "api" / "mecze_2025_26.csv",

    "flash_2023_24": ROOT / "data" / "raw" / "flash" / "flash_2023_24.csv",
    "flash_2024_25": ROOT / "data" / "raw" / "flash" / "flash_2024_25.csv",
    "flash_2025_26": ROOT / "data" / "raw" / "flash" / "flash_2025_26_druzyny.csv",

    "processed_czyste_2025_26": ROOT / "data" / "processed" / "czyste_2025_26.csv",
}


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1250"]
    seps = [",", ";", "\t"]

    best_df = None
    best_info = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if best_df is None or len(df.columns) > len(best_df.columns):
                    best_df = df
                    best_info = (enc, sep)
            except Exception:
                continue

    if best_df is None:
        raise RuntimeError(f"Nie udało się wczytać pliku: {path}")

    return best_df, best_info


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def analyze_file(name: str, path: Path):
    print_section(name)

    if not path.exists():
        print(f"BRAK PLIKU: {path}")
        return None

    df, info = read_csv_auto(path)
    encoding, sep = info

    print(f"Ścieżka: {path}")
    print(f"Encoding: {encoding}")
    print(f"Separator: {repr(sep)}")
    print(f"Wiersze: {len(df)}")
    print(f"Kolumny: {len(df.columns)}")

    print("\nLista kolumn:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:>3}. {col}")

    xg_cols = [c for c in df.columns if "xg" in c.lower()]
    print("\nKolumny zawierające 'xg':")
    if xg_cols:
        for col in xg_cols:
            missing = df[col].isna().sum()
            pct = round(missing / len(df) * 100, 2) if len(df) else 0
            print(f"- {col}: braki {missing}/{len(df)} ({pct}%)")
    else:
        print("- brak")

    return df


def compare_schemas(dfs: dict, names: list[str], title: str):
    print_section(title)

    existing = {name: dfs[name] for name in names if dfs.get(name) is not None}

    if len(existing) < 2:
        print("Za mało plików do porównania.")
        return

    base_name = list(existing.keys())[0]
    base_cols = set(existing[base_name].columns)

    for name, df in existing.items():
        cols = set(df.columns)

        missing_vs_base = sorted(base_cols - cols)
        extra_vs_base = sorted(cols - base_cols)

        print(f"\nPorównanie: {name} vs {base_name}")
        print(f"- Kolumny w {base_name}, których brakuje w {name}: {len(missing_vs_base)}")
        for col in missing_vs_base:
            print(f"  - {col}")

        print(f"- Kolumny w {name}, których nie ma w {base_name}: {len(extra_vs_base)}")
        for col in extra_vs_base:
            print(f"  + {col}")


def main():
    dfs = {}

    for name, path in FILES.items():
        try:
            dfs[name] = analyze_file(name, path)
        except Exception as e:
            print_section(name)
            print(f"BŁĄD: {e}")
            dfs[name] = None

    compare_schemas(
        dfs,
        ["api_2023_24", "api_2024_25", "api_2025_26"],
        "PORÓWNANIE SCHEMATÓW API"
    )

    compare_schemas(
        dfs,
        ["flash_2023_24", "flash_2024_25", "flash_2025_26"],
        "PORÓWNANIE SCHEMATÓW FLASHSCORE"
    )

    print_section("REKOMENDACJA TECHNICZNA - DO SPRAWDZENIA PO OUTPUTCIE")
    print(
        "Jeżeli API ma identyczną strukturę w 3 sezonach, a Flashscore różną, "
        "to finalny model MVP powinien dostać stały rdzeń API + tylko kontrolowane kolumny xG."
    )
    print(
        "Dla sezonu 2023/24 xG najlepiej zignorować całkowicie, nawet jeśli częściowo istnieje, "
        "żeby nie robić niespójnego i stronniczego zbioru."
    )


if __name__ == "__main__":
    main()