from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"D:\projects\Ekstraklasa_predictor\data")

PLIKI = [
    # API
    "mecze_2023_24.csv",
    "mecze_2024_25.csv",
    "mecze_2025_26.csv",
    # Flashscore
    "flash_2023_24.csv",
    "flash_2024_25.csv",
    "flash_2025_26_druzyny.csv",
]

SAVE_REPORT = True
REPORT_FILE = DATA_DIR / "audyt_brakow_raport.csv"


def read_csv_flexible(path: Path):
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False), enc
        except Exception:
            continue
    return None, None


def analyze_file(path: Path):
    df, enc = read_csv_flexible(path)

    if df is None:
        print(f"❌ Nie udało się odczytać: {path.name}")
        return None

    rows = []

    for col in df.columns:
        total = len(df)
        missing = int(df[col].isna().sum())
        filled = total - missing
        filled_pct = round((filled / total) * 100, 1) if total > 0 else 0.0
        missing_pct = round((missing / total) * 100, 1) if total > 0 else 0.0
        nunique = int(df[col].nunique(dropna=True))
        dtype = str(df[col].dtype)

        # przykładowe wartości (max 3)
        sample_vals = df[col].dropna().astype(str).unique()[:3]
        sample_text = " | ".join(sample_vals)
        if len(sample_text) > 80:
            sample_text = sample_text[:77] + "..."

        rows.append({
            "plik": path.name,
            "kolumna": col,
            "dtype": dtype,
            "wiersze": total,
            "wypelnione": filled,
            "wypelnione_pct": filled_pct,
            "braki": missing,
            "braki_pct": missing_pct,
            "unikalne": nunique,
            "przyklad": sample_text,
        })

    return pd.DataFrame(rows)


def print_file_summary(df_report, filename):
    file_data = df_report[df_report["plik"] == filename].copy()

    if file_data.empty:
        return

    total_rows = file_data["wiersze"].iloc[0]
    total_cols = len(file_data)
    perfect_cols = len(file_data[file_data["braki"] == 0])
    problem_cols = file_data[file_data["braki"] > 0].sort_values("braki_pct", ascending=False)

    print(f"\n{'=' * 100}")
    print(f"📄 {filename}")
    print(f"{'=' * 100}")
    print(f"   Wierszy (meczów) : {total_rows}")
    print(f"   Kolumn           : {total_cols}")
    print(f"   Kompletnych      : {perfect_cols} / {total_cols}")

    if problem_cols.empty:
        print(f"   ✅ ZERO BRAKÓW — wszystkie kolumny kompletne")
    else:
        print(f"   ⚠️ Kolumny z brakami: {len(problem_cols)}")
        print()
        print(f"   {'Kolumna':<45} {'Braki':>8} {'Braki %':>10} {'Wypełnione %':>14}")
        print(f"   {'-' * 45} {'-' * 8} {'-' * 10} {'-' * 14}")

        for _, row in problem_cols.iterrows():
            col_name = row["kolumna"]
            braki = int(row["braki"])
            braki_pct = row["braki_pct"]
            filled_pct = row["wypelnione_pct"]

            # emotki statusu
            if filled_pct >= 95:
                status = "🟢"
            elif filled_pct >= 70:
                status = "🟡"
            elif filled_pct >= 50:
                status = "🟠"
            else:
                status = "🔴"

            print(f"   {status} {col_name:<43} {braki:>8} {braki_pct:>9.1f}% {filled_pct:>13.1f}%")


def print_cross_file_comparison(df_report):
    print(f"\n{'=' * 100}")
    print("📊 PORÓWNANIE MIĘDZY PLIKAMI — PODSUMOWANIE")
    print(f"{'=' * 100}")

    summary_rows = []

    for filename in df_report["plik"].unique():
        file_data = df_report[df_report["plik"] == filename]
        total_rows = file_data["wiersze"].iloc[0]
        total_cols = len(file_data)
        perfect = len(file_data[file_data["braki"] == 0])
        worst_col = file_data.sort_values("braki_pct", ascending=False).iloc[0]

        summary_rows.append({
            "plik": filename,
            "meczów": total_rows,
            "kolumn": total_cols,
            "kompletnych": perfect,
            "z_brakami": total_cols - perfect,
            "najgorsza_kolumna": worst_col["kolumna"],
            "najgorszy_brak_pct": worst_col["braki_pct"],
        })

    summary_df = pd.DataFrame(summary_rows)

    print()
    print(f"{'Plik':<35} {'Meczów':>7} {'Kolumn':>7} {'OK':>5} {'Braki':>6} {'Najgorsza kolumna':<35} {'Brak %':>8}")
    print(f"{'-' * 35} {'-' * 7} {'-' * 7} {'-' * 5} {'-' * 6} {'-' * 35} {'-' * 8}")

    for _, row in summary_df.iterrows():
        print(
            f"{row['plik']:<35} "
            f"{int(row['meczów']):>7} "
            f"{int(row['kolumn']):>7} "
            f"{int(row['kompletnych']):>5} "
            f"{int(row['z_brakami']):>6} "
            f"{row['najgorsza_kolumna']:<35} "
            f"{row['najgorszy_brak_pct']:>7.1f}%"
        )


def print_flash_usability_verdict(df_report):
    print(f"\n{'=' * 100}")
    print("🎯 WERDYKT — UŻYTECZNOŚĆ KOLUMN FLASHSCORE DLA MODELU")
    print(f"{'=' * 100}")

    flash_files = [f for f in df_report["plik"].unique() if "flash" in f.lower()]

    if not flash_files:
        print("Brak plików Flashscore w audycie.")
        return

    # zbierz wszystkie unikalne kolumny Flash
    all_flash_cols = set()
    for f in flash_files:
        cols = df_report[df_report["plik"] == f]["kolumna"].tolist()
        all_flash_cols.update(cols)

    # dla każdej kolumny, pokaż pokrycie w każdym pliku
    col_summary = []

    for col in sorted(all_flash_cols):
        row_data = {"kolumna": col}

        for f in sorted(flash_files):
            file_data = df_report[(df_report["plik"] == f) & (df_report["kolumna"] == col)]
            if file_data.empty:
                row_data[f] = "BRAK"
            else:
                row_data[f] = f"{file_data.iloc[0]['wypelnione_pct']:.0f}%"

        col_summary.append(row_data)

    col_df = pd.DataFrame(col_summary)

    # klasyfikacja
    print()
    print("Legenda:")
    print("  🟢 >= 95% we wszystkich plikach Flash → KANDYDAT DO MODELU BAZOWEGO")
    print("  🟡 >= 70% we wszystkich → OPCJONALNY / KOREKCYJNY")
    print("  🟠 >= 50% w przynajmniej jednym → RYZYKOWNY")
    print("  🔴 < 50% gdziekolwiek lub BRAK → NIE UŻYWAĆ W MODELU BAZOWYM")
    print()

    for _, row in col_df.iterrows():
        col_name = row["kolumna"]
        values = []

        for f in sorted(flash_files):
            val = row.get(f, "BRAK")
            values.append(val)

        # ocena
        numeric_vals = []
        has_missing_file = False

        for v in values:
            if v == "BRAK":
                has_missing_file = True
            else:
                try:
                    numeric_vals.append(float(v.replace("%", "")))
                except ValueError:
                    has_missing_file = True

        if has_missing_file or not numeric_vals:
            status = "🔴"
        elif min(numeric_vals) >= 95:
            status = "🟢"
        elif min(numeric_vals) >= 70:
            status = "🟡"
        elif min(numeric_vals) >= 50:
            status = "🟠"
        else:
            status = "🔴"

        vals_str = "  ".join([f"{f}: {row.get(f, 'BRAK'):>5}" for f in sorted(flash_files)])
        print(f"  {status} {col_name:<45} {vals_str}")


def main():
    print("=" * 100)
    print("AUDYT BRAKÓW DANYCH — WSZYSTKIE PLIKI ŹRÓDŁOWE")
    print("=" * 100)

    all_reports = []

    for filename in PLIKI:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"\n⚠️ Plik nie istnieje: {filename}")
            continue

        report = analyze_file(path)

        if report is not None:
            all_reports.append(report)

    if not all_reports:
        print("\n❌ Nie udało się przeanalizować żadnego pliku.")
        return

    df_report = pd.concat(all_reports, ignore_index=True)

    # szczegóły per plik
    for filename in PLIKI:
        if filename in df_report["plik"].values:
            print_file_summary(df_report, filename)

    # porównanie między plikami
    print_cross_file_comparison(df_report)

    # werdykt Flash
    print_flash_usability_verdict(df_report)

    # zapis
    if SAVE_REPORT:
        df_report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n📁 Raport zapisany: {REPORT_FILE}")

    print("\n✅ Audyt zakończony.")


if __name__ == "__main__":
    main()