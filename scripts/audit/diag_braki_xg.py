from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

for sezon in ["2024_25", "2025_26"]:
    path = ROOT / "data" / "processed" / f"mvp_{sezon}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    braki = df[df["xg_gosp"].isna()].copy()

    print("\n" + "=" * 80)
    print(f"SEZON {sezon}")
    print("=" * 80)
    print(f"Braki xG: {len(braki)}/{len(df)}")

    if braki.empty:
        print("Brak braków.")
        continue

    print("\nTOP drużyny występujące w meczach bez xG:")
    teams = pd.concat([braki["gospodarz"], braki["gosc"]]).value_counts()
    print(teams.to_string())

    print("\nPrzykładowe mecze bez xG:")
    print(braki[["kolejka", "gospodarz", "gosc"]].head(30).to_string(index=False))