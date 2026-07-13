import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "db/ekstraklasa.db"
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "match_lineup_values.csv"

def load_priors():
    # Wczytujemy 2 sezony priorów, bo dla 2023/24 nie mamy historii z 2022/23
    df_24_25 = pd.read_csv(PROCESSED_DIR / "player_priors_2024_25.csv")
    df_24_25["target_season"] = "2024/25"
    
    df_25_26 = pd.read_csv(PROCESSED_DIR / "player_priors_2025_26.csv")
    df_25_26["target_season"] = "2025/26"
    
    combined = pd.concat([df_24_25, df_25_26], ignore_index=True)
    return combined

def main():
    print("1. Wczytuję priory z poprzednich sezonów...")
    try:
        priors = load_priors()
    except FileNotFoundError as e:
        print(f"BŁĄD: {e}")
        return

    # Wypełniamy puste wartości zerami (jeśli ktoś nie grał lub był rezerwowym)
    metrics = [
        "prior_pole_xg", 
        "prior_pole_xa", 
        "prior_pole_przechwyty", 
        "prior_pole_odbiory", 
        "prior_pole_pojedynki-obronne-wygrane",
        "prior_pole_minuty",
        "prior_gk_gk-minuty-rozegrane",
        "prior_gk_czyste-konta"
    ]
    for col in metrics:
        if col in priors.columns:
            priors[col] = pd.to_numeric(priors[col], errors="coerce").fillna(0.0)

    # Budujemy pule zawodnika
    priors["player_offense"] = priors["prior_pole_xg"] + priors["prior_pole_xa"]
    priors["player_defense"] = priors["prior_pole_przechwyty"] + priors["prior_pole_odbiory"] + priors["prior_pole_pojedynki-obronne-wygrane"]
    priors["player_minutes"] = priors["prior_pole_minuty"] + priors["prior_gk_gk-minuty-rozegrane"]
    priors["player_gk_cs"] = priors["prior_gk_czyste-konta"]

    # Zostawiamy tylko to co potrzebne do joina
    priors_clean = priors[[
        "target_season", "flash_team", "flash_name", "prior_available", 
        "player_offense", "player_defense", "player_minutes", "player_gk_cs"
    ]].copy()

    print("2. Wczytuję pierwsze jedenastki z bazy (tylko is_starter=1)...")
    conn = sqlite3.connect(DB_PATH)
    lineups = pd.read_sql_query(
        "SELECT sezon, match_id, team_side, team_name, player_name, is_goalkeeper FROM lineups WHERE is_starter = 1", 
        conn
    )
    conn.close()

    print("3. Łączę składy z wartościami zawodników...")
    # Join: po sezonie docelowym, drużynie i nazwisku
    merged = pd.merge(
        lineups, 
        priors_clean,
        left_on=["sezon", "team_name", "player_name"],
        right_on=["target_season", "flash_team", "flash_name"],
        how="left"
    )

    # Jeśli brakuje priora, uznajemy go za 0 (nowy gracz w lidze nie wnosi "sprawdzonego" doświadczenia ekstraklasowego)
    merged["prior_available"] = merged["prior_available"].fillna(0)
    for col in ["player_offense", "player_defense", "player_minutes", "player_gk_cs"]:
        merged[col] = merged[col].fillna(0.0)

    print("4. Agreguję wartości na poziom meczu i drużyny...")
    # Agregacja do poziomu (match_id, team_side)
    match_team_stats = merged.groupby(["match_id", "team_side", "sezon"]).agg(
        lineup_offense=("player_offense", "sum"),
        lineup_defense=("player_defense", "sum"),
        lineup_minutes=("player_minutes", "sum"),
        lineup_gk_cs=("player_gk_cs", "sum"),
        starters_with_prior=("prior_available", "sum")
    ).reset_index()

    print("5. Pivotowanie z 2 wierszy na mecz do 1 wiersza na mecz...")
    # Chcemy mieć po 1 wierszu na mecz, kolumny z prefiksami home_ / away_
    home = match_team_stats[match_team_stats["team_side"] == "home"].copy()
    home = home.add_prefix("home_")
    home = home.rename(columns={"home_match_id": "match_id", "home_sezon": "sezon"}).drop(columns=["home_team_side"])

    away = match_team_stats[match_team_stats["team_side"] == "away"].copy()
    away = away.add_prefix("away_")
    away = away.rename(columns={"away_match_id": "match_id", "away_sezon": "sezon"}).drop(columns=["away_team_side", "sezon"])

    match_priors = pd.merge(home, away, on="match_id", how="inner")

    # Obliczmy różnice (Home minus Away) - idealne do modelowania
    match_priors["diff_lineup_offense"] = match_priors["home_lineup_offense"] - match_priors["away_lineup_offense"]
    match_priors["diff_lineup_defense"] = match_priors["home_lineup_defense"] - match_priors["away_lineup_defense"]
    match_priors["diff_lineup_minutes"] = match_priors["home_lineup_minutes"] - match_priors["away_lineup_minutes"]
    
    match_priors = match_priors.sort_values("match_id").round(2)
    match_priors.to_csv(OUTPUT_PATH, index=False)

    print("\nGotowe! Próbka wygenerowanych danych:")
    print(match_priors.head(5)[["sezon", "match_id", "diff_lineup_offense", "diff_lineup_defense", "diff_lineup_minutes"]].to_string(index=False))
    print(f"\nZapisano plik: {OUTPUT_PATH}")
    print(f"Ilość meczów: {len(match_priors)}")

if __name__ == "__main__":
    main()