# scripts/audit/review_all_manual_decisions.py
"""
Generuje CSV z decyzjami wysokiego ryzyka do ręcznej weryfikacji.
Czyta player_mapping_status_{2023_24, 2024_25, 2025_26}.csv
Zapisuje: data/reports/player_identity/cross_review_all_seasons.csv

Ryzyko:
  HIGH  — matched_auto_transfer_candidate, matched_manual_review
  MED   — matched_auto_same_club_history gdy slug z innego sezonu
  LOW   — matched_existing (już zaakceptowane wcześniej)
  SKIP  — no_candidate, rejected_manual
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports/player_identity")

SEASON_FILES = {
    "2023/24": PROCESSED_DIR / "player_mapping_status_2023_24.csv",
    "2024/25": PROCESSED_DIR / "player_mapping_status_2024_25.csv",
    "2025/26": PROCESSED_DIR / "player_mapping_status_2025_26.csv",
}

SEASON_LABEL = {
    "2023/24": "2023_24",
    "2024/25": "2024_25",
    "2025/26": "2025_26",
}


def classify_risk(row, season):
    status = row["final_status"]

    # Pomiń — nie wymagają review
    if status in {"no_candidate", "rejected_manual"}:
        return None

    # WYSOKIE ryzyko — zawsze pokaż
    if status == "matched_auto_transfer_candidate":
        return "HIGH", "transfer_candidate — zawodnik w innym klubie"

    if status == "matched_manual_review":
        return "HIGH", "manual_review — subiektywna decyzja AI"

    # matched_existing — z poprzedniego mapowania, generalnie OK
    # ale pokażemy jako LOW żeby użytkownik mógł przejrzeć
    if status == "matched_existing":
        return "LOW", "existing_mapping — zaakceptowane wcześniej"

    # matched_auto_same_club_history — sprawdź czy slug z target season
    if status == "matched_auto_same_club_history":
        season_club_path = str(row.get("identity_season_club_path", ""))
        target_label = SEASON_LABEL[season]

        # Sprawdź czy w season_club_path jest target sezon
        # Format: "2023/24:klub-slug | 2024/25:inny-klub"
        # Szukamy czy target season pojawia się w path
        season_in_path = False
        for segment in season_club_path.split("|"):
            segment = segment.strip()
            if segment.startswith(season + ":") or segment.startswith(season):
                season_in_path = True
                break

        if not season_in_path:
            # Slug pochodzi z innego sezonu — ryzyko pułapki Kucharczyka!
            return "MED", f"same_club_history ale slug z innego sezonu (path: {season_club_path[:80]})"
        else:
            # Slug jest z target season w tym samym klubie — bezpieczne
            return None  # pomiń

    return None


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for season, path in SEASON_FILES.items():
        if not path.exists():
            print(f"BRAK: {path}")
            continue

        df = pd.read_csv(path)
        print(f"\n{season}: {len(df)} wierszy")

        for _, row in df.iterrows():
            result = classify_risk(row, season)
            if result is None:
                continue

            risk_level, evidence = result

            all_rows.append({
                "season": season,
                "flash_team": row.get("flash_team", ""),
                "flash_name": row.get("flash_name", ""),
                "final_status": row.get("final_status", ""),
                "player_slug": row.get("player_slug", ""),
                "identity_season_club_path": row.get("identity_season_club_path", ""),
                "identity_clubs_all": row.get("identity_clubs_all", ""),
                "identity_seasons_all": row.get("identity_seasons_all", ""),
                "identity_in_target_club": row.get("identity_in_target_club", ""),
                "match_method": row.get("match_method", ""),
                "risk_level": risk_level,
                "evidence": evidence,
                "user_decision": "",  # do wypełnienia: OK / REJECT / RE-MAP:nowy-slug
            })

    if not all_rows:
        print("Brak wierszy do review.")
        return

    out = pd.DataFrame(all_rows)

    # Sortowanie: najpierw HIGH, potem MED, potem LOW; w ramach — sezon, team, name
    risk_order = {"HIGH": 0, "MED": 1, "LOW": 2}
    out["_risk_sort"] = out["risk_level"].map(risk_order)
    out = out.sort_values(["_risk_sort", "season", "flash_team", "flash_name"])
    out = out.drop(columns=["_risk_sort"])

    output_path = REPORT_DIR / "cross_review_all_seasons.csv"
    out.to_csv(output_path, index=False, encoding="utf-8-sig")  # utf-8-sig dla Excel

    # Podsumowanie
    print("\n" + "=" * 70)
    print("CROSS-REVIEW — PODSUMOWANIE")
    print("=" * 70)
    for season in ["2023/24", "2024/25", "2025/26"]:
        s = out[out["season"] == season]
        if len(s) == 0:
            continue
        print(f"\n{season}:")
        for risk in ["HIGH", "MED", "LOW"]:
            n = (s["risk_level"] == risk).sum()
            if n:
                print(f"  {risk:4s}: {n:3d} wierszy")

    print(f"\nŁącznie do review: {len(out)}")
    print(f"\nZapisano: {output_path}")
    print("\nKolumna 'user_decision' — wpisz:")
    print("  OK       — potwierdzam, zostaje")
    print("  REJECT   — odrzucam, trafi do no_candidate")
    print("  RE-MAP:nowy-slug — zmapuj na inny slug")


if __name__ == "__main__":
    main()