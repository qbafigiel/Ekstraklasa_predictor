"""
recommend_value_bets.py
=======================
Rekomenduje najbezpieczniejsze typy z predykcji kolejki.

Filtry:
1. Bierze tylko rynki które w K1+K2 miały >=75% trafności
2. Wybiera linie gdzie model daje >=60% pewności (żeby mieć poduszkę)
3. Sortuje po pewności modelu
4. Grupuje per mecz

Rekomendacje są tylko dla meczów status='scheduled' (bez postponed).

Użycie:
python scripts/audit/recommend_value_bets.py --sezon 2026/27 --kolejka 3
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

DB_PATH = Path("db/ekstraklasa.db")
PRED_DIR = Path("data/processed/predictions")

# Rynki które w K1+K2 miały >=75% trafności zbiorczą
RELIABLE_MARKETS = {
    "spalone": {
        "mu_col": "mu_offsides",
        "prefix": "offsides",
        "label": "Spalone",
        "actual_h": "spalone_gosp",
        "actual_a": "spalone_gosc",
    },
    "yc": {
        "mu_col": "mu_yc",
        "prefix": "yc",
        "label": "Żółte kartki",
        "actual_h": "zk_gosp",
        "actual_a": "zk_gosc",
    },
    "sot": {
        "mu_col": "mu_sot",
        "prefix": "sot",
        "label": "Strzały celne",
        "actual_h": "celne_gosp",
        "actual_a": "celne_gosc",
    },
    "shots": {
        "mu_col": "mu_shots",
        "prefix": "shots",
        "label": "Strzały",
        "actual_h": "strzaly_gosp",
        "actual_a": "strzaly_gosc",
    },
    "corners": {
        "mu_col": "mu_corners",
        "prefix": "corners",
        "label": "Kornery",
        "actual_h": "rozne_gosp",
        "actual_a": "rozne_gosc",
    },
}

# Konkretne linie które w K1+K2 miały >=75% trafności
# (dane z podsumowania K1+K2 z 05.08)
RELIABLE_LINES = {
    "corners": [4.5, 5.5, 6.5, 7.5, 9.5],  # 82-100%
    "shots": [12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 35.5],
    "sot": [3.5, 4.5, 5.5, 6.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
    "offsides": [0.5, 1.5, 3.5, 4.5, 5.5, 6.5],
    "yc": [0.5, 1.5, 2.5, 4.5, 5.5, 6.5, 7.5],
    # UWAGA: faule celowo pomijamy - centralne linie (22-26) miały <50%, tylko skrajne działały
}

# Minimalne prawdopodobieństwo modelu dla rekomendacji (żeby mieć poduszkę)
MIN_MODEL_CONFIDENCE = 0.60


def load_predictions(sezon: str, kolejka: int) -> pd.DataFrame:
    season_part = sezon.replace("/", "-")
    path = PRED_DIR / f"predict_round_{season_part}_K{kolejka:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Brak predykcji: {path}")
    return pd.read_csv(path)


def get_scheduled_fixtures(sezon: str, kolejka: int) -> pd.DataFrame:
    """Bierze tylko mecze status='scheduled' (nie postponed)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT gospodarz, gosc, status, referee_full_name
        FROM fixtures_upcoming
        WHERE sezon=? AND kolejka=? AND status='scheduled'
    """, conn, params=(sezon, kolejka))
    conn.close()
    return df


# Zakres wokol mu gdzie bukmacher stawia linie z sensownymi kursami (1.6-2.2)
# Dla kazdego rynku inny bo skale roznia sie
MU_RANGE = {
    "corners": 2.0,   # +/- 2 kornery
    "shots": 3.0,     # +/- 3 strzaly
    "sot": 1.5,       # +/- 1.5 strzalu celnego
    "offsides": 1.0,  # +/- 1 spalonego
    "yc": 1.0,        # +/- 1 ZK
}

# Min pewnosc modelu dla rekomendacji na CENTRALNYCH liniach (gdzie kurs jest sensowny)
MIN_CENTRAL_CONFIDENCE = 0.58  # bo tam kursy sa ~1.7-1.9, wiec 58%+ jest value


def analyze_match(pred_row: pd.Series, sedzia: str) -> List[Dict]:
    """Analizuje mecz i zwraca tylko linie CENTRALNE (blisko mu) z sensownymi kursami."""
    picks = []
    
    home = pred_row["gospodarz"]
    away = pred_row["gosc"]
    
    for market_key, cfg in RELIABLE_MARKETS.items():
        prefix = cfg["prefix"]
        label = cfg["label"]
        mu = pred_row.get(cfg["mu_col"])
        
        if pd.isna(mu):
            continue
        
        reliable_lines = RELIABLE_LINES.get(prefix, [])
        mu_range = MU_RANGE.get(prefix, 2.0)
        
        for line in reliable_lines:
            # Filtruj tylko linie w okolicy mu
            if abs(line - float(mu)) > mu_range:
                continue
            
            line_int = int(line)
            line_dec = int(round((line - line_int) * 10))
            col_over = f"{prefix}_p_over_{line_int}_{line_dec}"
            col_under = f"{prefix}_p_under_{line_int}_{line_dec}"
            
            p_over = pred_row.get(col_over)
            p_under = pred_row.get(col_under)
            
            if pd.isna(p_over) or pd.isna(p_under):
                continue
            
            # Wybierz stronę z większą pewnością
            if p_over > p_under:
                side = "OVER"
                p = p_over
            else:
                side = "UNDER"
                p = p_under
            
            if p < MIN_CENTRAL_CONFIDENCE:
                continue
            
            # Oszacuj "expected value" - im dalej od 50%, tym bardziej pewny typ
            # ale linie centralne (blisko mu) mają wyższe kursy więc są bardziej value
            distance_from_mu = abs(line - float(mu))
            
            picks.append({
                "mecz": f"{home} vs {away}",
                "sedzia": sedzia,
                "rynek": label,
                "linia": line,
                "typ": side,
                "prawdopodobienstwo": p,
                "mu_modelu": mu,
                "odleglosc_od_mu": distance_from_mu,
                "sila": p,
            })
    
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True)
    parser.add_argument("--kolejka", type=int, required=True)
    parser.add_argument("--top", type=int, default=15, help="Ile najlepszych pokazać")
    args = parser.parse_args()
    
    preds = load_predictions(args.sezon, args.kolejka)
    scheduled = get_scheduled_fixtures(args.sezon, args.kolejka)
    
    # Filtruj tylko scheduled (nie postponed)
    scheduled_set = set(
        (row["gospodarz"], row["gosc"]) for _, row in scheduled.iterrows()
    )
    preds_filtered = preds[
        preds.apply(lambda r: (r["gospodarz"], r["gosc"]) in scheduled_set, axis=1)
    ]
    
    # Zbierz picks
    all_picks = []
    for _, pred in preds_filtered.iterrows():
        sedzia = pred.get("referee_full_name", "?")
        if pd.isna(sedzia):
            sedzia = "?"
        picks = analyze_match(pred, sedzia)
        all_picks.extend(picks)
    
    print("=" * 100)
    print(f"REKOMENDOWANE TYPY VALUE - {args.sezon} K{args.kolejka:02d}")
    print("=" * 100)
    print()
    print(f"Meczów scheduled: {len(preds_filtered)}")
    print(f"Rynków przeanalizowanych: 5 (spalone, żółte kartki, strzały celne, strzały, kornery)")
    print(f"Rynek FAULE POMINIĘTY - w K1+K2 środkowe linie miały <50% trafności")
    print(f"Min. pewność modelu: {MIN_MODEL_CONFIDENCE*100:.0f}%")
    print()
    
    if not all_picks:
        print("Brak typów spełniających kryteria.")
        return
    
    # Sortuj po pewności modelu
    df = pd.DataFrame(all_picks).sort_values("prawdopodobienstwo", ascending=False).reset_index(drop=True)
    
    # ===== TOP N =====
    print("=" * 100)
    print(f"TOP {args.top} NAJBEZPIECZNIEJSZYCH TYPÓW")
    print("=" * 100)
    print(f"{'#':<3} {'Mecz':<40} {'Rynek':<15} {'Typ':<15} {'Pewność':<10} {'Sędzia'}")
    print("-" * 100)
    
    for i, r in df.head(args.top).iterrows():
        typ_str = f"{r['typ']} {r['linia']:.1f}"
        print(f"{i+1:<3} {r['mecz']:<40} {r['rynek']:<15} {typ_str:<15} {r['prawdopodobienstwo']*100:>6.1f}%   {r['sedzia']}")
    
    # ===== PER MECZ =====
    print()
    print("=" * 100)
    print("REKOMENDACJE PER MECZ (tylko typy >=65% pewności)")
    print("=" * 100)
    
    for mecz in df["mecz"].unique():
        match_picks = df[(df["mecz"] == mecz) & (df["prawdopodobienstwo"] >= 0.65)]
        if len(match_picks) == 0:
            continue
        
        sedzia = match_picks.iloc[0]["sedzia"]
        print(f"\n{mecz}")
        print(f"Sędzia: {sedzia}")
        print("-" * 80)
        
        for _, r in match_picks.iterrows():
            typ_str = f"{r['typ']} {r['linia']:.1f}"
            print(f"  {r['rynek']:<15} {typ_str:<15} {r['prawdopodobienstwo']*100:>6.1f}% (μ={r['mu_modelu']:.2f})")
    
    # ===== STATYSTYKI =====
    print()
    print("=" * 100)
    print("STATYSTYKI ZBIORCZE")
    print("=" * 100)
    print(f"Wszystkich rekomendacji: {len(df)}")
    print(f"Powyżej 80% pewności:    {len(df[df['prawdopodobienstwo'] >= 0.80])}")
    print(f"70-80% pewności:         {len(df[(df['prawdopodobienstwo'] >= 0.70) & (df['prawdopodobienstwo'] < 0.80)])}")
    print(f"60-70% pewności:         {len(df[(df['prawdopodobienstwo'] >= 0.60) & (df['prawdopodobienstwo'] < 0.70)])}")
    
    print(f"\nPer rynek:")
    for rynek in df["rynek"].unique():
        count = len(df[df["rynek"] == rynek])
        avg_p = df[df["rynek"] == rynek]["prawdopodobienstwo"].mean()
        print(f"  {rynek:<20} {count:>3} typów, śr. pewność {avg_p*100:.1f}%")


if __name__ == "__main__":
    main()