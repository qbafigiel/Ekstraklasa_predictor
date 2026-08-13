"""
evaluate_predictions.py
=======================
Porównuje predykcje kolejki z rzeczywistymi wynikami.

Dla każdego meczu:
- 1X2: która była nasza predykcja (najwyższe p), czy trafiła
- BTTS: predykcja vs rzeczywistość
- Statystyki: mu vs rzeczywista wartość
- Rynki over/under: dla każdej linii - czy nasza predykcja by trafila

Użycie:
python scripts/audit/evaluate_predictions.py --sezon 2026/27 --kolejka 1
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DB_PATH = Path("db/ekstraklasa.db")
PRED_DIR = Path("data/processed/predictions")


def load_predictions(sezon: str, kolejka: int) -> pd.DataFrame:
    season_part = sezon.replace("/", "-")
    path = PRED_DIR / f"predict_round_{season_part}_K{kolejka:02d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Brak predykcji: {path}")
    return pd.read_csv(path)


def load_actuals(sezon: str, kolejka: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM matches WHERE sezon=? AND kolejka=?
    """, conn, params=(sezon, kolejka))
    conn.close()
    return df


def get_1x2(gh: int, ga: int) -> str:
    if gh > ga: return "H"
    if gh == ga: return "D"
    return "A"


def eval_1x2(pred_row: pd.Series, actual_row: pd.Series) -> Dict:
    p_H = pred_row["p_H"]
    p_D = pred_row["p_D"]
    p_A = pred_row["p_A"]
    
    probs = {"H": p_H, "D": p_D, "A": p_A}
    predicted = max(probs, key=probs.get)
    actual = get_1x2(int(actual_row["gole_gosp"]), int(actual_row["gole_gosc"]))
    
    return {
        "pred_1x2": predicted,
        "pred_p": probs[predicted],
        "actual_1x2": actual,
        "correct": predicted == actual,
    }


def eval_btts(pred_row: pd.Series, actual_row: pd.Series) -> Dict:
    p_yes = pred_row["p_btts_yes"]
    predicted_yes = p_yes > 0.5
    actual_yes = actual_row["gole_gosp"] > 0 and actual_row["gole_gosc"] > 0
    
    return {
        "pred_btts": "YES" if predicted_yes else "NO",
        "pred_p": p_yes if predicted_yes else (1 - p_yes),
        "actual_btts": "YES" if actual_yes else "NO",
        "correct": predicted_yes == actual_yes,
    }


def eval_goals_over(pred_row: pd.Series, actual_row: pd.Series) -> List[Dict]:
    total_gole = int(actual_row["gole_gosp"]) + int(actual_row["gole_gosc"])
    results = []
    
    for line, col in [(0.5, "p_over_05"), (1.5, "p_over_15"), (2.5, "p_over_25"), (3.5, "p_over_35")]:
        p = pred_row[col]
        pred_over = p > 0.5
        actual_over = total_gole > line
        results.append({
            "line": line,
            "p_over": p,
            "predicted": "OVER" if pred_over else "UNDER",
            "actual_total": total_gole,
            "correct": pred_over == actual_over,
        })
    return results


def eval_stat_market(pred_row: pd.Series, actual_row: pd.Series, market_key: str, actual_col_h: str, actual_col_a: str) -> Dict:
    """Ewaluuje wszystkie linie over/under dla rynku statystycznego."""
    actual_h = actual_row.get(actual_col_h)
    actual_a = actual_row.get(actual_col_a)
    
    if pd.isna(actual_h) or pd.isna(actual_a):
        return None
    
    actual_total = float(actual_h) + float(actual_a)
    mu = pred_row.get(f"mu_{market_key}")
    
    # Znajdz wszystkie linie w predykcjach
    pattern = re.compile(rf"^{re.escape(market_key)}_p_(over|under)_(\d+)_(\d+)$")
    lines_data = {}
    
    for col in pred_row.index:
        m = pattern.match(col)
        if not m:
            continue
        side = m.group(1)
        line = float(f"{m.group(2)}.{m.group(3)}")
        val = pred_row[col]
        if pd.isna(val):
            continue
        if line not in lines_data:
            lines_data[line] = {"line": line, "p_over": None, "p_under": None}
        lines_data[line][f"p_{side}"] = float(val)
    
    # Dla kazdej linii - trafienie
    lines_eval = []
    for line, data in sorted(lines_data.items()):
        actual_over = actual_total > line
        p_over = data["p_over"]
        p_under = data["p_under"]
        
        # Wybieramy stronę z większym prawdopodobieństwem
        if p_over is not None and p_under is not None:
            if p_over > p_under:
                predicted = "OVER"
                pred_p = p_over
            else:
                predicted = "UNDER"
                pred_p = p_under
        elif p_over is not None:
            predicted = "OVER"
            pred_p = p_over
        else:
            predicted = "UNDER"
            pred_p = p_under
        
        correct = (predicted == "OVER" and actual_over) or (predicted == "UNDER" and not actual_over)
        
        lines_eval.append({
            "line": line,
            "predicted": predicted,
            "pred_p": pred_p,
            "correct": correct,
        })
    
    return {
        "mu": mu,
        "actual_total": actual_total,
        "actual_h": actual_h,
        "actual_a": actual_a,
        "diff_from_mu": actual_total - float(mu) if mu else None,
        "lines": lines_eval,
    }


STAT_MARKETS = [
    ("corners", "rozne_gosp", "rozne_gosc", "Kornery"),
    ("shots", "strzaly_gosp", "strzaly_gosc", "Strzały"),
    ("sot", "celne_gosp", "celne_gosc", "Strzały celne"),
    ("offsides", "spalone_gosp", "spalone_gosc", "Spalone"),
    ("fouls", "faule_gosp", "faule_gosc", "Faule"),
    ("yc", "zk_gosp", "zk_gosc", "Żółte kartki"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sezon", required=True)
    parser.add_argument("--kolejka", type=int, required=True)
    args = parser.parse_args()
    
    print("=" * 100)
    print(f"EWALUACJA PREDYKCJI - {args.sezon} K{args.kolejka:02d}")
    print("=" * 100)
    
    preds = load_predictions(args.sezon, args.kolejka)
    actuals = load_actuals(args.sezon, args.kolejka)
    
    print(f"Predykcje: {len(preds)} meczów")
    print(f"Rozegrane: {len(actuals)} meczów")
    print()
    
    # Match predictions to actuals
    matched = []
    for _, pred in preds.iterrows():
        actual = actuals[
            (actuals["gospodarz"] == pred["gospodarz"]) & 
            (actuals["gosc"] == pred["gosc"])
        ]
        if len(actual) == 0:
            print(f"[SKIP] Brak w matches: {pred['gospodarz']} vs {pred['gosc']}")
            continue
        matched.append((pred, actual.iloc[0]))
    
    print(f"Zmatchowanych: {len(matched)}")
    print()
    
    # ==================== SZCZEGÓŁY PER MECZ ====================
    print("=" * 100)
    print("SZCZEGÓŁY PER MECZ")
    print("=" * 100)
    
    all_1x2 = []
    all_btts = []
    all_goals = {0.5: [], 1.5: [], 2.5: [], 3.5: []}
    all_stat_markets = {m[0]: [] for m in STAT_MARKETS}
    
    for pred, actual in matched:
        print(f"\n{pred['gospodarz']} {int(actual['gole_gosp'])}-{int(actual['gole_gosc'])} {pred['gosc']}")
        print("-" * 80)
        
        # 1X2
        r = eval_1x2(pred, actual)
        marker = "✓" if r["correct"] else "✗"
        print(f"  1X2: {marker} pred={r['pred_1x2']} ({r['pred_p']*100:.1f}%) actual={r['actual_1x2']}")
        all_1x2.append(r["correct"])
        
        # BTTS
        r = eval_btts(pred, actual)
        marker = "✓" if r["correct"] else "✗"
        print(f"  BTTS: {marker} pred={r['pred_btts']} ({r['pred_p']*100:.1f}%) actual={r['actual_btts']}")
        all_btts.append(r["correct"])
        
        # Gole over/under
        goals_res = eval_goals_over(pred, actual)
        print(f"  Gole (total={goals_res[0]['actual_total']}):")
        for g in goals_res:
            marker = "✓" if g["correct"] else "✗"
            print(f"    {marker} O{g['line']}: pred={g['predicted']} ({g['p_over']*100:.1f}%)")
            all_goals[g["line"]].append(g["correct"])
        
        # Statystyki
        for market_key, col_h, col_a, label in STAT_MARKETS:
            r = eval_stat_market(pred, actual, market_key, col_h, col_a)
            if r is None:
                continue
            print(f"  {label}: actual={r['actual_total']:.0f} ({int(r['actual_h'])}+{int(r['actual_a'])}) mu={r['mu']:.2f} diff={r['diff_from_mu']:+.2f}")
            
            # Pokazujemy tylko trafność wybranej linii (najbliższej mu)
            best_line = min(r["lines"], key=lambda x: abs(x["line"] - r["mu"]))
            marker = "✓" if best_line["correct"] else "✗"
            print(f"    {marker} Linia centralna {best_line['line']:.1f}: pred={best_line['predicted']} ({best_line['pred_p']*100:.1f}%)")
            
            all_stat_markets[market_key].extend([(l["line"], l["correct"], l["predicted"], l["pred_p"]) for l in r["lines"]])
    
    # ==================== PODSUMOWANIE ====================
    print()
    print("=" * 100)
    print("PODSUMOWANIE TRAFNOŚCI")
    print("=" * 100)
    
    def pct(lst):
        if not lst: return "N/A"
        return f"{sum(lst)}/{len(lst)} ({sum(lst)/len(lst)*100:.1f}%)"
    
    print(f"\n1X2:  {pct(all_1x2)}")
    print(f"BTTS: {pct(all_btts)}")
    
    print(f"\nGole over/under:")
    for line, results in all_goals.items():
        print(f"  O{line}: {pct(results)}")
    
    print(f"\nStatystyki (per rynek, wszystkie linie):")
    for market_key, _, _, label in STAT_MARKETS:
        results = all_stat_markets[market_key]
        if not results:
            print(f"  {label}: N/A")
            continue
        correct_all = [r[1] for r in results]
        print(f"  {label:<20} {pct(correct_all)}")
        
        # Rozklad po linii
        lines_dict = {}
        for line, correct, _, _ in results:
            if line not in lines_dict:
                lines_dict[line] = []
            lines_dict[line].append(correct)
        
        for line in sorted(lines_dict.keys()):
            print(f"      Linia {line:.1f}: {pct(lines_dict[line])}")
    
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()