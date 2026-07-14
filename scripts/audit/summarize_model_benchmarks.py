"""
summarize_model_benchmarks.py
==============================
Zbiera wyniki OOS wszystkich modeli backendowych w jedną tabelę.

Czyta pliki predykcji z data/processed/ i generuje:
- tabelę porównawczą log-loss OOS vs benchmark
- kalibrację (model vs rzeczywistość)
- rekomendacje które linie są gotowe produkcyjnie
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports/model")
REPORT_TXT = REPORT_DIR / "model_benchmarks_summary.txt"
REPORT_CSV = REPORT_DIR / "model_benchmarks_summary.csv"

BENCH_1X2 = np.log(3)
BENCH_BINARY = -np.log(0.5)


def ll_binary(p, actual):
    if actual == 1:
        return -np.log(max(float(p), 1e-12))
    return -np.log(max(1.0 - float(p), 1e-12))


def assess_line(ll, bench, delta, label=""):
    """
    Ocena jakości linii:
      EXCELLENT — ll wyraźnie poniżej benchmarku i kalibracja dobra
      GOOD      — ll poniżej benchmarku, kalibracja OK
      OK        — ll poniżej benchmarku, mały bias
      WEAK      — ll blisko benchmarku
      POOR      — ll powyżej benchmarku
    """
    diff = bench - ll
    abs_delta = abs(delta)

    if diff > 0.05 and abs_delta < 0.03:
        return "EXCELLENT"
    elif diff > 0.02 and abs_delta < 0.05:
        return "GOOD"
    elif diff > 0.005 and abs_delta < 0.07:
        return "OK"
    elif diff > 0:
        return "WEAK"
    else:
        return "POOR"


# =============================================================================
# 1X2
# =============================================================================

def analyze_1x2():
    path = PROCESSED_DIR / "baseline_xg_oos_predictions.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    df_test = df[df["sezon"] == "2025/26"] if "sezon" in df.columns else df

    if "ll_1x2" not in df_test.columns:
        return []

    ll = df_test["ll_1x2"].mean()
    bench = BENCH_1X2
    diff = bench - ll

    return [{
        "rynek": "1X2",
        "linia": "1X2",
        "ll_oos": round(ll, 4),
        "benchmark": round(bench, 4),
        "diff_vs_bench": round(diff, 4),
        "model_avg": round(df_test["p_home_cal"].mean(), 3),
        "rzecz_avg": round((df_test["wynik_1x2"] == "H").mean(), 3),
        "delta_kalibracji": round((df_test["wynik_1x2"] == "H").mean() - df_test["p_home_cal"].mean(), 3),
        "ocena": assess_line(ll, bench, 0.0),
        "uwaga": f"Losowy={bench:.4f}",
    }]


# =============================================================================
# BTTS
# =============================================================================

def analyze_btts():
    path = PROCESSED_DIR / "baseline_xg_oos_predictions.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    df_test = df[df["sezon"] == "2025/26"] if "sezon" in df.columns else df

    if "ll_btts" not in df_test.columns:
        return []

    ll = df_test["ll_btts"].mean()
    bench = BENCH_BINARY
    p_avg = df_test["p_btts_yes_cal"].mean()
    r_avg = df_test["btts_rzecz"].mean()
    delta = r_avg - p_avg

    return [{
        "rynek": "BTTS",
        "linia": "BTTS Yes",
        "ll_oos": round(ll, 4),
        "benchmark": round(bench, 4),
        "diff_vs_bench": round(bench - ll, 4),
        "model_avg": round(p_avg, 3),
        "rzecz_avg": round(r_avg, 3),
        "delta_kalibracji": round(delta, 3),
        "ocena": assess_line(ll, bench, delta),
        "uwaga": "",
    }]


# =============================================================================
# OVER/UNDER (generic)
# =============================================================================

def analyze_ou_generic(rynek, filename, ou_lines, col_p_prefix, col_actual_prefix, col_ll_prefix):
    path = PROCESSED_DIR / filename
    if not path.exists():
        return []

    df = pd.read_csv(path)

    rows = []
    for line in ou_lines:
        key = str(line).replace(".", "_")
        col_p = f"{col_p_prefix}{key}"
        col_a = f"over_{key}_rzecz"          # poprawny format
        col_ll = f"{col_ll_prefix}{key}"

        if col_ll not in df.columns:
            continue

        ll = df[col_ll].mean()
        bench = BENCH_BINARY
        p_avg = df[col_p].mean() if col_p in df.columns else np.nan
        r_avg = df[col_a].mean() if col_a in df.columns else np.nan
        delta = r_avg - p_avg if (not np.isnan(p_avg)) and (not np.isnan(r_avg)) else np.nan

        rows.append({
            "rynek": rynek,
            "linia": f"Over {line}",
            "ll_oos": round(ll, 4),
            "benchmark": round(bench, 4),
            "diff_vs_bench": round(bench - ll, 4),
            "model_avg": round(p_avg, 3) if not np.isnan(p_avg) else None,
            "rzecz_avg": round(r_avg, 3) if not np.isnan(r_avg) else None,
            "delta_kalibracji": round(delta, 3) if not np.isnan(delta) else None,
            "ocena": assess_line(ll, bench, delta if not np.isnan(delta) else 0.0),
            "uwaga": "",
        })

    return rows

# =============================================================================
# GOLE (Over/Under)
# =============================================================================

def analyze_gole():
    path = PROCESSED_DIR / "baseline_xg_oos_predictions.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    df_test = df[df["sezon"] == "2025/26"] if "sezon" in df.columns else df

    ou_lines = [0.5, 1.5, 2.5, 3.5]
    rows = []

    for line in ou_lines:
        key = str(line).replace(".", "").replace("_", "")
        col_p = f"p_over_{key}"
        col_a = f"over{key}_rzecz"           # format z baseline: over0_5_rzecz
        col_ll = f"ll_over_{key}"

        # sprawdź też alternatywny format
        if col_a not in df_test.columns:
            col_a = f"over_{key}_rzecz"

        if col_ll not in df_test.columns:
            continue

        ll = df_test[col_ll].mean()
        bench = BENCH_BINARY
        p_avg = df_test[col_p].mean() if col_p in df_test.columns else np.nan
        r_avg = df_test[col_a].mean() if col_a in df_test.columns else np.nan
        delta = r_avg - p_avg if (not np.isnan(p_avg)) and (not np.isnan(r_avg)) else np.nan

        rows.append({
            "rynek": "Gole OU",
            "linia": f"Over {line}",
            "ll_oos": round(ll, 4),
            "benchmark": round(bench, 4),
            "diff_vs_bench": round(bench - ll, 4),
            "model_avg": round(p_avg, 3) if not np.isnan(p_avg) else None,
            "rzecz_avg": round(r_avg, 3) if not np.isnan(r_avg) else None,
            "delta_kalibracji": round(delta, 3) if not np.isnan(delta) else None,
            "ocena": assess_line(ll, bench, delta if not np.isnan(delta) else 0.0),
            "uwaga": "",
        })

    return rows

# =============================================================================
# MAIN
# =============================================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("Analizuję modele...")

    # 1X2
    rows = analyze_1x2()
    all_rows.extend(rows)
    print(f"  1X2: {len(rows)} linii")

    # Gole OU
    rows = analyze_gole()
    all_rows.extend(rows)
    print(f"  Gole OU: {len(rows)} linii")

    # BTTS
    rows = analyze_btts()
    all_rows.extend(rows)
    print(f"  BTTS: {len(rows)} linii")

    # Kornery
    rows = analyze_ou_generic(
        rynek="Kornery",
        filename="model_corners_oos_predictions.csv",
        ou_lines=[4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
        col_p_prefix="p_over_",
        col_actual_prefix="over_",
        col_ll_prefix="ll_over_",
    )
    all_rows.extend(rows)
    print(f"  Kornery: {len(rows)} linii")

    # Strzały
    rows = analyze_ou_generic(
        rynek="Strzaly",
        filename="model_shots_oos_predictions.csv",
        ou_lines=[x + 0.5 for x in range(12, 36)],
        col_p_prefix="p_over_",
        col_actual_prefix="over_",
        col_ll_prefix="ll_over_",
    )
    all_rows.extend(rows)
    print(f"  Strzały: {len(rows)} linii")

    # Strzały celne
    rows = analyze_ou_generic(
        rynek="Strzaly celne",
        filename="model_shots_on_target_oos_predictions.csv",
        ou_lines=[x + 0.5 for x in range(3, 16)],
        col_p_prefix="p_over_",
        col_actual_prefix="over_",
        col_ll_prefix="ll_over_",
    )
    all_rows.extend(rows)
    print(f"  Strzały celne: {len(rows)} linii")

    # Spalone
    rows = analyze_ou_generic(
        rynek="Spalone",
        filename="model_offsides_oos_predictions.csv",
        ou_lines=[x + 0.5 for x in range(0, 7)],
        col_p_prefix="p_over_",
        col_actual_prefix="over_",
        col_ll_prefix="ll_over_",
    )
    all_rows.extend(rows)
    print(f"  Spalone: {len(rows)} linii")

    df = pd.DataFrame(all_rows)
    df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    # ==========================================================================
    # RAPORT TEKSTOWY
    # ==========================================================================

    lines = []
    lines.append("=" * 90)
    lines.append("BENCHMARK WSZYSTKICH MODELI — OOS 2025/26")
    lines.append("=" * 90)
    lines.append("")
    lines.append("LEGENDA OCEN:")
    lines.append("  EXCELLENT — ll >> benchmark, kalibracja świetna (delta < 3%)")
    lines.append("  GOOD      — ll < benchmark, kalibracja dobra (delta < 5%)")
    lines.append("  OK        — ll < benchmark, mały bias (delta < 7%)")
    lines.append("  WEAK      — ll ledwo poniżej benchmarku")
    lines.append("  POOR      — ll >= benchmark, nie używać")
    lines.append("")

    rynki_order = [
        "1X2",
        "Gole OU",
        "BTTS",
        "Kornery",
        "Strzaly",
        "Strzaly celne",
        "Spalone",
    ]

    for rynek in rynki_order:
        subset = df[df["rynek"] == rynek]
        if len(subset) == 0:
            continue

        lines.append(f"{'─' * 90}")
        lines.append(f"  {rynek.upper()}")
        lines.append(f"{'─' * 90}")
        lines.append(
            f"  {'Linia':12s} {'ll_oos':>7s} {'bench':>7s} {'diff':>7s} "
            f"{'model':>7s} {'rzecz':>7s} {'delta_kal':>9s} {'Ocena':>10s}"
        )
        lines.append(f"  {'-' * 80}")

        for row in subset.itertuples(index=False):
            model_str = f"{row.model_avg:.3f}" if row.model_avg is not None else "  n/a "
            rzecz_str = f"{row.rzecz_avg:.3f}" if row.rzecz_avg is not None else "  n/a "
            delta_str = f"{row.delta_kalibracji:+.3f}" if row.delta_kalibracji is not None else "  n/a "

            ocena = row.ocena
            ocena_icon = {
                "EXCELLENT": "★★★",
                "GOOD": "★★ ",
                "OK": "★  ",
                "WEAK": "·  ",
                "POOR": "✗  ",
            }.get(ocena, "   ")

            lines.append(
                f"  {row.linia:12s} {row.ll_oos:7.4f} {row.benchmark:7.4f} "
                f"{row.diff_vs_bench:+7.4f} {model_str:>7s} {rzecz_str:>7s} "
                f"{delta_str:>9s} {ocena_icon} {ocena}"
            )

        lines.append("")

    # Podsumowanie per rynek
    lines.append("=" * 90)
    lines.append("PODSUMOWANIE PER RYNEK")
    lines.append("=" * 90)
    lines.append(
        f"  {'Rynek':15s} {'Linii':>6s} {'EXCELLENT':>10s} {'GOOD':>6s} "
        f"{'OK':>4s} {'WEAK':>6s} {'POOR':>6s} {'Najlepsza linia':>20s}"
    )
    lines.append(f"  {'-' * 80}")

    for rynek in rynki_order:
        subset = df[df["rynek"] == rynek]
        if len(subset) == 0:
            continue

        n_total = len(subset)
        n_exc = (subset["ocena"] == "EXCELLENT").sum()
        n_good = (subset["ocena"] == "GOOD").sum()
        n_ok = (subset["ocena"] == "OK").sum()
        n_weak = (subset["ocena"] == "WEAK").sum()
        n_poor = (subset["ocena"] == "POOR").sum()

        best_row = subset.loc[subset["ll_oos"].idxmin()]
        best_line = best_row["linia"]

        lines.append(
            f"  {rynek:15s} {n_total:6d} {n_exc:10d} {n_good:6d} "
            f"{n_ok:4d} {n_weak:6d} {n_poor:6d} {best_line:>20s}"
        )

    lines.append("")
    lines.append("=" * 90)
    lines.append("TOP 10 LINII OOS (najniższy ll_oos)")
    lines.append("=" * 90)
    top10 = df.nsmallest(10, "ll_oos")
    lines.append(
        f"  {'Rynek':15s} {'Linia':12s} {'ll_oos':>7s} {'bench':>7s} "
        f"{'diff':>7s} {'delta_kal':>9s} {'Ocena':>10s}"
    )
    lines.append(f"  {'-' * 70}")
    for row in top10.itertuples(index=False):
        delta_str = f"{row.delta_kalibracji:+.3f}" if row.delta_kalibracji is not None else "  n/a"
        lines.append(
            f"  {row.rynek:15s} {row.linia:12s} {row.ll_oos:7.4f} "
            f"{row.benchmark:7.4f} {row.diff_vs_bench:+7.4f} "
            f"{delta_str:>9s} {row.ocena}"
        )

    lines.append("")
    lines.append("TOP 10 LINII OOS — WŚRÓD LINII SENSOWNYCH RYNKOWO (50-70% over)")
    lines.append("=" * 90)
    # sensowne rynkowo = model_avg między 0.25 a 0.75
    sensowne = df[
        (df["model_avg"].notna()) &
        (df["model_avg"] >= 0.25) &
        (df["model_avg"] <= 0.75)
    ]
    top10_sensowne = sensowne.nsmallest(10, "ll_oos")
    lines.append(
        f"  {'Rynek':15s} {'Linia':12s} {'ll_oos':>7s} {'bench':>7s} "
        f"{'diff':>7s} {'model':>7s} {'rzecz':>7s} {'Ocena':>10s}"
    )
    lines.append(f"  {'-' * 75}")
    for row in top10_sensowne.itertuples(index=False):
        lines.append(
            f"  {row.rynek:15s} {row.linia:12s} {row.ll_oos:7.4f} "
            f"{row.benchmark:7.4f} {row.diff_vs_bench:+7.4f} "
            f"{row.model_avg:7.3f} {row.rzecz_avg:7.3f} {row.ocena}"
        )

    lines.append("")
    lines.append("PLIKI")
    lines.append(f"  CSV:    {REPORT_CSV}")
    lines.append(f"  Raport: {REPORT_TXT}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {REPORT_TXT}")
    print(f"Zapisano: {REPORT_CSV}")


if __name__ == "__main__":
    main()