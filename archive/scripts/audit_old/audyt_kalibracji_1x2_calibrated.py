import numpy as np
import pandas as pd

RAW_PATH = "data/processed/backtesting_wyniki_v2.csv"
CAL_PATH = "data/processed/backtesting_wyniki_v2_calibrated.csv"

BINS = np.linspace(0, 1, 11)  # 0.0..1.0 step 0.1


def brier_multiclass(df, prefix=""):
    # p_H, p_D, p_A oraz one-hot y
    pH = df[f"{prefix}p_home"].astype(float).values
    pD = df[f"{prefix}p_draw"].astype(float).values
    pA = df[f"{prefix}p_away"].astype(float).values

    yH = (df["wynik_1x2"].values == "H").astype(float)
    yD = (df["wynik_1x2"].values == "D").astype(float)
    yA = (df["wynik_1x2"].values == "A").astype(float)

    return np.mean((pH - yH) ** 2 + (pD - yD) ** 2 + (pA - yA) ** 2)


def class_reliability(df, p_col, y_mask, bins=BINS):
    p = df[p_col].astype(float).values
    y = y_mask.astype(float).values

    bin_idx = np.digitize(p, bins) - 1
    # bins length = 11 edges => 10 bins
    stats = []
    ece = 0.0
    n = len(df)

    for b in range(len(bins) - 1):
        m = bin_idx == b
        if not np.any(m):
            continue
        conf = p[m].mean()
        acc = y[m].mean()  # frequency of the class in this bin
        frac = m.sum() / n
        ece += frac * abs(acc - conf)
        stats.append((b, bins[b], bins[b+1], m.sum(), conf, acc))
    return ece, stats


def top1_reliability(df, prefix=""):
    pH = df[f"{prefix}p_home"].astype(float).values
    pD = df[f"{prefix}p_draw"].astype(float).values
    pA = df[f"{prefix}p_away"].astype(float).values

    pmat = np.vstack([pH, pD, pA]).T
    pmax = pmat.max(axis=1)
    top_idx = pmat.argmax(axis=1)

    y = df["wynik_1x2"].values
    # map: 0->H, 1->D, 2->A
    y_top = np.array([1.0 if (top_idx[i] == 0 and y[i] == "H") or
                              (top_idx[i] == 1 and y[i] == "D") or
                              (top_idx[i] == 2 and y[i] == "A")
                      else 0.0 for i in range(len(y))])

    bin_idx = np.digitize(pmax, BINS) - 1
    n = len(df)
    ece = 0.0
    stats = []

    for b in range(len(BINS) - 1):
        m = bin_idx == b
        if not np.any(m):
            continue
        conf = pmax[m].mean()
        acc = y_top[m].mean()
        frac = m.sum() / n
        ece += frac * abs(acc - conf)
        stats.append((b, BINS[b], BINS[b+1], m.sum(), conf, acc))
    return ece, stats


def report(df, label, p_prefix=""):
    print(f"\n==== {label} ====")

    yH = (df["wynik_1x2"] == "H")
    yD = (df["wynik_1x2"] == "D")
    yA = (df["wynik_1x2"] == "A")

    pH_col = f"{p_prefix}p_home"
    pD_col = f"{p_prefix}p_draw"
    pA_col = f"{p_prefix}p_away"

    # Mean calibration (avg p vs true freq)
    for name, pcol, ymask in [
        ("HOME", pH_col, yH),
        ("DRAW", pD_col, yD),
        ("AWAY", pA_col, yA),
    ]:
        avg_p = df[pcol].astype(float).mean()
        freq = ymask.mean()
        print(f"{name:5s} | avg p={avg_p:.3f} | true freq={freq:.3f} | delta={freq-avg_p:+.3f}")

    # Brier
    brier = brier_multiclass(df, prefix=p_prefix)
    print(f"Brier multi-class: {brier:.5f}")

    # ECE per class
    for name, pcol, ymask in [
        ("HOME", pH_col, yH),
        ("DRAW", pD_col, yD),
        ("AWAY", pA_col, yA),
    ]:
        ece, _ = class_reliability(df, pcol, ymask)
        print(f"ECE (by p_{name.lower()} bins): {ece:.5f}")

    # Top-1 reliability (najbardziej praktyczne pod argmax)
    ece_top, _ = top1_reliability(df, prefix=p_prefix)
    print(f"ECE (top1 prob bins): {ece_top:.5f}")


def main():
    df_raw = pd.read_csv(RAW_PATH)
    df_cal = pd.read_csv(CAL_PATH)

    # W calibrated pliku prawdopodobnie są kolumny: p_h_cal, p_d_cal, p_a_cal
    # oraz wynik_pred_cal, ll_cal.
    # Jeśli kolumny są inne, dopasujemy prefixy.
    # U nas: p_h_cal, p_d_cal, p_a_cal -> tworzymy tymczasowe mapowanie:
    df_cal = df_cal.copy()
    if "p_h_cal" in df_cal.columns:
        df_cal["p_home"] = df_cal["p_h_cal"]
        df_cal["p_draw"] = df_cal["p_d_cal"]
        df_cal["p_away"] = df_cal["p_a_cal"]

    report(df_raw, "RAW v2", p_prefix="")
    report(df_cal, "CALIBRATED v2", p_prefix="")

if __name__ == "__main__":
    main()