import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/backtesting_wyniki_xg_v1.csv"

# Parametry z kalibratora xG
T = 1.8429
bH = 0.2982
bD = -0.0163
bA = -0.2819


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def apply_calibration(row):

    p_raw = np.array([
        row["p_home"],
        row["p_draw"],
        row["p_away"]
    ])

    logits = np.log(np.maximum(p_raw, 1e-12))
    logits_cal = (logits + np.array([bH, bD, bA])) / T
    p_cal = softmax(logits_cal)

    return pd.Series({
        "p_home_cal": p_cal[0],
        "p_draw_cal": p_cal[1],
        "p_away_cal": p_cal[2]
    })


def log_loss(p, y):
    return -np.log(max(p, 1e-12))


def main():

    df = pd.read_csv(INPUT_PATH)

    df_cal = df.apply(apply_calibration, axis=1)
    df = pd.concat([df, df_cal], axis=1)

    ll_list = []

    for _, row in df.iterrows():

        if row["wynik_1x2"] == "H":
            p = row["p_home_cal"]
        elif row["wynik_1x2"] == "D":
            p = row["p_draw_cal"]
        else:
            p = row["p_away_cal"]

        ll_list.append(log_loss(p, row["wynik_1x2"]))

    df["log_loss_cal"] = ll_list

    print("\n==============================")
    print("WYNIKI xG + KALIBRACJA")
    print("==============================")
    print(f"Log-loss xG RAW: {df['log_loss'].mean():.4f}")
    print(f"Log-loss xG CAL: {df['log_loss_cal'].mean():.4f}")

    # Rozkład typowań
    df["pred_cal"] = df[["p_home_cal","p_draw_cal","p_away_cal"]].idxmax(axis=1)
    df["pred_cal"] = df["pred_cal"].map({
        "p_home_cal": "H",
        "p_draw_cal": "D",
        "p_away_cal": "A"
    })

    print("\nRozkład typowań CAL:")
    for res in ["H","D","A"]:
        print(res, (df["pred_cal"] == res).sum())

    print("\nAccuracy CAL:",
          (df["pred_cal"] == df["wynik_1x2"]).mean())

    df.to_csv("data/processed/backtesting_wyniki_xg_v1_calibrated.csv", index=False)


if __name__ == "__main__":
    main()