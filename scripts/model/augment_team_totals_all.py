"""
augment_team_totals_all.py
==========================
Dogrywa TEAM TOTALS do już istniejących modeli OOS.

Rynki:
- Kornery
- Strzały
- Strzały celne
- Spalone
- Faule
- Żółte kartki

Dla każdego rynku:
- czyta gotowy plik OOS z lambdami
- odtwarza rozkład gospodarza i gościa
- liczy linie O/U dla gospodarza i gościa
- zapisuje nowy plik *_team_totals.csv

Nie rusza istniejących modeli.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports/model")
REPORT_TXT = REPORT_DIR / "team_totals_augmentation_report.txt"


def ll_binary(p, actual):
    if actual == 1:
        return -np.log(max(float(p), 1e-12))
    return -np.log(max(1.0 - float(p), 1e-12))


def build_team_distribution(mu, alpha=None, max_k=30):
    if alpha is None or alpha <= 1e-8:
        return np.array([poisson.pmf(k, mu) for k in range(max_k + 1)])
    else:
        r = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        return np.array([nbinom.pmf(k, r, p) for k in range(max_k + 1)])


def exact_distribution_dict(p_vec, prefix, max_exact):
    out = {}
    for k in range(min(max_exact + 1, len(p_vec))):
        out[f"{prefix}_{k}"] = round(float(p_vec[k]), 6)
    tail = float(sum(p_vec[max_exact + 1:])) if max_exact + 1 < len(p_vec) else 0.0
    out[f"{prefix}_{max_exact + 1}_plus"] = round(tail, 6)
    return out


def compute_team_lines(p_vec, lines, prefix):
    out = {}
    for line in lines:
        key = str(line).replace(".", "_")
        p_over = float(sum(p_vec[k] for k in range(len(p_vec)) if k > line))
        out[f"{prefix}_p_over_{key}"] = round(p_over, 6)
        out[f"{prefix}_p_under_{key}"] = round(1.0 - p_over, 6)
    return out


def augment_one_market(
    input_filename,
    output_filename,
    home_actual_col,
    away_actual_col,
    team_lines,
    team_max_k,
    team_exact_max,
    use_alpha=False,
):
    input_path = PROCESSED_DIR / input_filename
    output_path = PROCESSED_DIR / output_filename

    if not input_path.exists():
        return {
            "status": "missing",
            "input": str(input_path),
            "output": str(output_path),
            "rows": 0,
        }

    df = pd.read_csv(input_path)

    required = {"lambda_home", "lambda_away", home_actual_col, away_actual_col}
    missing = required - set(df.columns)
    if missing:
        return {
            "status": f"missing_cols: {sorted(missing)}",
            "input": str(input_path),
            "output": str(output_path),
            "rows": 0,
        }

    rows = []

    for _, row in df.iterrows():
        mu_home = float(row["lambda_home"])
        mu_away = float(row["lambda_away"])
        alpha = float(row["alpha"]) if use_alpha and "alpha" in df.columns else None

        p_home_vec = build_team_distribution(mu_home, alpha=alpha, max_k=team_max_k)
        p_away_vec = build_team_distribution(mu_away, alpha=alpha, max_k=team_max_k)

        r = row.to_dict()

        # exact distributions teamowe
        r.update(exact_distribution_dict(p_home_vec, "p_home", team_exact_max))
        r.update(exact_distribution_dict(p_away_vec, "p_away", team_exact_max))

        # team lines
        r.update(compute_team_lines(p_home_vec, team_lines, "home"))
        r.update(compute_team_lines(p_away_vec, team_lines, "away"))

        # actual + log-loss dla team lines
        actual_home = float(row[home_actual_col])
        actual_away = float(row[away_actual_col])

        for line in team_lines:
            key = str(line).replace(".", "_")

            # home
            p_home_over = r[f"home_p_over_{key}"]
            act_home = int(actual_home > line)
            r[f"home_over_{key}_rzecz"] = act_home
            r[f"home_ll_over_{key}"] = round(ll_binary(p_home_over, act_home), 6)

            # away
            p_away_over = r[f"away_p_over_{key}"]
            act_away = int(actual_away > line)
            r[f"away_over_{key}_rzecz"] = act_away
            r[f"away_ll_over_{key}"] = round(ll_binary(p_away_over, act_away), 6)

        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "status": "ok",
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(out),
        "home_lines": len(team_lines),
        "away_lines": len(team_lines),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "Kornery",
            "input_filename": "model_corners_oos_predictions.csv",
            "output_filename": "model_corners_oos_predictions_team_totals.csv",
            "home_actual_col": "rozne_gosp_rzecz",
            "away_actual_col": "rozne_gosc_rzecz",
            "team_lines": [x + 0.5 for x in range(0, 11)],   # 0.5 ... 10.5
            "team_max_k": 20,
            "team_exact_max": 15,
            "use_alpha": True,
        },
        {
            "name": "Strzaly",
            "input_filename": "model_shots_oos_predictions.csv",
            "output_filename": "model_shots_oos_predictions_team_totals.csv",
            "home_actual_col": "strzaly_gosp_rzecz",
            "away_actual_col": "strzaly_gosc_rzecz",
            "team_lines": [x + 0.5 for x in range(2, 23)],   # 2.5 ... 22.5
            "team_max_k": 35,
            "team_exact_max": 25,
            "use_alpha": True,
        },
        {
            "name": "Strzaly celne",
            "input_filename": "model_shots_on_target_oos_predictions.csv",
            "output_filename": "model_shots_on_target_oos_predictions_team_totals.csv",
            "home_actual_col": "celne_gosp_rzecz",
            "away_actual_col": "celne_gosc_rzecz",
            "team_lines": [x + 0.5 for x in range(0, 11)],   # 0.5 ... 10.5
            "team_max_k": 18,
            "team_exact_max": 12,
            "use_alpha": False,
        },
        {
            "name": "Spalone",
            "input_filename": "model_offsides_oos_predictions.csv",
            "output_filename": "model_offsides_oos_predictions_team_totals.csv",
            "home_actual_col": "spalone_gosp_rzecz",
            "away_actual_col": "spalone_gosc_rzecz",
            "team_lines": [x + 0.5 for x in range(0, 5)],    # 0.5 ... 4.5
            "team_max_k": 10,
            "team_exact_max": 8,
            "use_alpha": False,
        },
        {
            "name": "Faule",
            "input_filename": "model_fouls_oos_predictions.csv",
            "output_filename": "model_fouls_oos_predictions_team_totals.csv",
            "home_actual_col": "faule_gosp_rzecz",
            "away_actual_col": "faule_gosc_rzecz",
            "team_lines": [x + 0.5 for x in range(4, 19)],   # 4.5 ... 18.5
            "team_max_k": 30,
            "team_exact_max": 22,
            "use_alpha": True,
        },
        {
            "name": "ZK",
            "input_filename": "model_yellow_cards_oos_predictions.csv",
            "output_filename": "model_yellow_cards_oos_predictions_team_totals.csv",
            "home_actual_col": "zk_gosp_rzecz",
            "away_actual_col": "zk_gosc_rzecz",
            "team_lines": [x + 0.5 for x in range(0, 6)],    # 0.5 ... 5.5
            "team_max_k": 12,
            "team_exact_max": 9,
            "use_alpha": False,
        },
    ]

    results = []

    print("Dogrywam team totals do wszystkich modeli...\n")

    for cfg in configs:
        print(f"=== {cfg['name']} ===")
        res = augment_one_market(
            input_filename=cfg["input_filename"],
            output_filename=cfg["output_filename"],
            home_actual_col=cfg["home_actual_col"],
            away_actual_col=cfg["away_actual_col"],
            team_lines=cfg["team_lines"],
            team_max_k=cfg["team_max_k"],
            team_exact_max=cfg["team_exact_max"],
            use_alpha=cfg["use_alpha"],
        )
        results.append({"name": cfg["name"], **res})
        print(res)
        print()

    # raport tekstowy
    lines = []
    lines.append("=" * 90)
    lines.append("TEAM TOTALS AUGMENTATION REPORT")
    lines.append("=" * 90)
    lines.append("")
    for res in results:
        lines.append(f"{res['name']}")
        lines.append("-" * 90)
        for k, v in res.items():
            if k == "name":
                continue
            lines.append(f"  {k}: {v}")
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nZapisano: {REPORT_TXT}")


if __name__ == "__main__":
    main()