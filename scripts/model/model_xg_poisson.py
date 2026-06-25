import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Wagi dla xG (możemy zacząć od 100% xG jako sygnału siły)
def przygotuj_dane_xg(df_trening):
    druzyny = sorted(set(df_trening["gospodarz"]) | set(df_trening["gosc"]))
    n = len(druzyny)
    t2i = {t: i for i, t in enumerate(druzyny)}
    i2t = {i: t for t, i in t2i.items()}

    return {
        "n_druzyn": n,
        "idx_to_team": i2t,
        "home_idx": df_trening["gospodarz"].map(t2i).values,
        "away_idx": df_trening["gosc"].map(t2i).values,
        # UŻYWAMY xG ZAMIAST GOLI DO MLE
        "goals_home": df_trening["xg_gosp"].values.astype(float),
        "goals_away": df_trening["xg_gosc"].values.astype(float),
        "weights": df_trening["waga_sezonu"].values.astype(float),
    }

# Funkcja NLL zostaje taka sama jak w model_gole_poisson.py
# (Poisson świetnie modeluje xG, mimo że xG nie jest liczbą całkowitą)

def trenuj_model_xg(data):
    N = data["n_druzyn"]
    theta0 = np.zeros(2 + 2 * (N - 1))
    # MLE na xG
    res = minimize(neg_log_likelihood, theta0, args=(data,), method="L-BFGS-B")
    return res.x