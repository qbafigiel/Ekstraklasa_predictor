import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("db/ekstraklasa.db")
df = pd.read_sql_query("SELECT * FROM matches", conn)
conn.close()

stats = [
    ("Strzaly",  "strzaly_gosp",  "strzaly_gosc"),
    ("Celne",    "celne_gosp",    "celne_gosc"),
    ("Spalone",  "spalone_gosp",  "spalone_gosc"),
    ("Kornery",  "rozne_gosp",    "rozne_gosc"),
    ("Faule",    "faule_gosp",    "faule_gosc"),
    ("ZK",       "zk_gosp",       "zk_gosc"),
    ("CK",       "czk_gosp",      "czk_gosc"),
]

print(f"{'Stat':12s} {'Mean':>7s} {'Med':>5s} {'Min':>5s} {'Max':>5s} {'Std':>6s} {'Var/Mean':>8s} {'Model':>12s}")
print("-" * 72)

for name, col_h, col_g in stats:
    suma = df[col_h] + df[col_g]
    mean = suma.mean()
    med  = suma.median()
    mn   = suma.min()
    mx   = suma.max()
    std  = suma.std()
    var  = suma.var()
    disp = var / mean if mean > 0 else 0

    if disp < 1.3:
        model = "Poisson OK"
    elif disp < 2.0:
        model = "NegBin lekki"
    else:
        model = "NegBin mocny"

    print(f"{name:12s} {mean:7.2f} {med:5.1f} {mn:5.0f} {mx:5.0f} {std:6.2f} {disp:8.2f} {model:>12s}")

print()
print("Var/Mean = 1.0  -> idealny Poisson")
print("Var/Mean > 1.5  -> overdispersion, lepszy Negative Binomial")
print("Var/Mean < 0.8  -> underdispersion (rzadkie)")

print()
print("=== ROZKŁAD PER DRUZYNA (nie suma) ===")
print(f"{'Stat':12s} {'H_mean':>7s} {'A_mean':>7s} {'H_std':>6s} {'A_std':>6s} {'H_Var/M':>8s} {'A_Var/M':>8s}")
print("-" * 60)

for name, col_h, col_g in stats:
    h = df[col_h].dropna()
    a = df[col_g].dropna()
    h_mean = h.mean()
    a_mean = a.mean()
    h_std  = h.std()
    a_std  = a.std()
    h_disp = h.var() / h_mean if h_mean > 0 else 0
    a_disp = a.var() / a_mean if a_mean > 0 else 0
    print(f"{name:12s} {h_mean:7.2f} {a_mean:7.2f} {h_std:6.2f} {a_std:6.2f} {h_disp:8.2f} {a_disp:8.2f}")

print()
print("=== PROPOZYCJE LINII OVER/UNDER ===")
for name, col_h, col_g in stats:
    suma = df[col_h] + df[col_g]
    mean = suma.mean()
    print(f"\n{name} (srednia sumy={mean:.1f}):")
    candidates = sorted(set([
        round(mean * 0.70 * 2) / 2,
        round(mean * 0.85 * 2) / 2,
        round(mean * 1.00 * 2) / 2,
        round(mean * 1.15 * 2) / 2,
        round(mean * 1.30 * 2) / 2,
    ]))
    for line in candidates:
        pct_over = (suma > line).mean()
        pct_under = 1 - pct_over
        print(f"  Over {line:5.1f}: {pct_over:5.1%}  |  Under {line:5.1f}: {pct_under:5.1%}")