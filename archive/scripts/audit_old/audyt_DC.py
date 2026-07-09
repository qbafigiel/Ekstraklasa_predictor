# uruchom to jako szybki skrypt diagnostyczny
import sqlite3
import pandas as pd
from pathlib import Path

conn = sqlite3.connect("db/ekstraklasa.db")
df = pd.read_sql("SELECT * FROM matches", conn)
conn.close()

total = len(df)
n_00 = ((df["gole_gosp"] == 0) & (df["gole_gosc"] == 0)).sum()
n_10 = ((df["gole_gosp"] == 1) & (df["gole_gosc"] == 0)).sum()
n_01 = ((df["gole_gosp"] == 0) & (df["gole_gosc"] == 1)).sum()
n_11 = ((df["gole_gosp"] == 1) & (df["gole_gosc"] == 1)).sum()

print(f"Total meczów : {total}")
print(f"0:0  : {n_00} ({n_00/total:.1%})")
print(f"1:0  : {n_10} ({n_10/total:.1%})")
print(f"0:1  : {n_01} ({n_01/total:.1%})")
print(f"1:1  : {n_11} ({n_11/total:.1%})")

# Poisson przewiduje przy mu_home=1.51, mu_away=1.14:
import numpy as np
from scipy.stats import poisson
lh, la = 1.51, 1.14
print(f"\nPoisson przewiduje:")
print(f"P(0:0) = {poisson.pmf(0,lh)*poisson.pmf(0,la):.3f}")
print(f"P(1:1) = {poisson.pmf(1,lh)*poisson.pmf(1,la):.3f}")