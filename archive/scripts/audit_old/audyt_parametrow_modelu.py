"""
audyt_parametrow_modelu.py
==========================
Cel:
- porównanie surowych statystyk drużyn z parametrami MLE
- wykrycie drużyn gdzie model znacząco odbiega od surowych danych
- identyfikacja potencjalnych problemów z estymacją

Porównujemy:
  atak_surowy    = avg_gole_strzelone / mu_ogolne
  obrona_surowa  = avg_gole_stracone / mu_ogolne
  atak_MLE       = parametr α z modelu
  obrona_MLE     = parametr β z modelu

Źródła:
  db/ekstraklasa.db
  data/processed/parametry_modelu_gole.json
"""

import sqlite3
import json
from pathlib import Path
import pandas as pd
import numpy as np

# ── ścieżki ───────────────────────────────────────────────────────────────────
DB_PATH = Path("db/ekstraklasa.db")
PARAMS_PATH = Path("data/processed/parametry_modelu_gole.json")
OUTPUT_PATH = Path("data/processed/audyt_parametrow.csv")

# ── wczytanie danych ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM matches", conn)
conn.close()

with open(PARAMS_PATH, encoding="utf-8") as f:
    params = json.load(f)

df_mle = pd.DataFrame(params["druzyny"])
mu_home = params["mu_home"]
mu_away = params["mu_away"]
mu_ogolne = (mu_home + mu_away) / 2

print(f"Wczytano {len(df)} meczów.")
print(f"μ_home={mu_home}, μ_away={mu_away}, μ_ogólne={mu_ogolne:.4f}\n")

SEZONY = sorted(df["sezon"].unique())

# =============================================================================
# 1. SUROWE STATYSTYKI PER DRUŻYNA PER SEZON
# =============================================================================

def surowe_statystyki(df, druzyna, sezon=None):
    """
    Liczy surowe statystyki drużyny.
    Jeśli sezon=None → całość danych.
    """
    if sezon:
        df_f = df[df["sezon"] == sezon]
    else:
        df_f = df

    as_home = df_f[df_f["gospodarz"] == druzyna]
    as_away = df_f[df_f["gosc"] == druzyna]

    n_home = len(as_home)
    n_away = len(as_away)
    n_total = n_home + n_away

    if n_total == 0:
        return None

    # gole strzelone
    gole_str = (
        as_home["gole_gosp"].sum() +
        as_away["gole_gosc"].sum()
    )

    # gole stracone
    gole_strac = (
        as_home["gole_gosc"].sum() +
        as_away["gole_gosp"].sum()
    )

    avg_str = gole_str / n_total
    avg_strac = gole_strac / n_total

    return {
        "n_mecze": n_total,
        "n_home": n_home,
        "n_away": n_away,
        "gole_strzelone": int(gole_str),
        "gole_stracone": int(gole_strac),
        "avg_strzelone": round(avg_str, 3),
        "avg_stracone": round(avg_strac, 3),
        "atak_surowy": round(avg_str / mu_ogolne, 4),
        "obrona_surowa": round(avg_strac / mu_ogolne, 4),
    }


# =============================================================================
# 2. TABELA PORÓWNAWCZA — CAŁOŚĆ DANYCH
# =============================================================================

print("=" * 70)
print("PORÓWNANIE SUROWE vs MLE — CAŁOŚĆ DANYCH (3 sezony)")
print("=" * 70)

wszystkie_druzyny = sorted(
    set(df["gospodarz"].unique()) | set(df["gosc"].unique())
)

rows_global = []

for druzyna in wszystkie_druzyny:
    sur = surowe_statystyki(df, druzyna)
    if sur is None:
        continue

    mle_row = df_mle[df_mle["druzyna"] == druzyna]
    if mle_row.empty:
        continue

    atak_mle = mle_row.iloc[0]["atak"]
    obrona_mle = mle_row.iloc[0]["obrona"]

    delta_atak = round(atak_mle - sur["atak_surowy"], 4)
    delta_obrona = round(obrona_mle - sur["obrona_surowa"], 4)

    rows_global.append({
        "druzyna": druzyna,
        "n_mecze": sur["n_mecze"],
        "gole_str": sur["gole_strzelone"],
        "gole_strac": sur["gole_stracone"],
        "avg_str": sur["avg_strzelone"],
        "avg_strac": sur["avg_stracone"],
        "atak_surowy": sur["atak_surowy"],
        "atak_MLE": round(atak_mle, 4),
        "delta_atak": delta_atak,
        "obrona_surowa": sur["obrona_surowa"],
        "obrona_MLE": round(obrona_mle, 4),
        "delta_obrona": delta_obrona,
    })

df_audit = pd.DataFrame(rows_global)
df_audit = df_audit.sort_values("delta_atak", ascending=False).reset_index(drop=True)

# wyświetlenie pełnej tabeli
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

print(df_audit.to_string(index=False))

# =============================================================================
# 3. DRUŻYNY Z NAJWIĘKSZYMI ODCHYLENIAMI
# =============================================================================

PROG_DELTA = 0.10  # próg odchylenia który uznajemy za "podejrzany"

print("\n" + "=" * 70)
print(f"DRUŻYNY Z ODCHYLENIEM > {PROG_DELTA} (atak lub obrona)")
print("=" * 70)

podejrzane = df_audit[
    (df_audit["delta_atak"].abs() > PROG_DELTA) |
    (df_audit["delta_obrona"].abs() > PROG_DELTA)
].copy()

if podejrzane.empty:
    print("Brak drużyn z dużymi odchyleniami.")
else:
    for _, row in podejrzane.iterrows():
        print(f"\n  {row['druzyna']} ({row['n_mecze']} meczów)")
        print(f"    ATAK:   surowy={row['atak_surowy']:.4f}  "
              f"MLE={row['atak_MLE']:.4f}  "
              f"delta={row['delta_atak']:+.4f}")
        print(f"    OBRONA: surowa={row['obrona_surowa']:.4f}  "
              f"MLE={row['obrona_MLE']:.4f}  "
              f"delta={row['delta_obrona']:+.4f}")

        # interpretacja
        if row["delta_atak"] > PROG_DELTA:
            print(f"    ⚠️  MLE PRZESZACOWUJE atak o {row['delta_atak']:+.4f}")
        elif row["delta_atak"] < -PROG_DELTA:
            print(f"    ⚠️  MLE NIEDOSZACOWUJE atak o {row['delta_atak']:+.4f}")

        if row["delta_obrona"] > PROG_DELTA:
            print(f"    ⚠️  MLE PRZESZACOWUJE przepuszczalność obrony "
                  f"o {row['delta_obrona']:+.4f}")
        elif row["delta_obrona"] < -PROG_DELTA:
            print(f"    ⚠️  MLE NIEDOSZACOWUJE przepuszczalność obrony "
                  f"o {row['delta_obrona']:+.4f}")

# =============================================================================
# 4. STATYSTYKI PER SEZON — SZCZEGÓŁOWY WIDOK
# =============================================================================

print("\n" + "=" * 70)
print("SUROWE STATYSTYKI PER DRUŻYNA PER SEZON")
print("=" * 70)

rows_per_sezon = []

for druzyna in wszystkie_druzyny:
    for sezon in SEZONY:
        sur = surowe_statystyki(df, druzyna, sezon)
        if sur is None or sur["n_mecze"] == 0:
            continue

        rows_per_sezon.append({
            "druzyna": druzyna,
            "sezon": sezon,
            "n_mecze": sur["n_mecze"],
            "gole_str": sur["gole_strzelone"],
            "gole_strac": sur["gole_stracone"],
            "avg_str": sur["avg_strzelone"],
            "avg_strac": sur["avg_stracone"],
            "atak_surowy": sur["atak_surowy"],
            "obrona_surowa": sur["obrona_surowa"],
        })

df_per_sezon = pd.DataFrame(rows_per_sezon)

# wyświetl tylko drużyny które grają w 2025/26
druzyny_aktywne = sorted(
    set(df[df["sezon"] == "2025/26"]["gospodarz"].unique()) |
    set(df[df["sezon"] == "2025/26"]["gosc"].unique())
)

print("\nAktywne drużyny w 2025/26 — ewolucja per sezon:\n")

for druzyna in druzyny_aktywne:
    df_d = df_per_sezon[df_per_sezon["druzyna"] == druzyna]
    if df_d.empty:
        continue

    mle_row = df_mle[df_mle["druzyna"] == druzyna]
    atak_mle = mle_row.iloc[0]["atak"] if not mle_row.empty else None
    obrona_mle = mle_row.iloc[0]["obrona"] if not mle_row.empty else None

    print(f"  {'─' * 60}")
    print(f"  {druzyna}  "
          f"[MLE → atak={atak_mle:.4f}, obrona={obrona_mle:.4f}]")

    for _, row in df_d.iterrows():
        print(f"    {row['sezon']}: "
              f"{row['n_mecze']} meczów | "
              f"str={row['gole_str']} ({row['avg_str']:.2f}/mecz) | "
              f"strac={row['gole_strac']} ({row['avg_strac']:.2f}/mecz) | "
              f"atak_sur={row['atak_surowy']:.3f} "
              f"obr_sur={row['obrona_surowa']:.3f}")

# =============================================================================
# 5. KORELACJA SUROWE vs MLE
# =============================================================================

print("\n" + "=" * 70)
print("KORELACJA SUROWE vs MLE")
print("=" * 70)

corr_atak = df_audit["atak_surowy"].corr(df_audit["atak_MLE"])
corr_obrona = df_audit["obrona_surowa"].corr(df_audit["obrona_MLE"])

print(f"\n  Korelacja atak_surowy  vs atak_MLE:   {corr_atak:.4f}")
print(f"  Korelacja obrona_surowa vs obrona_MLE: {corr_obrona:.4f}")
print()

if corr_atak > 0.90:
    print("  ✅ Bardzo wysoka korelacja ataku — MLE i surowe są spójne")
elif corr_atak > 0.75:
    print("  ⚠️  Umiarkowana korelacja ataku — są istotne odchylenia")
else:
    print("  ❌ Niska korelacja ataku — MLE znacząco odbiega od surowych")

if corr_obrona > 0.90:
    print("  ✅ Bardzo wysoka korelacja obrony — MLE i surowe są spójne")
elif corr_obrona > 0.75:
    print("  ⚠️  Umiarkowana korelacja obrony — są istotne odchylenia")
else:
    print("  ❌ Niska korelacja obrony — MLE znacząco odbiega od surowych")

# =============================================================================
# 6. PODSUMOWANIE DIAGNOSTYCZNE
# =============================================================================

print("\n" + "=" * 70)
print("PODSUMOWANIE DIAGNOSTYCZNE")
print("=" * 70)

print(f"\n  Liczba drużyn w audycie: {len(df_audit)}")
print(f"  Drużyny z odchyleniem > {PROG_DELTA}: {len(podejrzane)}")

print(f"\n  Największe odchylenie ataku (MLE vs surowy):")
max_delta_atak = df_audit.loc[df_audit["delta_atak"].abs().idxmax()]
print(f"    {max_delta_atak['druzyna']}: "
      f"surowy={max_delta_atak['atak_surowy']:.4f} "
      f"MLE={max_delta_atak['atak_MLE']:.4f} "
      f"delta={max_delta_atak['delta_atak']:+.4f}")

print(f"\n  Największe odchylenie obrony (MLE vs surowy):")
max_delta_obr = df_audit.loc[df_audit["delta_obrona"].abs().idxmax()]
print(f"    {max_delta_obr['druzyna']}: "
      f"surowa={max_delta_obr['obrona_surowa']:.4f} "
      f"MLE={max_delta_obr['obrona_MLE']:.4f} "
      f"delta={max_delta_obr['delta_obrona']:+.4f}")

print(f"\n  Średnie odchylenie bezwzględne:")
print(f"    atak:   {df_audit['delta_atak'].abs().mean():.4f}")
print(f"    obrona: {df_audit['delta_obrona'].abs().mean():.4f}")

# =============================================================================
# 7. ZAPIS
# =============================================================================

df_audit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ Audyt zapisany: {OUTPUT_PATH}")
print("\n✅ Audyt parametrów zakończony.")