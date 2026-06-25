"""
audyt_druzyny.py
================
Cel:
- zidentyfikować które drużyny grały w których sezonach
- sklasyfikować je jako: stabilne / spadkowicze / beniaminkowie
- policzyć empiryczny prior dla beniaminków
  (jak radzą sobie w pierwszych 10 meczach vs reszta ligi)

Źródło danych: db/ekstraklasa.db, tabela matches
"""

import sqlite3
from pathlib import Path
import pandas as pd

# ── ścieżki ──────────────────────────────────────────────────────────────────
DB_PATH = Path("db/ekstraklasa.db")

# ── wczytanie danych ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM matches", conn)
conn.close()

print(f"Wczytano {len(df)} meczów.\n")

# ── lista sezonów ─────────────────────────────────────────────────────────────
SEZONY = sorted(df["sezon"].unique())
print(f"Sezony w bazie: {SEZONY}\n")

# ── które drużyny grały w którym sezonie ─────────────────────────────────────
# każda drużyna pojawia się jako gospodarz LUB gość → bierzemy unię

sezon_druzyny = {}

for sezon in SEZONY:
    df_s = df[df["sezon"] == sezon]
    druzyny = set(df_s["gospodarz"].unique()) | set(df_s["gosc"].unique())
    sezon_druzyny[sezon] = druzyny

# ── tabela obecności drużyn w sezonach ───────────────────────────────────────
wszystkie_druzyny = set()
for d in sezon_druzyny.values():
    wszystkie_druzyny |= d

rows = []
for druzyna in sorted(wszystkie_druzyny):
    row = {"druzyna": druzyna}
    for sezon in SEZONY:
        row[sezon] = "TAK" if druzyna in sezon_druzyny[sezon] else "---"
    rows.append(row)

df_obecnosc = pd.DataFrame(rows)

print("=" * 60)
print("TABELA OBECNOŚCI DRUŻYN W SEZONACH")
print("=" * 60)
print(df_obecnosc.to_string(index=False))
print()

# ── klasyfikacja drużyn ───────────────────────────────────────────────────────
SEZON_AKTUALNY = SEZONY[-1]   # 2025/26
SEZON_POPRZEDNI = SEZONY[-2]  # 2024/25
SEZON_NAJSTARSZY = SEZONY[0]  # 2023/24

stabilne = []
spadkowicze = []
beniaminkowie = []
powracajace = []

for druzyna in sorted(wszystkie_druzyny):
    obecnosc = [druzyna in sezon_druzyny[s] for s in SEZONY]
    w_aktualnym = obecnosc[-1]  # czy gra teraz

    if not w_aktualnym:
        # nie gra w aktualnym sezonie → spadkowicz lub po prostu nieobecny
        spadkowicze.append(druzyna)
    else:
        # gra w aktualnym sezonie
        liczba_sezonow = sum(obecnosc)
        if liczba_sezonow == len(SEZONY):
            # grała we wszystkich sezonach → stabilna
            stabilne.append(druzyna)
        elif obecnosc[0] is False and obecnosc[1] is False:
            # grała tylko w aktualnym sezonie → beniaminek
            beniaminkowie.append(druzyna)
        else:
            # grała w niektórych poprzednich, ale nie we wszystkich → powracająca
            powracajace.append(druzyna)

print("=" * 60)
print("KLASYFIKACJA DRUŻYN")
print("=" * 60)

print(f"\n✅ STABILNE ({len(stabilne)}) — grały we wszystkich 3 sezonach:")
for d in stabilne:
    print(f"   {d}")

print(f"\n🔽 SPADKOWICZE ({len(spadkowicze)}) — nie grają w {SEZON_AKTUALNY}:")
for d in spadkowicze:
    print(f"   {d}")

print(f"\n🆕 BENIAMINKOWIE ({len(beniaminkowie)}) — grają tylko w {SEZON_AKTUALNY}:")
for d in beniaminkowie:
    print(f"   {d}")

print(f"\n🔄 POWRACAJĄCE ({len(powracajace)}) — były, nie było ich, wróciły:")
for d in powracajace:
    print(f"   {d}")

# ── liczba meczów per drużyna per sezon ──────────────────────────────────────
print("\n" + "=" * 60)
print("LICZBA MECZÓW PER DRUŻYNA PER SEZON")
print("=" * 60)

mecze_rows = []
for druzyna in sorted(wszystkie_druzyny):
    row = {"druzyna": druzyna}
    for sezon in SEZONY:
        df_s = df[df["sezon"] == sezon]
        n = ((df_s["gospodarz"] == druzyna) | (df_s["gosc"] == druzyna)).sum()
        row[sezon] = int(n)
    row["RAZEM"] = sum(row[s] for s in SEZONY)
    mecze_rows.append(row)

df_mecze = pd.DataFrame(mecze_rows)
print(df_mecze.to_string(index=False))

# ── empiryczny prior dla beniaminków ─────────────────────────────────────────
print("\n" + "=" * 60)
print("EMPIRYCZNY PRIOR DLA BENIAMINKÓW")
print("=" * 60)
print("(pierwsze 10 meczów drużyn debiutujących vs reszta ligi)")
print()

# beniaminkowie w sensie historycznym:
# drużyny które pojawiły się w sezonie X ale nie grały w X-1
beniaminkowie_hist = []  # (druzyna, sezon_debiutu)

for sezon_idx, sezon in enumerate(SEZONY):
    if sezon_idx == 0:
        continue  # pierwszy sezon — brak poprzedniego do porównania
    sezon_poprz = SEZONY[sezon_idx - 1]
    nowe = sezon_druzyny[sezon] - sezon_druzyny[sezon_poprz]
    for d in nowe:
        beniaminkowie_hist.append((d, sezon))

print(f"Znaleziono {len(beniaminkowie_hist)} historycznych debiutów:\n")
for d, s in beniaminkowie_hist:
    print(f"   {d} → debiut w sezonie {s}")

# ── analiza wyników beniaminków w pierwszych N meczach ───────────────────────

def get_mecze_druzyny_chronologicznie(df_sezon, druzyna):
    """Zwraca mecze drużyny posortowane po kolejce."""
    maska = (df_sezon["gospodarz"] == druzyna) | (df_sezon["gosc"] == druzyna)
    return df_sezon[maska].sort_values("kolejka").reset_index(drop=True)


N_PIERWSZYCH = 10

stats_beniaminkow = []

for druzyna, sezon_debiutu in beniaminkowie_hist:
    df_sezon = df[df["sezon"] == sezon_debiutu].copy()
    mecze = get_mecze_druzyny_chronologicznie(df_sezon, druzyna)

    pierwsze = mecze.head(N_PIERWSZYCH)
    reszta = mecze.tail(len(mecze) - N_PIERWSZYCH)

    def gole_strzelone_stracone(df_m, druzyna):
        """Liczy sumaryczne gole strzelone i stracone."""
        strzelone = 0
        stracone = 0
        for _, row in df_m.iterrows():
            if row["gospodarz"] == druzyna:
                strzelone += row["gole_gosp"]
                stracone += row["gole_gosc"]
            else:
                strzelone += row["gole_gosc"]
                stracone += row["gole_gosp"]
        return strzelone, stracone

    if len(pierwsze) > 0:
        str_p, strac_p = gole_strzelone_stracone(pierwsze, druzyna)
        stats_beniaminkow.append({
            "druzyna": druzyna,
            "sezon": sezon_debiutu,
            "mecze_pierwszych_10": len(pierwsze),
            "gole_strzelone": str_p,
            "gole_stracone": strac_p,
            "avg_strzelone": round(str_p / len(pierwsze), 3),
            "avg_stracone": round(strac_p / len(pierwsze), 3),
        })

df_prior = pd.DataFrame(stats_beniaminkow)

if not df_prior.empty:
    print(f"\nStatystyki w pierwszych {N_PIERWSZYCH} meczach:\n")
    print(df_prior.to_string(index=False))

    # średnia ligowa dla referencji
    avg_gole_gosp = df["gole_gosp"].mean()
    avg_gole_gosc = df["gole_gosc"].mean()
    avg_ligowa = (avg_gole_gosp + avg_gole_gosc) / 2

    avg_strzelone_ben = df_prior["avg_strzelone"].mean()
    avg_stracone_ben = df_prior["avg_stracone"].mean()

    wspolczynnik_ataku = round(avg_strzelone_ben / avg_ligowa, 3)
    wspolczynnik_obrony = round(avg_stracone_ben / avg_ligowa, 3)

    print(f"\nŚrednia ligowa goli na mecz: {avg_ligowa:.3f}")
    print(f"Beniaminkowie avg strzelone: {avg_strzelone_ben:.3f}")
    print(f"Beniaminkowie avg stracone:  {avg_stracone_ben:.3f}")
    print(f"\n→ EMPIRYCZNY PRIOR:")
    print(f"   siła_ataku_beniaminka  = {wspolczynnik_ataku} × średnia ligowa")
    print(f"   siła_obrony_beniaminka = {wspolczynnik_obrony} × średnia ligowa")
else:
    print("Brak wystarczających danych do obliczenia priora.")

print("\n✅ Audyt zakończony.")