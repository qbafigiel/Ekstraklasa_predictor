"""
audyt_druzyny_v2.py
====================
Cel:
- klasyfikacja drużyn per sezon (stabilna / spadkowicz / beniaminek)
- obliczenie priora beniaminków BEZ data leakage
- przygotowanie tabeli statusów gotowej pod model

Zasada:
- prior dla beniaminków sezonu X liczymy TYLKO z danych sprzed sezonu X
- nigdy nie używamy przyszłych informacji

Źródło: db/ekstraklasa.db, tabela matches
"""

import sqlite3
from pathlib import Path
import pandas as pd
import json

# ── konfiguracja ──────────────────────────────────────────────────────────────
DB_PATH = Path("db/ekstraklasa.db")
OUTPUT_PATH = Path("data/processed/statusy_druzyn.csv")
PRIOR_PATH = Path("data/processed/priory_beniaminkow.json")
K_PRIOR = 10  # siła priora = równowartość tylu pseudo-meczów

# ── wczytanie danych ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM matches", conn)
conn.close()

print(f"Wczytano {len(df)} meczów.\n")

SEZONY = sorted(df["sezon"].unique())
print(f"Sezony: {SEZONY}\n")

# ── drużyny per sezon ─────────────────────────────────────────────────────────

def druzyny_w_sezonie(df, sezon):
    """Zwraca zbiór drużyn grających w danym sezonie."""
    df_s = df[df["sezon"] == sezon]
    return set(df_s["gospodarz"].unique()) | set(df_s["gosc"].unique())


sezon_druzyny = {s: druzyny_w_sezonie(df, s) for s in SEZONY}

# ── klasyfikacja per sezon ────────────────────────────────────────────────────

def klasyfikuj_druzyny(sezon_idx, sezony, sezon_druzyny):
    """
    Klasyfikuje drużyny w danym sezonie:
    - stabilna: grała w tym I poprzednim sezonie
    - beniaminek: gra w tym, NIE grała w poprzednim
    - spadkowicz: grała w poprzednim, NIE gra w tym

    Dla pierwszego sezonu: wszystkie drużyny = stabilne
    (brak danych do porównania).
    """
    sezon = sezony[sezon_idx]
    druzyny_teraz = sezon_druzyny[sezon]

    if sezon_idx == 0:
        # pierwszy sezon — brak historii, wszystkie traktujemy jako stabilne
        return {
            "stabilne": sorted(druzyny_teraz),
            "beniaminkowie": [],
            "spadkowicze": [],
        }

    sezon_poprzedni = sezony[sezon_idx - 1]
    druzyny_poprzednio = sezon_druzyny[sezon_poprzedni]

    stabilne = sorted(druzyny_teraz & druzyny_poprzednio)
    beniaminkowie = sorted(druzyny_teraz - druzyny_poprzednio)
    spadkowicze = sorted(druzyny_poprzednio - druzyny_teraz)

    return {
        "stabilne": stabilne,
        "beniaminkowie": beniaminkowie,
        "spadkowicze": spadkowicze,
    }


# ── obliczenie priorów BEZ leakage ───────────────────────────────────────────

def gole_druzyny_w_sezonie(df, druzyna, sezon, n_pierwszych=None):
    """
    Zwraca (avg_strzelone, avg_stracone) dla drużyny w sezonie.
    Jeśli n_pierwszych podane — bierze tylko pierwsze N meczów.
    """
    df_s = df[df["sezon"] == sezon].copy()
    maska = (df_s["gospodarz"] == druzyna) | (df_s["gosc"] == druzyna)
    mecze = df_s[maska].sort_values("kolejka").reset_index(drop=True)

    if n_pierwszych is not None:
        mecze = mecze.head(n_pierwszych)

    if len(mecze) == 0:
        return None, None

    strzelone = 0
    stracone = 0
    for _, row in mecze.iterrows():
        if row["gospodarz"] == druzyna:
            strzelone += row["gole_gosp"]
            stracone += row["gole_gosc"]
        else:
            strzelone += row["gole_gosc"]
            stracone += row["gole_gosp"]

    return strzelone / len(mecze), stracone / len(mecze)


def srednia_ligowa_sezon(df, sezon):
    """Średnia goli na mecz w sezonie (per drużyna, nie per mecz)."""
    df_s = df[df["sezon"] == sezon]
    total_gole = df_s["gole_gosp"].sum() + df_s["gole_gosc"].sum()
    total_wystapien = len(df_s) * 2  # 2 drużyny na mecz
    return total_gole / total_wystapien


def oblicz_prior_beniaminkow(df, sezony, sezon_druzyny, sezon_docelowy_idx):
    """
    Liczy prior dla beniaminków sezonu docelowego.
    Używa TYLKO danych z WCZEŚNIEJSZYCH sezonów.

    Metoda:
    - znajduje beniaminków z poprzednich sezonów
    - liczy ich avg strzelone/stracone w pierwszych 10 meczach
    - porównuje ze średnią ligową tamtego sezonu
    """
    N_PIERWSZYCH = 10

    # zbieramy historycznych beniaminków z sezonów PRZED docelowym
    hist_beniaminkowie = []

    for idx in range(1, sezon_docelowy_idx):
        sezon = sezony[idx]
        sezon_poprz = sezony[idx - 1]
        nowi = sezon_druzyny[sezon] - sezon_druzyny[sezon_poprz]

        for druzyna in nowi:
            avg_str, avg_strac = gole_druzyny_w_sezonie(
                df, druzyna, sezon, N_PIERWSZYCH
            )
            avg_lig = srednia_ligowa_sezon(df, sezon)

            if avg_str is not None:
                hist_beniaminkowie.append({
                    "druzyna": druzyna,
                    "sezon": sezon,
                    "avg_strzelone": avg_str,
                    "avg_stracone": avg_strac,
                    "avg_ligowa": avg_lig,
                    "wspolczynnik_ataku": avg_str / avg_lig,
                    "wspolczynnik_obrony": avg_strac / avg_lig,
                })

    if not hist_beniaminkowie:
        # brak historycznych beniaminków — domyślny prior
        return {
            "prior_atak": 0.80,
            "prior_obrona": 1.10,
            "zrodlo": "domyslny (brak danych historycznych)",
            "historia": [],
        }

    df_hist = pd.DataFrame(hist_beniaminkowie)
    prior_atak = round(df_hist["wspolczynnik_ataku"].mean(), 4)
    prior_obrona = round(df_hist["wspolczynnik_obrony"].mean(), 4)

    return {
        "prior_atak": prior_atak,
        "prior_obrona": prior_obrona,
        "zrodlo": "empiryczny z poprzednich sezonów",
        "historia": hist_beniaminkowie,
    }


# ── główna pętla ──────────────────────────────────────────────────────────────

print("=" * 70)
print("KLASYFIKACJA I PRIORY — BEZ DATA LEAKAGE")
print("=" * 70)

wszystkie_statusy = []
wszystkie_priory = {}

for idx, sezon in enumerate(SEZONY):
    print(f"\n{'─' * 70}")
    print(f"SEZON: {sezon}")
    print(f"{'─' * 70}")

    klas = klasyfikuj_druzyny(idx, SEZONY, sezon_druzyny)

    print(f"\n  ✅ Stabilne ({len(klas['stabilne'])}):")
    for d in klas["stabilne"]:
        print(f"     {d}")

    print(f"\n  🆕 Beniaminkowie ({len(klas['beniaminkowie'])}):")
    for d in klas["beniaminkowie"]:
        print(f"     {d}")

    print(f"\n  🔽 Spadkowicze ({len(klas['spadkowicze'])}):")
    for d in klas["spadkowicze"]:
        print(f"     {d}")

    # prior dla beniaminków
    if klas["beniaminkowie"]:
        prior = oblicz_prior_beniaminkow(df, SEZONY, sezon_druzyny, idx)
        wszystkie_priory[sezon] = prior

        print(f"\n  📊 Prior beniaminków dla {sezon}:")
        print(f"     Źródło: {prior['zrodlo']}")
        print(f"     prior_atak  = {prior['prior_atak']}")
        print(f"     prior_obrona = {prior['prior_obrona']}")

        if prior["historia"]:
            print(f"\n     Oparty na historycznych debiutach:")
            for h in prior["historia"]:
                print(f"       {h['druzyna']} ({h['sezon']}): "
                      f"str={h['avg_strzelone']:.2f} "
                      f"strac={h['avg_stracone']:.2f} "
                      f"atak={h['wspolczynnik_ataku']:.3f} "
                      f"obr={h['wspolczynnik_obrony']:.3f}")

    # budowanie tabeli statusów
    avg_lig = srednia_ligowa_sezon(df, sezon)

    for d in klas["stabilne"]:
        avg_str, avg_strac = gole_druzyny_w_sezonie(df, d, sezon)
        n_mecze = ((df[df["sezon"] == sezon]["gospodarz"] == d) |
                   (df[df["sezon"] == sezon]["gosc"] == d)).sum()
        wszystkie_statusy.append({
            "sezon": sezon,
            "druzyna": d,
            "status": "stabilna",
            "mecze_w_sezonie": int(n_mecze),
            "avg_strzelone": round(avg_str, 3) if avg_str else None,
            "avg_stracone": round(avg_strac, 3) if avg_strac else None,
            "avg_ligowa": round(avg_lig, 3),
            "prior_atak": None,
            "prior_obrona": None,
            "k_prior": None,
        })

    for d in klas["beniaminkowie"]:
        avg_str, avg_strac = gole_druzyny_w_sezonie(df, d, sezon)
        n_mecze = ((df[df["sezon"] == sezon]["gospodarz"] == d) |
                   (df[df["sezon"] == sezon]["gosc"] == d)).sum()
        p = wszystkie_priory.get(sezon, {})
        wszystkie_statusy.append({
            "sezon": sezon,
            "druzyna": d,
            "status": "beniaminek",
            "mecze_w_sezonie": int(n_mecze),
            "avg_strzelone": round(avg_str, 3) if avg_str else None,
            "avg_stracone": round(avg_strac, 3) if avg_strac else None,
            "avg_ligowa": round(avg_lig, 3),
            "prior_atak": p.get("prior_atak"),
            "prior_obrona": p.get("prior_obrona"),
            "k_prior": K_PRIOR,
        })

    for d in klas["spadkowicze"]:
        wszystkie_statusy.append({
            "sezon": sezon,
            "druzyna": d,
            "status": "spadkowicz",
            "mecze_w_sezonie": 0,
            "avg_strzelone": None,
            "avg_stracone": None,
            "avg_ligowa": round(avg_lig, 3),
            "prior_atak": None,
            "prior_obrona": None,
            "k_prior": None,
        })

# ── zapis wyników ─────────────────────────────────────────────────────────────

df_statusy = pd.DataFrame(wszystkie_statusy)

# sortowanie
df_statusy = df_statusy.sort_values(["sezon", "status", "druzyna"]).reset_index(
    drop=True
)

df_statusy.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n\n✅ Tabela statusów zapisana: {OUTPUT_PATH}")
print(f"   Rekordów: {len(df_statusy)}")

# zapis priorów do JSON
with open(PRIOR_PATH, "w", encoding="utf-8") as f:
    json.dump(wszystkie_priory, f, ensure_ascii=False, indent=2, default=str)
print(f"✅ Priory zapisane: {PRIOR_PATH}")

# ── podsumowanie ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PODSUMOWANIE PRIORÓW PER SEZON")
print("=" * 70)

for sezon, prior in wszystkie_priory.items():
    print(f"\n  {sezon}:")
    print(f"    prior_atak  = {prior['prior_atak']}")
    print(f"    prior_obrona = {prior['prior_obrona']}")
    print(f"    źródło: {prior['zrodlo']}")

print(f"\n  K_PRIOR (siła priora) = {K_PRIOR} pseudo-meczów")
print(f"  Znaczenie: beniaminek potrzebuje ~{K_PRIOR} meczów żeby prior")
print(f"  przestał dominować i dane realne przejęły kontrolę.")

print("\n" + "=" * 70)
print("FORMUŁA SHRINKAGE DLA BENIAMINKA")
print("=" * 70)
print("""
  Po n rozegranych meczach:

  atak_final = (K × prior_atak + n × atak_z_danych) / (K + n)
  obrona_final = (K × prior_obrona + n × obrona_z_danych) / (K + n)

  Przykład z K=10:
    Po 0 meczach:  100% prior
    Po 5 meczach:  67% prior + 33% dane
    Po 10 meczach: 50% prior + 50% dane
    Po 20 meczach: 33% prior + 67% dane
    Po 30 meczach: 25% prior + 75% dane
""")

print("✅ Audyt v2 zakończony.")