import pandas as pd

PLIK = "data/flash_2025_26.csv"

# ============================================================
# DANE DO WPISANIA
# ============================================================

MECZE = [
    {
        "url": "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=MRhRBEoD",
        "dane": {
            "xg_gosp": 2.21, "xg_gosc": 0.63,
            "xgot_gosp": 2.08, "xgot_gosc": 0.76,
            "xa_gosp": 1.00, "xa_gosc": 1.31,
            "posiadanie_gosp": 43, "posiadanie_gosc": 57,
            "strzaly_gosp": 13, "strzaly_gosc": 7,
            "celne_gosp": 3, "celne_gosc": 3,
            "strzaly_niecelne_gosp": 7, "strzaly_niecelne_gosc": 4,
            "strzaly_zablokowane_gosp": 3, "strzaly_zablokowane_gosc": 0,
            "strzaly_pk_gosp": 11, "strzaly_pk_gosc": 3,
            "strzaly_spoza_pk_gosp": 2, "strzaly_spoza_pk_gosc": 4,
            "wielkie_szanse_gosp": 3, "wielkie_szanse_gosc": 0,
            "rozne_gosp": 2, "rozne_gosc": 5,
            "kontakty_pk_gosp": 27, "kontakty_pk_gosc": 20,
            "spalone_gosp": 3, "spalone_gosc": 2,
            "rzuty_wolne_gosp": 16, "rzuty_wolne_gosc": 15,
            "podania_gosp": 248, "podania_gosc": 355,          # skuteczne
            "dlugie_podania_gosp": 18, "dlugie_podania_gosc": 27,
            "dosrodkowania_gosp": 4, "dosrodkowania_gosc": 4,
            "faule_gosp": 15, "faule_gosc": 16,
            "odbiory_gosp": 10, "odbiory_gosc": 9,
            "pojedynki_gosp": 61, "pojedynki_gosc": 59,
            "wybicia_gosp": 31, "wybicia_gosc": 29,
            "przechwyty_gosp": 9, "przechwyty_gosc": 8,
            "bledy_strzal_gosp": 0, "bledy_strzal_gosc": 1,
            "bledy_gol_gosp": 0, "bledy_gol_gosc": 0,
            "obrony_bramkarza_gosp": 3, "obrony_bramkarza_gosc": 0,
            "xgot_przeciw_gosp": 0.76, "xgot_przeciw_gosc": 2.08,
            "zapobiegniecia_gosp": 0.76, "zapobiegniecia_gosc": -0.92,
            "zk_gosp": 5, "zk_gosc": 3,
            "czk_gosp": 0, "czk_gosc": 0,
            "data_meczu_flash": "2025-08-08",
            "kolejka_flash": 4,
            "flash_id": "MRhRBEoD",
        }
    },
    {
        "url": "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=pIwRSRAT",
        "dane": {
            "xg_gosp": 1.85, "xg_gosc": 0.66,
            "xgot_gosp": 0.92, "xgot_gosc": 0.00,
            "xa_gosp": 1.90, "xa_gosc": 0.73,
            "posiadanie_gosp": 63, "posiadanie_gosc": 37,
            "strzaly_gosp": 20, "strzaly_gosc": 6,
            "celne_gosp": 3, "celne_gosc": 0,
            "strzaly_niecelne_gosp": 7, "strzaly_niecelne_gosc": 3,
            "strzaly_zablokowane_gosp": 10, "strzaly_zablokowane_gosc": 3,
            "strzaly_pk_gosp": 15, "strzaly_pk_gosc": 5,
            "strzaly_spoza_pk_gosp": 5, "strzaly_spoza_pk_gosc": 1,
            "wielkie_szanse_gosp": 4, "wielkie_szanse_gosc": 2,
            "rozne_gosp": 13, "rozne_gosc": 4,
            "kontakty_pk_gosp": 44, "kontakty_pk_gosc": 10,
            "spalone_gosp": 1, "spalone_gosc": 1,
            "rzuty_wolne_gosp": 13, "rzuty_wolne_gosc": 19,
            "podania_gosp": 388, "podania_gosc": 213,
            "dlugie_podania_gosp": 34, "dlugie_podania_gosc": 22,
            "dosrodkowania_gosp": 11, "dosrodkowania_gosc": 6,
            "faule_gosp": 19, "faule_gosc": 13,
            "odbiory_gosp": 7, "odbiory_gosc": 8,
            "pojedynki_gosp": 51, "pojedynki_gosc": 48,
            "wybicia_gosp": 24, "wybicia_gosc": 49,
            "przechwyty_gosp": 7, "przechwyty_gosc": 10,
            "bledy_strzal_gosp": 0, "bledy_strzal_gosc": 0,
            "bledy_gol_gosp": 0, "bledy_gol_gosc": 0,
            "obrony_bramkarza_gosp": 0, "obrony_bramkarza_gosc": 3,
            "xgot_przeciw_gosp": 0.00, "xgot_przeciw_gosc": 0.92,
            "zapobiegniecia_gosp": 0.00, "zapobiegniecia_gosc": 0.92,
            "zk_gosp": 3, "zk_gosc": 3,
            "czk_gosp": 0, "czk_gosc": 0,
            "data_meczu_flash": "2025-08-03",
            "kolejka_flash": 3,
            "flash_id": "pIwRSRAT",
        }
    },
]

# ============================================================
# WPISZ DO CSV
# ============================================================

df = pd.read_csv(PLIK)
print(f"CSV wczytany: {len(df)} wierszy, {len(df.columns)} kolumn")

for mecz in MECZE:
    url = mecz["url"]
    dane = mecz["dane"]

    maska = df["url"] == url
    if maska.sum() == 0:
        print(f"  BŁĄD: nie znaleziono wiersza dla URL: {url}")
        continue

    idx = df[maska].index[0]

    for kolumna, wartosc in dane.items():
        if kolumna not in df.columns:
            print(f"  UWAGA: kolumna '{kolumna}' nie istnieje w CSV — pomijam")
            continue
        try:
            dtype = df[kolumna].dtype
            if pd.api.types.is_float_dtype(dtype):
                wartosc = float(wartosc)
            elif pd.api.types.is_integer_dtype(dtype):
                wartosc = int(wartosc)
        except (ValueError, TypeError):
            pass
        df.at[idx, kolumna] = wartosc

    print(f"  OK: wiersz {idx} — {dane.get('data_meczu_flash')} kolejka {dane.get('kolejka_flash')} xG={dane.get('xg_gosp')}/{dane.get('xg_gosc')}")

df.to_csv(PLIK, index=False, encoding="utf-8-sig")
print(f"\nZapisano {PLIK}")

# Weryfikacja
brak = df[df["xg_gosp"].isna() | df["kolejka_flash"].isna()]
print(f"Mecze bez danych po uzupełnieniu: {len(brak)}")
if len(brak) > 0:
    print(brak[["kolejka_flash", "data_meczu_flash", "url"]].to_string())
else:
    print("Wszystko kompletne w zakresie xG i kolejki!")