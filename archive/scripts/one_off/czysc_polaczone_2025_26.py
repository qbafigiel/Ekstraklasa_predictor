from pathlib import Path
import pandas as pd

INPUT_CSV = Path(r"D:\projects\Ekstraklasa_predictor\data\pelne_2025_26.csv")
OUTPUT_CSV = Path(r"D:\projects\Ekstraklasa_predictor\data\czyste_2025_26.csv")


TEAM_CODE_MAP = {
    "LPO": "Lech Poznań",
    "LEG": "Legia Warszawa",
    "JAG": "Jagiellonia Białystok",
    "RCZ": "Raków Częstochowa",
    "POG": "Pogoń Szczecin",
    "GOR": "Górnik Zabrze",
    "ZAG": "Zagłębie Lubin",
    "CRA": "Cracovia",
    "WPL": "Wisła Płock",
    "PIA": "Piast Gliwice",
    "WID": "Widzew Łódź",
    "GKS": "GKS Katowice",
    "RAD": "Radomiak Radom",
    "MOT": "Motor Lublin",
    "KOR": "Korona Kielce",
    "ARK": "Arka Gdynia",
    "LGD": "Lechia Gdańsk",
    "BBT": "Bruk-Bet Termalica Nieciecza",
}


DUPLICATE_FLASH_COLUMNS = [
    "posiadanie_gosp_flash",
    "posiadanie_gosc_flash",
    "strzaly_gosp_flash",
    "strzaly_gosc_flash",
    "celne_gosp_flash",
    "celne_gosc_flash",
    "rozne_gosp_flash",
    "rozne_gosc_flash",
    "zk_gosp_flash",
    "zk_gosc_flash",
    "strzaly_niecelne_gosp_flash",
    "strzaly_niecelne_gosc_flash",
    "strzaly_zablokowane_gosp_flash",
    "strzaly_zablokowane_gosc_flash",
    "spalone_gosp_flash",
    "spalone_gosc_flash",
    "faule_gosp_flash",
    "faule_gosc_flash",
    "czk_gosp_flash",
    "czk_gosc_flash",
    "data_meczu_flash",
    "kolejka_flash",
]

DUPLICATE_API_ALIAS_COLUMNS = [
    "podania_sk_gosp",
    "podania_wszy_gosp",
    "podania_sk_gosc",
    "podania_wszy_gosc",
    "dosrod_sk_gosp",
    "dosrod_wszy_gosp",
    "dosrod_sk_gosc",
    "dosrod_wszy_gosc",
]

TEAM_HELPER_COLUMNS = [
    "gosp_nazwa",
    "gosc_nazwa",
    "gosp_kod",
    "gosc_kod",
]

RENAME_MAP = {
    "url": "flash_url",

    "dl_pod_sk_gosp": "dlugie_podania_celne_gosp",
    "dl_pod_wszy_gosp": "dlugie_podania_gosp",
    "dl_pod_sk_gosc": "dlugie_podania_celne_gosc",
    "dl_pod_wszy_gosc": "dlugie_podania_gosc",

    "pod_strefa_sk_gosp": "podania_w_strefe_obrony_przeciwnika_celne_gosp",
    "pod_strefa_wszy_gosp": "podania_w_strefe_obrony_przeciwnika_gosp",
    "pod_strefa_sk_gosc": "podania_w_strefe_obrony_przeciwnika_celne_gosc",
    "pod_strefa_wszy_gosc": "podania_w_strefe_obrony_przeciwnika_gosc",

    "odbiory_sk_gosp": "odbiory_skuteczne_fs_gosp",
    "odbiory_wszy_gosp": "proby_odbioru_fs_gosp",
    "odbiory_sk_gosc": "odbiory_skuteczne_fs_gosc",
    "odbiory_wszy_gosc": "proby_odbioru_fs_gosc",

    "gospodarz_kod_clean": "gospodarz_kod",
    "gosc_kod_clean": "gosc_kod",
}


POLISH_CHAR_MAP = str.maketrans({
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
})


def read_csv_flexible(path: Path):
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df, enc
        except Exception:
            continue
    raise RuntimeError(f"Nie udało się odczytać pliku CSV: {path}")


def normalize_team_code(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    text = text.translate(POLISH_CHAR_MAP)
    return text


def map_team_name(code_or_name, fallback_name=None):
    code = normalize_team_code(code_or_name)

    if code in TEAM_CODE_MAP:
        return TEAM_CODE_MAP[code]

    if pd.notna(code_or_name):
        raw = str(code_or_name).strip()
        if len(raw) > 3:
            return raw

    if pd.notna(fallback_name):
        fallback = str(fallback_name).strip()
        if fallback:
            return fallback

    if pd.notna(code_or_name):
        return str(code_or_name).strip()

    return None


def columns_are_identical(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    if col_a not in df.columns or col_b not in df.columns:
        return False

    tmp = df[[col_a, col_b]].dropna()
    if tmp.empty:
        return False

    a_num = pd.to_numeric(tmp[col_a], errors="coerce")
    b_num = pd.to_numeric(tmp[col_b], errors="coerce")

    if a_num.notna().all() and b_num.notna().all():
        return a_num.eq(b_num).all()

    a_txt = tmp[col_a].astype(str).str.strip()
    b_txt = tmp[col_b].astype(str).str.strip()
    return a_txt.eq(b_txt).all()


def main():
    if not INPUT_CSV.exists():
        print(f"❌ Nie znaleziono pliku:\n{INPUT_CSV}")
        return

    df, enc = read_csv_flexible(INPUT_CSV)
    df = df.copy()

    print("=" * 100)
    print("CZYSZCZENIE PLIKU POŁĄCZONEGO API + FLASHSCORE")
    print("=" * 100)
    print(f"Plik wejściowy : {INPUT_CSV}")
    print(f"Encoding       : {enc}")
    print(f"Wiersze        : {len(df)}")
    print(f"Kolumny przed  : {len(df.columns)}")
    print()

    # źródła kodów drużyn
    home_code_source = df["gosp_kod"] if "gosp_kod" in df.columns else df["gospodarz"]
    away_code_source = df["gosc_kod"] if "gosc_kod" in df.columns else df["gosc"]

    # pomocnicze nazwy
    home_helper_names = df["gosp_nazwa"] if "gosp_nazwa" in df.columns else pd.Series([None] * len(df))
    away_helper_names = df["gosc_nazwa"] if "gosc_nazwa" in df.columns else pd.Series([None] * len(df))

    # czyste kody drużyn
    df["gospodarz_kod_clean"] = home_code_source.apply(normalize_team_code)
    df["gosc_kod_clean"] = away_code_source.apply(normalize_team_code)

    # pełne nazwy drużyn
    df["gospodarz"] = [
        map_team_name(code, helper)
        for code, helper in zip(df["gospodarz_kod_clean"], home_helper_names)
    ]
    df["gosc"] = [
        map_team_name(code, helper)
        for code, helper in zip(df["gosc_kod_clean"], away_helper_names)
    ]

    # nierozpoznane kody
    all_codes = set(df["gospodarz_kod_clean"].dropna()) | set(df["gosc_kod_clean"].dropna())
    unknown_codes = sorted([
        code for code in all_codes
        if code not in TEAM_CODE_MAP and len(str(code)) <= 5 and " " not in str(code)
    ])

    if unknown_codes:
        print("⚠️ Nierozpoznane kody drużyn:")
        for code in unknown_codes:
            print(f"   - {code}")
        print()
    else:
        print("✅ Wszystkie kody drużyn zostały poprawnie rozpoznane.\n")

    columns_to_drop = []

    for col in DUPLICATE_FLASH_COLUMNS + DUPLICATE_API_ALIAS_COLUMNS + TEAM_HELPER_COLUMNS:
        if col in df.columns:
            columns_to_drop.append(col)

    # odbiory - tylko jeśli identyczne z API, to usuwamy skuteczne fs
    if columns_are_identical(df, "odbiory_gosp", "odbiory_sk_gosp"):
        columns_to_drop.append("odbiory_sk_gosp")
        print("ℹ️ Usuwam 'odbiory_sk_gosp' - identyczne z API 'odbiory_gosp'")
    else:
        print("ℹ️ Zostawiam 'odbiory_sk_gosp' - nie jest identyczne z API 'odbiory_gosp'")

    if columns_are_identical(df, "odbiory_gosc", "odbiory_sk_gosc"):
        columns_to_drop.append("odbiory_sk_gosc")
        print("ℹ️ Usuwam 'odbiory_sk_gosc' - identyczne z API 'odbiory_gosc'")
    else:
        print("ℹ️ Zostawiam 'odbiory_sk_gosc' - nie jest identyczne z API 'odbiory_gosc'")

    columns_to_drop = sorted(set(col for col in columns_to_drop if col in df.columns))

    print()
    print(f"Kolumny do usunięcia: {len(columns_to_drop)}")
    for col in columns_to_drop:
        print(f" - {col}")

    df = df.drop(columns=columns_to_drop, errors="ignore").copy()

    rename_existing = {old: new for old, new in RENAME_MAP.items() if old in df.columns}
    df = df.rename(columns=rename_existing).copy()

    preferred_order = [
        "match_id",
        "kolejka",
        "data_meczu",
        "gospodarz",
        "gosc",
        "gospodarz_kod",
        "gosc_kod",
        "gole_gosp",
        "gole_gosc",

        "posiadanie_gosp",
        "posiadanie_gosc",
        "strzaly_gosp",
        "strzaly_gosc",
        "celne_gosp",
        "celne_gosc",
        "strzaly_zablokowane_gosp",
        "strzaly_zablokowane_gosc",
        "strzaly_niecelne_gosp",
        "strzaly_niecelne_gosc",
        "rozne_gosp",
        "rozne_gosc",
        "faule_gosp",
        "faule_gosc",
        "spalone_gosp",
        "spalone_gosc",
        "zk_gosp",
        "zk_gosc",
        "czk_gosp",
        "czk_gosc",
        "druga_zk_gosp",
        "druga_zk_gosc",
        "dosrodkowania_gosp",
        "dosrodkowania_gosc",
        "dosrodkowania_celne_gosp",
        "dosrodkowania_celne_gosc",
        "odbiory_gosp",
        "odbiory_gosc",
        "podania_gosp",
        "podania_gosc",
        "podania_celne_gosp",
        "podania_celne_gosc",

        "xg_gosp",
        "xg_gosc",
        "xgot_gosp",
        "xgot_gosc",
        "xa_gosp",
        "xa_gosc",
        "wielkie_szanse_gosp",
        "wielkie_szanse_gosc",
        "strzaly_pk_gosp",
        "strzaly_pk_gosc",
        "strzaly_spoza_pk_gosp",
        "strzaly_spoza_pk_gosc",
        "kontakty_pk_gosp",
        "kontakty_pk_gosc",
        "rzuty_wolne_gosp",
        "rzuty_wolne_gosc",
        "pojedynki_gosp",
        "pojedynki_gosc",
        "wybicia_gosp",
        "wybicia_gosc",
        "przechwyty_gosp",
        "przechwyty_gosc",
        "bledy_strzal_gosp",
        "bledy_strzal_gosc",
        "bledy_gol_gosp",
        "bledy_gol_gosc",
        "obrony_bramkarza_gosp",
        "obrony_bramkarza_gosc",
        "xgot_przeciw_gosp",
        "xgot_przeciw_gosc",
        "zapobiegniecia_gosp",
        "zapobiegniecia_gosc",

        "dlugie_podania_celne_gosp",
        "dlugie_podania_gosp",
        "dlugie_podania_celne_gosc",
        "dlugie_podania_gosc",
        "podania_w_strefe_obrony_przeciwnika_celne_gosp",
        "podania_w_strefe_obrony_przeciwnika_gosp",
        "podania_w_strefe_obrony_przeciwnika_celne_gosc",
        "podania_w_strefe_obrony_przeciwnika_gosc",
        "odbiory_skuteczne_fs_gosp",
        "proby_odbioru_fs_gosp",
        "odbiory_skuteczne_fs_gosc",
        "proby_odbioru_fs_gosc",

        "flash_id",
        "flash_url",
    ]

    existing_preferred = [col for col in preferred_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in existing_preferred]
    df = df[existing_preferred + remaining_columns].copy()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"Kolumny po      : {len(df.columns)}")
    print(f"Plik wyjściowy  : {OUTPUT_CSV}")
    print()
    print("Końcowe kolumny:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:>3}. {col}")

    print()
    print("✅ Gotowe.")


if __name__ == "__main__":
    main()