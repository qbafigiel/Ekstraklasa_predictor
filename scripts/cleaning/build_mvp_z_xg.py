from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WAGI_SEZONOWE = {
    "2023_24": 0.4,
    "2024_25": 0.7,
    "2025_26": 1.0,
}

# kod -> pełna nazwa (zgodnie z handoff)
TEAM_CODE_MAP = {
    "LPO": "Lech Poznań",
    "LEG": "Legia Warszawa",
    "JAG": "Jagiellonia Białystok",
    "RCZ": "Raków Częstochowa",
    "POG": "Pogoń Szczecin",

    # Górnik - API występuje jako GÓR
    "GOR": "Górnik Zabrze",
    "GÓR": "Górnik Zabrze",

    "ZAG": "Zagłębie Lubin",
    "CRA": "Cracovia",

    # Wisła Płock - API występuje jako WPŁ
    "WPL": "Wisła Płock",
    "WPŁ": "Wisła Płock",

    "PIA": "Piast Gliwice",
    "WID": "Widzew Łódź",
    "GKS": "GKS Katowice",
    "RAD": "Radomiak Radom",
    "MOT": "Motor Lublin",
    "KOR": "Korona Kielce",
    "ARK": "Arka Gdynia",
    "LGD": "Lechia Gdańsk",
    "BBT": "Bruk-Bet Termalica Nieciecza",

    # historyczne / starsze sezony
    "STM": "Stal Mielec",
    "WAR": "Warta Poznań",
    "GLE": "Górnik Łęczna",

    # Puszcza - API występuje jako PUN
    "PUS": "Puszcza Niepołomice",
    "PUN": "Puszcza Niepołomice",

    "MIE": "Miedź Legnica",

    # Śląsk - API występuje jako ŚLĄ
    "SLK": "Śląsk Wrocław",
    "ŚLĄ": "Śląsk Wrocław",

    "LKS": "ŁKS Łódź",
    "ŁKS": "ŁKS Łódź",
    "RCH": "Ruch Chorzów",
}

POLISH_CHAR_MAP = str.maketrans({
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
})


def normalize_text(s: str | None) -> str | None:
    if s is None or pd.isna(s):
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.translate(POLISH_CHAR_MAP).upper()
    t = " ".join(t.split())
    return t


def extract_team_name_from_flash_field(x: str | None) -> str | None:
    """
    Flash home name bywa typu: "GOR 1-1 KOR | Górnik Zabrze"
    Away name bywa zwykle już pełną nazwą: "Korona Kielce"
    Ta funkcja zwraca pełną nazwę drużyny.
    """
    if x is None or pd.isna(x):
        return None
    t = str(x).strip()
    if "|" in t:
        # bierzemy część po ostatnim '|'
        t = t.split("|")[-1].strip()
    return t if t else None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Nie udało się odczytać: {path}")


def build_name_to_code() -> dict[str, str]:
    """
    Mapowanie: pełna nazwa drużyny -> dokładny kod używany w API.
    To mapowanie służy do MERGE z API, więc musi być 1:1 zgodne z realnymi kodami API.
    """
    return {
        normalize_text("Lech Poznań"): "LPO",
        normalize_text("Legia Warszawa"): "LEG",
        normalize_text("Jagiellonia Białystok"): "JAG",
        normalize_text("Raków Częstochowa"): "RCZ",
        normalize_text("Pogoń Szczecin"): "POG",
        normalize_text("Górnik Zabrze"): "GÓR",
        normalize_text("Zagłębie Lubin"): "ZAG",
        normalize_text("Cracovia"): "CRA",
        normalize_text("Wisła Płock"): "WPŁ",
        normalize_text("Piast Gliwice"): "PIA",
        normalize_text("Widzew Łódź"): "WID",
        normalize_text("GKS Katowice"): "GKS",
        normalize_text("Radomiak Radom"): "RAD",
        normalize_text("Motor Lublin"): "MOT",
        normalize_text("Korona Kielce"): "KOR",
        normalize_text("Arka Gdynia"): "ARK",
        normalize_text("Lechia Gdańsk"): "LGD",
        normalize_text("Bruk-Bet Termalica Nieciecza"): "BBT",
        normalize_text("Bruk-Bet T."): "BBT",
        normalize_text("Bruk-Bet Termalica"): "BBT",

        # historyczne
        normalize_text("Stal Mielec"): "STM",
        normalize_text("Warta Poznań"): "WAR",
        normalize_text("Górnik Łęczna"): "GLE",
        normalize_text("Puszcza Niepołomice"): "PUN",
        normalize_text("Miedź Legnica"): "MIE",
        normalize_text("Śląsk Wrocław"): "ŚLĄ",
        normalize_text("ŁKS Łódź"): "ŁKS",
        normalize_text("Ruch Chorzów"): "RCH",
    }


def parse_kolejka_int(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # zostawiamy NaN jako NaN (UInt/Int64 jest ok), ale do merge potrzebujemy int.
    # W praktyce w tych plikach kolejka jest poprawna, więc robimy floor po konwersji.
    return s.astype(int)


API_COLUMNS = [
    "match_id",
    "kolejka",
    "data_meczu",
    "gospodarz",
    "gosc",
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
    "podania_celne_gosp",
    "podania_gosc",
    "podania_celne_gosc",
]


MVP_COLUMNS_FINAL = [
    "match_id",
    "sezon",
    "waga_sezonu",
    "kolejka",
    "data_meczu",
    "gospodarz",
    "gosc",
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
    # xG tylko dla 24/25 i 25/26, w 23/24 zostanie NULL
    "xg_gosp",
    "xg_gosc",
    # debug/trace
    "flash_id",
    "flash_url",
]


def map_api_codes_to_names(df_api: pd.DataFrame) -> pd.DataFrame:
    df = df_api.copy()

    df["gospodarz"] = (
        df["gospodarz"]
        .astype(str)
        .str.strip()
        .map(TEAM_CODE_MAP)
        .fillna(df["gospodarz"])
    )

    df["gosc"] = (
        df["gosc"]
        .astype(str)
        .str.strip()
        .map(TEAM_CODE_MAP)
        .fillna(df["gosc"])
    )

    return df


def build_xg_for_flash_season(sezon: str) -> pd.DataFrame:
    """
    Buduje df do merge z API: kody drużyn + kolejka + xG + flash_id + url.
    Obsługuje dwa przypadki:
    - 2024_25: jeden plik flash_2024_25_druzyny.csv (ma wszystko)
    - 2025_26: jeden plik flash_2025_26_druzyny.csv (ma wszystko)
    """
    flash_druzyny_path = ROOT / "data" / "raw" / "flash" / f"flash_{sezon}_druzyny.csv"

    if not flash_druzyny_path.exists():
        raise FileNotFoundError(f"Brak pliku: {flash_druzyny_path}")

    df = read_csv_flexible(flash_druzyny_path)

    # Sprawdź wymagane kolumny
    needed = {"flash_id", "xg_gosp", "xg_gosc", "url", "gosp_nazwa", "gosc_nazwa", "kolejka_flash"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{flash_druzyny_path.name}: brakuje kolumn: {missing}")

    name_to_code = build_name_to_code()

    df["gospodarz_nazwa_flash_clean"] = df["gosp_nazwa"].apply(extract_team_name_from_flash_field)
    df["gosc_nazwa_flash_clean"] = df["gosc_nazwa"].apply(extract_team_name_from_flash_field)

    df["gospodarz_kod_flash"] = df["gospodarz_nazwa_flash_clean"].apply(
        lambda n: name_to_code.get(normalize_text(n))
    )
    df["gosc_kod_flash"] = df["gosc_nazwa_flash_clean"].apply(
        lambda n: name_to_code.get(normalize_text(n))
    )

    # Diagnostyka nierozpoznanych nazw
    nieznane_gosp = df[df["gospodarz_kod_flash"].isna()]["gospodarz_nazwa_flash_clean"].dropna().unique()
    nieznane_gosc = df[df["gosc_kod_flash"].isna()]["gosc_nazwa_flash_clean"].dropna().unique()
    if len(nieznane_gosp) > 0:
        print(f"  ⚠️  Nierozpoznani gospodarze: {sorted(nieznane_gosp)}")
    if len(nieznane_gosc) > 0:
        print(f"  ⚠️  Nierozpoznani goście: {sorted(nieznane_gosc)}")

    df["kolejka_flash"] = parse_kolejka_int(df["kolejka_flash"])

    out = df[[
        "gospodarz_kod_flash",
        "gosc_kod_flash",
        "kolejka_flash",
        "xg_gosp",
        "xg_gosc",
        "flash_id",
        "url",
    ]].rename(columns={"url": "flash_url"})

    return out


def merge_api_with_xg(sezon: str, waga_sezonu: float, xg_df: pd.DataFrame | None) -> pd.DataFrame:
    api_path = ROOT / "data" / "raw" / "api" / f"mecze_{sezon}.csv"
    df_api = read_csv_flexible(api_path)

    df_api["sezon"] = sezon.replace("_", "/")
    df_api["waga_sezonu"] = waga_sezonu

    df_api["kolejka"] = parse_kolejka_int(df_api["kolejka"])

    # Normalizacja: wymuszamy typy int dla merge
    if xg_df is None:
        df_api["xg_gosp"] = None
        df_api["xg_gosc"] = None
        df_api["flash_id"] = None
        df_api["flash_url"] = None
        df_api = map_api_codes_to_names(df_api)
        df_out = df_api[MVP_COLUMNS_FINAL].copy()
        return df_out

    # Merge: API (gospodarz, gosc, kolejka) z Flash (gospodarz_kod_flash, gosc_kod_flash, kolejka_flash)
    df_merge_normal = df_api.merge(
        xg_df,
        left_on=["gospodarz", "gosc", "kolejka"],
        right_on=["gospodarz_kod_flash", "gosc_kod_flash", "kolejka_flash"],
        how="left",
        suffixes=("", "_xg"),
    )

    cnt_normal = df_merge_normal["xg_gosp"].notna().sum()
    total = len(df_merge_normal)

    # Dodatkowy test orientacji (tylko diagnostyka / ewentualna korekta)
    df_merge_swapped = df_api.merge(
        xg_df,
        left_on=["gospodarz", "gosc", "kolejka"],
        right_on=["gosc_kod_flash", "gospodarz_kod_flash", "kolejka_flash"],
        how="left",
        suffixes=("", "_xg"),
    )
    cnt_swapped = df_merge_swapped["xg_gosc"].notna().sum()  # bo w swapped to "xg_gosc" jest xG dla API-gospodarza

    use_swapped = cnt_swapped > cnt_normal and cnt_swapped >= int(0.9 * total)
    use_normal = not use_swapped

    print(f"  {sezon}: matchy z xG normal = {cnt_normal}/{total}, swapped = {cnt_swapped}/{total} -> wybór: {'swapped' if use_swapped else 'normal'}")

    if use_swapped:
        # xg dla API gospodarza = flash xg_gosc, dla API gościa = flash xg_gosp
        df_out = df_merge_swapped.copy()
        df_out["xg_gosp"] = df_out["xg_gosc"]
        df_out["xg_gosc"] = df_out["xg_gosp"]
        # flash_id/url są już z wiersza trafionego (bez zmiany)
    else:
        df_out = df_merge_normal.copy()

    # Posprzątanie kolumn pomocniczych po merge
    for c in ["gospodarz_kod_flash", "gosc_kod_flash", "kolejka_flash"]:
        if c in df_out.columns:
            df_out = df_out.drop(columns=[c])

    # Mapujemy kody API na pełne nazwy (tylko do MVP kolumn finalnych)
    df_out = map_api_codes_to_names(df_out)

    # Upewniamy się że flash_url jest pod właściwą nazwą
    if "flash_url" not in df_out.columns and "flash_url_xg" in df_out.columns:
        df_out = df_out.rename(columns={"flash_url_xg": "flash_url"})

    df_out = df_out[MVP_COLUMNS_FINAL].copy()
    return df_out


def main():
    print("BUILD MVP DATASET: API (rdzeń) + xG z Flash tylko dla 24/25 i 25/26")
    print("-"*100)

    SEZONY = ["2023_24", "2024_25", "2025_26"]

    all_out = []

    for sezon in SEZONY:
        w = WAGI_SEZONOWE[sezon]
        print(f"\nSEZON: {sezon} (waga={w})")

        if sezon in ["2024_25", "2025_26"]:
            xg_df = build_xg_for_flash_season(sezon)
        else:
            xg_df = None

        df_mvp = merge_api_with_xg(sezon, w, xg_df)

        out_path = ROOT / "data" / "processed" / f"mvp_{sezon}.csv"
        df_mvp.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Zapis: {out_path} ({len(df_mvp)} wierszy, {len(df_mvp.columns)} kolumn)")

        all_out.append(df_mvp)

    df_all = pd.concat(all_out, ignore_index=True)
    merged_path = ROOT / "data" / "processed" / "mvp_merged_2023_26.csv"
    df_all.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Zapis zbiorczy: {merged_path} ({len(df_all)} wierszy)")

    # mały audyt xG
    for col in ["xg_gosp", "xg_gosc"]:
        na = df_all[col].isna().sum()
        print(f"xG brak: {col}: {na}/{len(df_all)}")


if __name__ == "__main__":
    main()