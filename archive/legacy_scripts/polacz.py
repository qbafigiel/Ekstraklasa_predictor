import pandas as pd

# Słownik mapowania: fragment URL Flashscore -> kod drużyny z API
MAPPING_DRUZYN = {
    "lech-poznan": "LPO",
    "legia-warszawa": "LEG",
    "jagiellonia-bialystok": "JAG",
    "rakow-czestochowa": "RCZ",
    "pogon-szczecin": "POG",
    "gornik-zabrze": "GOR",
    "zaglebie-lubin": "ZAG",
    "cracovia": "CRA",
    "wisla-plock": "WPL",
    "piast-gliwice": "PIA",
    "widzew-lodz": "WID",
    "gks-katowice": "GKS",
    "radomiak-radom": "RAD",
    "radomiak": "RAD",
    "motor-lublin": "MOT",
    "korona-kielce": "KOR",
    "arka-gdynia": "ARK",
    "lechia-gdansk": "LGD",
    "nieciecza": "BBT",
    "bruk-bet-termalica": "BBT",
    "stal-mielec": "STM",
    "warta-poznan": "WAR",
    "gornik-leczna": "GLE",
    "puszcza-niepolomice": "PUS",
    "puszcza": "PUS",           # skrócony slug na Flashscore
    "miedz-legnica": "MIE",
    "slask-wroclaw": "SLK",
    "lks-lodz": "LKS",
    "ruch-chorzow": "RCH",
}


def wyciagnij_kod_druzyny(url_fragment):
    """Wyciąga kod drużyny z fragmentu URL Flashscore."""
    for klucz, kod in MAPPING_DRUZYN.items():
        if klucz in url_fragment:
            return kod
    return None


def wyciagnij_druzyny_z_url(url):
    """
    URL format: /mecz/pilka-nozna/{gospodarz-ID}/{gosc-ID}/...
    Pierwszy segment = gospodarz, drugi = gość.
    """
    try:
        czesc = url.split("/mecz/pilka-nozna/")[1]
        segmenty = czesc.split("/")
        if len(segmenty) >= 2:
            gosp_slug = segmenty[0]
            gosc_slug = segmenty[1]
            gosp_kod = wyciagnij_kod_druzyny(gosp_slug)
            gosc_kod = wyciagnij_kod_druzyny(gosc_slug)
            return gosp_kod, gosc_kod
    except Exception:
        pass
    return None, None


def polacz_sezony():
    SEZONY = [
        {
            "api": "data/mecze_2025_26.csv",
            "flash": "data/flash_2025_26.csv",
            "wynik": "data/pelne_2025_26.csv",
            "nazwa": "2025/26"
        },
        {
            "api": "data/mecze_2024_25.csv",
            "flash": "data/flash_2024_25.csv",
            "wynik": "data/pelne_2024_25.csv",
            "nazwa": "2024/25"
        },
        {
            "api": "data/mecze_2023_24.csv",
            "flash": "data/flash_2023_24.csv",
            "wynik": "data/pelne_2023_24.csv",
            "nazwa": "2023/24"
        },
    ]

    for sezon in SEZONY:
        print(f"\n=== ŁĄCZENIE SEZONU {sezon['nazwa']} ===")

        try:
            df_api = pd.read_csv(sezon["api"])
            df_flash = pd.read_csv(sezon["flash"])
        except FileNotFoundError as e:
            print(f"  Brak pliku: {e} — pomijam")
            continue

        print(f"  API: {len(df_api)} meczów")
        print(f"  Flashscore: {len(df_flash)} wierszy")

        # Wyciągnij kody drużyn z URL Flashscore
        df_flash["gosp_flash"] = df_flash["url"].apply(
            lambda u: wyciagnij_druzyny_z_url(u)[0] if pd.notna(u) else None
        )
        df_flash["gosc_flash"] = df_flash["url"].apply(
            lambda u: wyciagnij_druzyny_z_url(u)[1] if pd.notna(u) else None
        )

        # Sprawdź ile udało się zdekodować
        zdekodowane = df_flash["gosp_flash"].notna().sum()
        print(f"  Zdekodowane drużyny: {zdekodowane}/{len(df_flash)}")

        # Pokaż nierozpoznane drużyny (unikalne slugi)
        nierozpoznane_url = df_flash[df_flash["gosp_flash"].isna()]["url"]
        if len(nierozpoznane_url) > 0:
            print(f"  Nierozpoznane URL ({len(nierozpoznane_url)} meczów):")
            for u in nierozpoznane_url.head(10):
                print(f"    {u}")

        # Sprawdź duplikaty w Flashscore po kluczu gospodarz+gość
        dupl_flash = df_flash[df_flash.duplicated(subset=["gosp_flash", "gosc_flash"], keep=False)]
        if len(dupl_flash) > 0:
            print(f"  UWAGA: {len(dupl_flash)} zduplikowanych par gospodarz+gość w Flashscore")
            print(f"  Przykłady duplikatów:")
            print(dupl_flash[["gosp_flash", "gosc_flash", "url"]].head(6).to_string())
            # Usuń duplikaty — zachowaj pierwszy (zazwyczaj nowszy/pełniejszy wpis)
            df_flash = df_flash.drop_duplicates(subset=["gosp_flash", "gosc_flash"], keep="first")
            print(f"  Po usunięciu duplikatów: {len(df_flash)} wierszy Flashscore")

        # Połącz po gospodarz + gość
        df_merged = df_api.merge(
            df_flash,
            left_on=["gospodarz", "gosc"],
            right_on=["gosp_flash", "gosc_flash"],
            how="left",
            suffixes=("", "_flash")
        )

        # Weryfikacja: powinniśmy mieć dokładnie tyle wierszy co w API
        if len(df_merged) != len(df_api):
            print(f"  BŁĄD: po merge mamy {len(df_merged)} wierszy zamiast {len(df_api)}!")
            print(f"  Szukam przyczyny — pary gospodarz+gość z wielokrotnymi dopasowaniami:")
            problemy = df_merged[df_merged.duplicated(subset=["gospodarz", "gosc", "kolejka"], keep=False)]
            print(problemy[["kolejka", "gospodarz", "gosc"]].drop_duplicates().head(10).to_string())
        else:
            print(f"  OK: {len(df_merged)} wierszy (zgodne z API)")

        # Sprawdź wyniki łączenia
        polaczone = df_merged["xg_gosp"].notna().sum() if "xg_gosp" in df_merged.columns else 0
        print(f"  Połączono (mają xG): {polaczone}/{len(df_api)} meczów")

        # Usuń pomocnicze kolumny
        df_merged = df_merged.drop(
            columns=["gosp_flash", "gosc_flash", "url"],
            errors="ignore"
        )

        df_merged.to_csv(sezon["wynik"], index=False, encoding="utf-8-sig")
        print(f"  Zapisano do: {sezon['wynik']}")

        # Pokaż przykład
        print(f"\n  Przykład (pierwsze 3 mecze):")
        kolumny_pokaz = ["kolejka", "gospodarz", "gosc", "gole_gosp", "gole_gosc",
                         "xg_gosp", "xg_gosc", "wielkie_szanse_gosp", "wielkie_szanse_gosc"]
        kolumny_dostepne = [k for k in kolumny_pokaz if k in df_merged.columns]
        print(df_merged[kolumny_dostepne].head(3).to_string())


def weryfikuj_polaczenie(plik):
    """Szczegółowa weryfikacja połączonego pliku."""
    print(f"\n=== WERYFIKACJA: {plik} ===")
    try:
        df = pd.read_csv(plik)
    except Exception:
        print("  Brak pliku")
        return

    print(f"  Wierszy: {len(df)}")
    print(f"  Kolumn: {len(df.columns)}")

    braki = df.isnull().sum()
    braki = braki[braki > 0].sort_values(ascending=False)
    if len(braki) > 0:
        print(f"\n  Brakujące wartości:")
        print(braki.to_string())
    else:
        print(f"  Brakujące wartości: BRAK — wszystko kompletne!")

    # Sprawdź duplikaty
    duplikaty = df.duplicated(subset=["gospodarz", "gosc", "kolejka"]).sum()
    print(f"\n  Duplikaty (gospodarz+gość+kolejka): {duplikaty}")

    # Statystyki xG
    if "xg_gosp" in df.columns:
        bez_xg = df["xg_gosp"].isna().sum()
        print(f"\n  Mecze z xG: {len(df) - bez_xg}/{len(df)}")
        xg_data = df["xg_gosp"].dropna()
        if len(xg_data) > 0:
            print(f"  xG gospodarz — średnia: {xg_data.mean():.2f}, "
                  f"min: {xg_data.min():.2f}, max: {xg_data.max():.2f}")
            xg_data_g = df["xg_gosc"].dropna()
            print(f"  xG gość     — średnia: {xg_data_g.mean():.2f}, "
                  f"min: {xg_data_g.min():.2f}, max: {xg_data_g.max():.2f}")


if __name__ == "__main__":
    polacz_sezony()

    print("\n\n=== WERYFIKACJA PLIKÓW WYNIKOWYCH ===")
    for plik in ["data/pelne_2025_26.csv", "data/pelne_2024_25.csv", "data/pelne_2023_24.csv"]:
        weryfikuj_polaczenie(plik)