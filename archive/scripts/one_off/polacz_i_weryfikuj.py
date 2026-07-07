import pandas as pd
import numpy as np

# Słownik mapowania: fragment URL -> oficjalny kod z API
MAPPING_DRUZYN = {
    "lech-poznan": "LPO", "legia-warszawa": "LEG", "jagiellonia-bialystok": "JAG",
    "rakow-czestochowa": "RCZ", "pogon-szczecin": "POG", "gornik-zabrze": "GOR",
    "zaglebie-lubin": "ZAG", "cracovia": "CRA", "wisla-plock": "WPL",
    "piast-gliwice": "PIA", "widzew-lodz": "WID", "gks-katowice": "GKS",
    "radomiak-radom": "RAD", "radomiak": "RAD", "motor-lublin": "MOT",
    "korona-kielce": "KOR", "arka-gdynia": "ARK", "lechia-gdansk": "LGD",
    "nieciecza": "BBT", "bruk-bet-termalica": "BBT", "stal-mielec": "STM",
    "warta-poznan": "WAR", "gornik-leczna": "GLE", "puszcza-niepolomice": "PUS",
    "puszcza": "PUS", "miedz-legnica": "MIE", "slask-wroclaw": "SLK",
    "lks-lodz": "LKS", "ruch-chorzow": "RCH",
}

def wyciagnij_druzyny_z_url(url):
    """Zwraca kody obu drużyn występujących w URL."""
    if pd.isna(url):
        return None, None
    try:
        czesc = str(url).split("/mecz/pilka-nozna/")[1]
        segmenty = czesc.split("/")
        if len(segmenty) >= 2:
            g1 = next((kod for klucz, kod in MAPPING_DRUZYN.items() if klucz in segmenty[0]), None)
            g2 = next((kod for klucz, kod in MAPPING_DRUZYN.items() if klucz in segmenty[1]), None)
            return g1, g2
    except:
        pass
    return None, None

# 1. Wczytanie plików
print("Wczytywanie plików...")
df_api = pd.read_csv('data/mecze_2025_26.csv')
df_flash = pd.read_csv('data/flash_2025_26_druzyny.csv')

# 2. Wyznaczenie dokładnych kodów gospodarza i gościa dla Flashscore
print("Określanie gospodarza i gościa w danych Flashscore...")
gospodarze_flash = []
goscie_flash = []

for idx, row in df_flash.iterrows():
    # Wyciągamy obie drużyny z URL
    team_a, team_b = wyciagnij_druzyny_z_url(row['url'])
    
    # Wyciągamy kod z początku gosp_nazwa (np. "BBT" z "BBT 3-2 GDA | Bruk-Bet T.")
    prefiks_gosp = str(row['gosp_nazwa'])[:3].strip()
    
    # Mapujemy GDA na LGD jeśli występuje (częsta rozbieżność w skrótach Lechii Gdańsk)
    if prefiks_gosp == "GDA":
        prefiks_gosp = "LGD"
        
    if prefiks_gosp in [team_a, team_b]:
        gosp = prefiks_gosp
        gosc = team_b if prefiks_gosp == team_a else team_a
    else:
        # Fallback jeśli prefiks nie pasuje
        gosp = team_a
        gosc = team_b
        
    gospodarze_flash.append(gosp)
    goscie_flash.append(gosc)

df_flash['gosp_kod'] = gospodarze_flash
df_flash['gosc_kod'] = goscie_flash

# Ujednolicenie kolejek do typu int
df_api['kolejka'] = df_api['kolejka'].astype(int)
df_flash['kolejka_flash'] = df_flash['kolejka_flash'].fillna(0).astype(int)

# 3. Łączenie danych (po Gospodarzu, Gościu i Kolejce)
print("Łączenie plików (LEFT JOIN po gospodarz + gosc + kolejka)...")
df_merged = df_api.merge(
    df_flash,
    left_on=['gospodarz', 'gosc', 'kolejka'],
    right_on=['gosp_kod', 'gosc_kod', 'kolejka_flash'],
    how='left',
    suffixes=('', '_flash')
)

# 4. Zapis połączonego pliku
plik_wynikowy = 'data/pelne_2025_26.csv'
df_merged.to_csv(plik_wynikowy, index=False, encoding='utf-8-sig')
print(f"Zapisano połączone dane do: {plik_wynikowy} ({len(df_merged)} wierszy)")

# 5. GENEROWANIE LOSOWEJ PRÓBKI DO WERYFIKACJI
print("\n" + "="*70)
print("LOSOWA PRÓBKA 5 MECZÓW DO RĘCZNEJ WERYFIKACJI")
print("="*70)

# Filtrujemy tylko te mecze, które pomyślnie połączyły się z Flashscore (mają uzupełnione xG)
df_polaczone = df_merged[df_merged['xg_gosp'].notna()]

if len(df_polaczone) >= 5:
    probka = df_polaczone.sample(5, random_state=np.random.randint(1, 1000))
    for i, (_, row) in enumerate(probka.iterrows(), 1):
        print(f"\n[{i}] Kolejka {row['kolejka']}: {row['gospodarz']} vs {row['gosc']}")
        print(f"    Gole (API): {row['gole_gosp']} - {row['gole_gosc']}")
        print(f"    xG (Flashscore): {row['xg_gosp']} - {row['xg_gosc']}")
        print(f"    Wielkie szanse (Flashscore): {row['wielkie_szanse_gosp']} - {row['wielkie_szanse_gosc']}")
        print(f"    Link do weryfikacji: {row['url']}")
else:
    print("Brak połączonych meczów do wylosowania próbki. Sprawdź poprawność danych.")