import pandas as pd

# Słownik mapowania: fragment URL Flashscore -> kod drużyny
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

def wyciagnij_kody(url):
    """Wyciąga kody drużyn (lub surowe slugi, jeśli nie ma w słowniku) z URL."""
    if pd.isna(url):
        return None, None
        
    try:
        czesc = str(url).split("/mecz/pilka-nozna/")[1]
        segmenty = czesc.split("/")
        
        if len(segmenty) >= 2:
            gosp_slug = segmenty[0]
            gosc_slug = segmenty[1]
            
            gosp_kod = next((kod for klucz, kod in MAPPING_DRUZYN.items() if klucz in gosp_slug), gosp_slug)
            gosc_kod = next((kod for klucz, kod in MAPPING_DRUZYN.items() if klucz in gosc_slug), gosc_slug)
            
            return gosp_kod, gosc_kod
    except Exception:
        pass
        
    return None, None

# 1. Wczytanie pliku
plik_wejsciowy = 'data/flash_2025_26.csv'
plik_wyjsciowy = 'data/flash_2025_26_z_druzynami.csv'

print(f"Wczytywanie {plik_wejsciowy}...")
df = pd.read_csv(plik_wejsciowy)

# 2. Utworzenie nowych kolumn 'gospodarz' i 'gosc' na podstawie kolumny 'url'
print("Przetwarzanie adresów URL...")
df[['gospodarz', 'gosc']] = df.apply(lambda row: pd.Series(wyciagnij_kody(row['url'])), axis=1)

# 3. Zapis do nowego pliku
df.to_csv(plik_wyjsciowy, index=False, encoding='utf-8-sig')
print(f"Gotowe! Wynik zapisano w: {plik_wyjsciowy}")