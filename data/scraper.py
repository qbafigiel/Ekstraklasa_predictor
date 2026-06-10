import requests
from bs4 import BeautifulSoup
import pandas as pd

def pobierz_statystyki_druzyn():
    url = "https://www.ekstraklasa.org/statystyki/druzynowe"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Błąd pobierania: {e}")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    tabela = soup.find("table")
    if not tabela:
        print("Nie znaleziono tabeli na stronie")
        return None
    
    kolumny = []
    for th in tabela.find_all("th"):
        kolumny.append(th.get_text(strip=True))
    
    wiersze = []
    for tr in tabela.find_all("tr")[1:]:
        wiersz = []
        for td in tr.find_all("td"):
            wiersz.append(td.get_text(strip=True))
        if wiersz:
            wiersze.append(wiersz)
    
    if not kolumny or not wiersze:
        print("Brak danych w tabeli")
        return None
    
    df = pd.DataFrame(wiersze, columns=kolumny[:len(wiersze[0])])
    return df

if __name__ == "__main__":
    df = pobierz_statystyki_druzyn()
    if df is not None:
        print(df.head())
        print(f"\nKolumny: {list(df.columns)}")