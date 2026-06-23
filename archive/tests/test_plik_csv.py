import pandas as pd

for nazwa_pliku in ['data/mecze_2025_26.csv', 'data/flash_2025_26_druzyny.csv']:    
    try:
        df = pd.read_csv(nazwa_pliku)
        print(f"\n=== PLIK: {nazwa_pliku} ===")
        print(f"Kolumny: {list(df.columns)}")
        print(f"Liczba wierszy: {len(df)}")
        print("Pierwsze 3 wiersze:")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"Błąd przy pliku {nazwa_pliku}: {e}")