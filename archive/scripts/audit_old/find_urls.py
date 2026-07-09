import os
import pandas as pd
from pathlib import Path

print("Szukam plików CSV z linkami do Flashscore...")
found = False

for root, dirs, files in os.walk("data"):
    for f in files:
        if f.endswith(".csv"):
            path = Path(root) / f
            try:
                # Wczytaj tylko nagłówki i 1 wiersz żeby było szybko
                df = pd.read_csv(path, nrows=1)
                
                # Szukamy kolumn, które mogą zawierać linki
                cols = [c for c in df.columns if "url" in c.lower() or "link" in c.lower() or "flash" in c.lower()]
                
                if cols:
                    found = True
                    print(f"\nPlik: {path}")
                    print(f"Kolumny: {cols}")
                    # Pokaż próbkę danych z tej kolumny
                    for c in cols:
                        print(f"  Przykład z {c}: {df[c].iloc[0]}")
            except Exception:
                pass

if not found:
    print("\nNie znalazłem żadnego pliku CSV z podejrzanymi kolumnami.")