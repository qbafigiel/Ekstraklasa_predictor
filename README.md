# Ekstraklasa Predictor

Narzędzie do predykcji wyników meczów PKO BP Ekstraklasy.

Predykcje obejmują: gole (1X2, BTTS, Over/Under), kartki, rzuty rożne, faule, strzały.  
Docelowo: wybór 7-8 value betów na weekend.

---

## Stack technologiczny

- Python 3.12
- Streamlit (interfejs webowy / PWA)
- SQLite (baza danych)
- Playwright + BeautifulSoup (scraping)
- pandas, scipy (analiza i model)

---

## Struktura projektu

```
EKSTRAKLASA_PREDICTOR/
│
├── app.py                          # Główna aplikacja Streamlit
├── README.md                       # Ten plik
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── api/                    # Dane źródłowe z API ekstraklasy.org
│   │   │   ├── mecze_2023_24.csv
│   │   │   ├── mecze_2024_25.csv
│   │   │   └── mecze_2025_26.csv
│   │   └── flash/                  # Dane źródłowe z Flashscore
│   │       ├── flash_2023_24.csv
│   │       ├── flash_2024_25.csv
│   │       └── flash_2025_26_druzyny.csv
│   │
│   ├── processed/                  # Dane przetworzone
│   │   └── czyste_2025_26.csv
│   │
│   └── reports/                    # Raporty diagnostyczne
│       └── audyt_brakow_raport.csv
│
├── scripts/
│   ├── scraping/                   # Aktywne scrapery
│   │   ├── scraper_api.py
│   │   ├── scraper_api_historia.py
│   │   ├── scraper_flashscore.py
│   │   ├── scraper_podania.py
│   │   └── scraper_druzyny.py
│   │
│   ├── cleaning/                   # Łączenie i czyszczenie danych
│   │   ├── polacz_i_weryfikuj.py
│   │   └── czysc_polaczone_2025_26.py
│   │
│   └── audit/                      # Narzędzia diagnostyczne
│       ├── audyt_brakow.py
│       └── wypisz_kolumny.py
│
├── archive/                        # Archiwum (nieaktywne)
│   ├── legacy_scripts/             # Starsze wersje skryptów
│   ├── one_off/                    # Skrypty jednorazowe
│   ├── tests/                      # Wczesne testy / debugowanie
│   └── legacy_data/                # Pliki pośrednie z wcześniejszych etapów
│
├── db/                             # Baza SQLite (jeszcze nie wdrożona)
├── docs/                           # Dokumentacja projektu
└── pages/                          # Streamlit pages
```

---

## Źródła danych

### API ekstraklasy.org (główne)

- URL: `https://production-umpire-api.ekstraklasa.tisagroup.ch/api/v3/`
- Token: pobierany dynamicznie przez Playwright (sesyjny)
- Pokrycie: 3 sezony × 306 meczów = **918 meczów, 0 braków**
- Statystyki bazowe: gole, strzały, posiadanie, kartki, faule, rożne, podania, dośrodkowania, odbiory

### Flashscore (warstwa premium)

- URL: `https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa-{rok}/wyniki/`
- Scraping: Firefox headless (Chromium niestabilny na Windows)
- Pokrycie advanced stats różne per sezon:
  - 2025/26 — prawie pełne (xG, xGOT, xA, wielkie szanse, kontakty PK itd.)
  - 2024/25 — częściowe (xG OK, reszta zmienna)
  - 2023/24 — głównie xG (~63%), reszta z dużymi dziurami

---

## Decyzje architektoniczne

1. **API jako fundament modelu** — kompletne dane dla 3 sezonów
2. **Flash jako warstwa korekcyjna** — opcjonalna, dla wzbogacenia (głównie xG)
3. **Wagi sezonowe**:
   - 2025/26 → 1.0
   - 2024/25 → 0.7
   - 2023/24 → 0.4
4. **Osobne submodele per rynek** — gole, kartki, rożne, faule, strzały
5. **SQLite jako baza lokalna** (bez serwera, prosto)

---

## Stan projektu

### ✅ Zakończone

- [x] Scraping API (3 sezony)
- [x] Scraping Flashscore (3 sezony)
- [x] Łączenie i czyszczenie danych (sezon 2025/26)
- [x] Audyt braków
- [x] Reorganizacja struktury projektu

### 🔄 W trakcie

- [ ] Czyszczenie sezonów 2023/24 i 2024/25
- [ ] Budowa bazy SQLite

### 📋 Planowane

- [ ] Model predykcji (Dixon-Coles dla goli, Poisson dla kartek/rożnych)
- [ ] Statystyki sędziowskie
- [ ] Backtesting
- [ ] Interfejs Streamlit z wyborem value betów

---

## Uruchomienie aplikacji

```bash
cd D:\projects\Ekstraklasa_predictor
streamlit run app.py
```

Aplikacja dostępna pod: `http://localhost:8501`

---

## Autor

Projekt prywatny, faza rozwoju.
