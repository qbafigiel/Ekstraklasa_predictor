Markdown

# EKSTRAKLASA PREDICTOR — HANDOFF

**Data:** 09.07.2026
**Ostatnia sesja:** Scraping ekstraklasa.org (zawodnicy + drużynowe xG)
**Repozytorium:** GitHub private — Ekstraklasa_predictor
**Lokalizacja:** D:\projects\Ekstraklasa_predictor\

---

## 1. STAN MODELU — PRODUKCJA

### Silnik goli (O/U)

- Poisson MLE na golach historycznych
- Prior bayesowski dla beniaminków (K=10)
- Wagi sezonów: 2023/24=0.4, 2024/25=0.7, 2025/26=1.0
- Prior beniaminka: atak=0.7744, obrona=1.0648

### Silnik 1X2 — CURRENT BEST (log-loss 1.0571)

- Poisson(xG) + Softmax Temperature Calibration
- Kalibracja: T=1.8429, bH=0.2982, bD=-0.0163, bA=-0.2819
- Trenowany na 2024/25, testowany na 2025/26

**Ranking modeli:**

Model Log-loss
──────────────────────────────────────────
Benchmark losowy 1.0986
Poisson(gole) surowy 1.0807
Poisson(gole) + kalibracja 1.0647
Poisson(xG) surowy 1.0628
Poisson(xG) + kalibracja 1.0571 🔥 BEST
Poisson(xG) + forma + kalibracja 1.0638 (overfit)
Availability Score r=-0.08 (odrzucony)

---

## 2. STAN DANYCH

### Baza SQLite (`db/ekstraklasa.db`)

**918 meczów** (3 sezony × 306), 6+ tabel.
Kompletność statystyk: **100%** dla wszystkich pól **oprócz xG 2023/24** (Flashscore nie miał wtedy xG).

**Główne tabele:**

- `matches` (918) — statystyki meczów
- `lineups` (37535) — składy z Flashscore
- `match_absences` (4172) — kontuzje/kartki
- `match_coaches` (918)
- `match_substitutions` (8126)
- `player_minutes` (37535)
- `match_availability` (1836) — availability score (odrzucony jako feature)

### Dane z ekstraklasa.org (NOWE — sesja 09.07.2026)

**Zawodnicy per ranking:** `data/raw/ekstraklasa_org/{sezon}/`

- 3 sezony × 46 rankingów = **138 plików CSV**
- Format: `pole_*.csv` (zawodnicy z pola) i `gk_*.csv` (bramkarze)
- Kolumny: pozycja, player_slug, klub_slug, nazwa, wartosc

**Drużynowe xG/xGA:** `data/raw/ekstraklasa_org/druzyny/`

- 6 plików: {sezon}\_druzynowe-xg.csv i xga.csv (3 sezony × 2 metryki)
- Kluczowe dla uzupełnienia braku xG w 2023/24

**Merged tabela zawodników:** `data/processed/zawodnicy_ekstraklasa_org_2025_26.csv`

- **537 zawodników × 49 kolumn** (dla sezonu 2025/26)
- Klucz łączący: `player_slug`
- Dla starszych sezonów jeszcze nie zrobiony merge

### Braki w danych ekstraklasa.org

Niektóre rankingi nie są dostępne dla wszystkich sezonów:

- 2024/25: brak `podania-celne`, `gk-podania-celne`
- 2023/24: brak `dogodne-szanse`, `podania-celne`, `stworzone-dogodne-szanse`, `gk-podania-celne`

Nie problem — mamy `podania` + `celnosc` żeby wyliczyć.

---

## 3. WERYFIKACJA JAKOŚCI DANYCH

**Sanity check xG (sesja 09.07.2026):**

- Suma xG per drużyna 2025/26 z bazy (Flashscore) vs ekstraklasa.org
- Różnice: **0.01 – 0.79 (max ~1.5%)**
- **Średnio poniżej 1%**
- Wniosek: Flashscore xG jest **wiarygodny**

**Audyt bazy meczowej:**

- 17/18 statystyk: 100% pokrycie we wszystkich 3 sezonach
- Jedyny brak: xG w 2023/24

---

## 4. STRUKTURA PROJEKTU

Ekstraklasa*predictor/
├── db/
│ └── ekstraklasa.db (918 meczów, 3 sezony)
├── data/
│ ├── raw/
│ │ ├── api/ (mecze_2023_24.csv, mecze_2024_25.csv, mecze_2025_26.csv)
│ │ ├── flash/ (flash*.csv, flash\_\_druzyny.csv)
│ │ └── ekstraklasa_org/ (NOWE)
│ │ ├── 2023-2024/ (46 CSV)
│ │ ├── 2024-2025/ (46 CSV)
│ │ ├── 2025-2026/ (46 CSV)
│ │ └── druzyny/ (6 CSV)
│ ├── processed/
│ │ ├── mvp_2023_24.csv
│ │ ├── mvp_2024_25.csv
│ │ ├── mvp_2025_26.csv
│ │ ├── mvp_merged_2023_26.csv
│ │ ├── parametry_modelu_gole.json
│ │ ├── priory_beniaminkow.json
│ │ ├── kalibrator_xg_form_best.json
│ │ └── zawodnicy_ekstraklasa_org_2025_26.csv (NOWE)
│ └── reports/
├── docs/
│ ├── data_dictionary.md
│ └── handoff.md
├── scripts/
│ ├── audit/
│ │ ├── audit_matches_completeness.py (NOWY)
│ │ └── check_db.py
│ ├── cleaning/
│ │ └── build_mvp_z_xg.py
│ ├── db/
│ │ └── baza.py
│ ├── model/
│ │ ├── backtesting_gole_v3.py (produkcyjny O/U)
│ │ ├── backtesting_xg_calibrated.py (BEST 1X2)
│ │ ├── kalibrator_1x2.py
│ │ └── model_xg_poisson.py
│ └── scraping/
│ ├── flashscore/
│ │ ├── scrape_lineups.py
│ │ └── backfill_flash_2023_24.py
│ └── ekstraklasa_org/ (NOWY MODUŁ)
│ ├── scrape_rankings.py (46 rankingów per sezon)
│ ├── scrape_druzyny_xg.py (drużynowy xG/xGA)
│ └── merge_rankings.py (buduje pivot tabelę)
└── archive/
├── backtests/
├── data/legacy_data/
└── scripts/
├── audit_old/ (11 plików)
├── diagnostics/
├── experiments/
├── model_old/ (1 plik)
├── one_off/
├── scraping_ekstraklasa_api_debug/ (2 pliki)
├── scraping_ekstraklasa_org_debug/ (18 plików)
└── scraping_flashscore_debug/ (9 plików)

---

## 5. SCRAPING EKSTRAKLASA.ORG — JAK DZIAŁA

### Kluczowe odkrycia sesji 09.07.2026

1. **Ekstraklasa.org ma zakładkę DRUŻYNOWE** (nie tylko INDYWIDUALNE)
2. **URL rankingu drużynowego wymaga `tab=team`:**  
   `https://ekstraklasa.org/statystyki/?tab=team&season=2023-2024&ranking=druzynowe-xg`
3. **URL rankingu zawodników:**  
   `https://ekstraklasa.org/statystyki/?season=2025-2026&ranking=xg`  
   z opcjonalnymi `&category=defensive|passing|other` lub `&typ=bramkarze`
4. **Struktura HTML rekordu (kontener rankingu):**

   ```html
   <div class="grid min-h-20 grid-cols-[44px_1fr_64px]">
     <span>{pozycja}</span>
     <a href="/kluby/{klub}/zawodnik/{gracz}/">{nazwa}</a>
     <span>{wartość}</span>
   </div>

   Filtruj kontenery z klasą min-h-20 (podglądy Top-3 mają min-h-14) Deduplikuj
   po player_slug (każdy rekord jest 2× w HTML: mobile + desktop)
   ```

Wydajność

    46 rankingów per sezon = ~8 minut
    3 sezony = ~24 minuty łącznie
    6 rankingów drużynowych = ~40 sekund

Alternatywa (odrzucona)

    Scraping profili zawodników 1 po 1 (Playwright klika 4 zakładki + zmienia sezon) = ~27s/zawodnik
    448 zawodników = ~3 godziny per sezon
    Zbyt wolne, wybraliśmy scraping rankingów zamiast profili

6. PLAYER VALUE MODEL — PLAN

Cel: Wycenić skład danej drużyny w danym meczu i skorygować lambdy w Poissonie.

Mechanizm:

    Z Flashscore lineups (mamy w bazie) wiemy KTO gra
    Z ekstraklasa.org wiemy ile WART jest każdy gracz (xG/90, xA/90)
    Sumujemy wartość 11 graczy → dzisiejszy skład
    Porównujemy z typowym składem → deficyt/nadwyżka
    Deficyt → korekta lambdy Poissona

Do zrobienia (następna sesja):
KROK 6 — Mapowanie nazwisk Flashscore ↔ ekstraklasa.org

    Flashscore używa nazw z lineups (np. "Bobcek T.")
    Ekstraklasa.org używa player_slug (np. "tomas-bobcek")
    Trzeba zbudować mapowanie (fuzzy match nazwisk?)

KROK 7 — Merge zawodników dla wszystkich sezonów

    Uruchomić merge_rankings.py z argparse --season dla 2024/25 i 2023/24
    Obecnie merged tylko 2025/26

KROK 8 — Player Value Model

    Obliczyć xG/90 dla każdego zawodnika
    Zbudować feature "team_value" per mecz

KROK 9 — Test korelacji z xG drużyny per mecz
KROK 10 — Jeśli sygnał → integracja z Poissonem 7. UŻYCIE MODELU PRODUKCYJNEGO

Python

import numpy as np
from scipy.stats import poisson

MAX_GOLE = 10
CALIBRATOR = {"T": 1.8429, "bH": 0.2982, "bD": -0.0163, "bA": -0.2819}

def softmax(x):
e = np.exp(x - np.max(x))
return e / e.sum()

def predict*match(params, home, away, calibrate=True):
if home not in params["alpha"] or away not in params["alpha"]:
return None
lh = params["mu_h"] * params["alpha"][home] * params["beta"][away]
la = params["mu_a"] * params["alpha"][away] * params["beta"][home]
m = np.zeros((MAX_GOLE, MAX_GOLE))
for i in range(MAX_GOLE):
for j in range(MAX_GOLE):
m[i, j] = poisson.pmf(i, lh) \* poisson.pmf(j, la)
m /= m.sum()
p_home = float(np.sum(np.tril(m, -1)))
p_draw = float(np.sum(np.diag(m)))
p_away = float(np.sum(np.triu(m, 1)))
ou = {}
for prog in [0, 1, 2, 3]:
s = sum(m[i,j] for i in range(MAX_GOLE)
for j in range(MAX_GOLE) if i+j <= prog)
ou[f"under*{prog+1}5"] = s
ou[f"over_{prog+1}5"] = 1 - s
btts = float((1 - poisson.pmf(0, lh)) \* (1 - poisson.pmf(0, la)))
if calibrate:
T = CALIBRATOR["T"]
bH, bD, bA = CALIBRATOR["bH"], CALIBRATOR["bD"], CALIBRATOR["bA"]
logits = np.log(np.maximum([p_home, p_draw, p_away], 1e-12))
p_cal = softmax((logits + np.array([bH, bD, bA])) / T)
p_home, p_draw, p_away = p_cal
return {
"lambda_home": round(lh, 4), "lambda_away": round(la, 4),
"p_home": round(p_home, 4), "p_draw": round(p_draw, 4),
"p_away": round(p_away, 4),
\*\*{k: round(v, 4) for k, v in ou.items()},
"btts": round(btts, 4),
}

8. VALUE BETTING — ZASADA

Model mówi: P(H) = 58%
Bukmacher: kurs 1.80 → implied prob = 56%
Różnica +2pp → VALUE BET → GRASZ

Nie ma "magicznego progu" typu "65% = gram zawsze".
Ważny jest edge, nie prawdopodobieństwo. 9. PIERWSZE KOMENDY W NOWEJ SESJI

# Stan bazy meczowej

python scripts/audit/check_db.py

# Kompletność statystyk (per sezon)

python scripts/audit/audit_matches_completeness.py

# Sprawdź dane zawodników z ekstraklasa.org

ls data/raw/ekstraklasa_org/2025-2026/
head data/processed/zawodnicy_ekstraklasa_org_2025_26.csv

10. ZASADY PORZĄDKU

    Skrypt nie zadziałał → od razu do archive
    Po każdej dużej operacji → commit
    Nie zbieramy śmieci "na potem"
    Debug/eksperymenty → archive/scripts/\*\_debug/
    Produkcyjne skrypty → scripts/
