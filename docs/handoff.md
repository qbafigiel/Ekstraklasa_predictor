# EKSTRAKLASA PREDICTOR — HANDOFF

**Data ostatniej aktualizacji:** [wpisz dziś]
**Repozytorium:** GitHub private — Ekstraklasa_predictor
**Lokalizacja:** D:\projects\Ekstraklasa_predictor\

---

## 1. STAN PROJEKTU

### Model — STABILNY, ZAMKNIĘTY

- **Silnik goli (O/U):** Poisson MLE + prior beniaminków — PRODUKCJA
- **Silnik 1X2:** Poisson(xG) + Softmax Calibration
- **Log-loss:** 1.0571 na sezonie 2025/26
- **Wszystkie eksperymenty matematyczne wyczerpane:**
  - Dixon-Coles ❌
  - Rolling Form ❌
  - Time Decay ❌
  - Logistic Regression ❌
  - Availability score (składy) ❌

### Baza danych — 918 meczów (3 sezony)

- 2023/24, 2024/25, 2025/26
- Pełne statystyki drużyn per mecz (z Flashscore)
- Pełne lineups + absences + coaches + substitutions (z Flashscore)

---

## 2. CO PRÓBOWALIŚMY W OSTATNIEJ SESJI (i dlaczego nie działa)

### Availability Score — porażka udowodniona matematycznie

**Hipoteza:** brak kluczowych zawodników → spadek xG drużyny → korekta lambdy

**Metodologia:**

1. Zbudowaliśmy tabelę `player_minutes` (37k rekordów) — minuty każdego zawodnika w każdym meczu
2. Zbudowaliśmy `match_availability` — dla każdego meczu score 0.0-1.0 opisujący "ile procent core roster było dostępne"
3. Testowaliśmy 3 wersje:
   - v1: prosta średnia minut z całej historii
   - v2: recency window (ostatnie 12 meczów)
   - v3: sezon-aware (tylko bieżący sezon)

**Testy korelacji:**

- Test bezpośredni: korelacje ~0.05, p > 0.1 (statystyczny szum)
- Test within-team (kontrola siły drużyn): korelacje w ODWROTNYM kierunku
- Bucket analysis: osłabione drużyny mają WIĘCEJ xG (nie mniej)

**Interpretacja:**

- Osłabione drużyny grają rozpaczliwie ofensywnie (kontrataki, dalekie strzały)
- Wysoki xG bez skuteczności
- xG jest już produktem tego składu który zagrał — nie da się wyodrębnić efektu składu z xG
- 918 meczów za mało żeby wychwycić subtelny sygnał

**Wniosek:**

- W naszych danych "kogo brakuje w kadrze" nie ma wartości predykcyjnej
- Musimy zejść na poziom JAKOŚCI konkretnych zawodników, nie ich ilości

---

## 3. NOWY KIERUNEK — ekstraklasa.org player value

### Źródło danych

**URL:** https://ekstraklasa.org/statystyki/?season=2025-2026

**Dostępne kategorie (per zawodnik, per sezon):**

- Ofensywne: xG, xGOT, gole, strzały, celne strzały, dogodne szanse
- Defensywne: pojedynki obronne, przechwyty, wybicia, zablokowane strzały
- Podania: podania celne, kluczowe podania, xA, dośrodkowania
- Pozostałe: dryblingi, minuty, spalone, faule, kartki
- Bramkarze: osobna zakładka (czyste konta, obronione strzały, obronione karne)

**Sezony dostępne:** 2023/24, 2024/25, 2025/26 (potwierdzone)

### Dlaczego to jest lepsze niż to co mamy

- Oficjalne źródło (bez anty-botów typu SofaScore)
- Kompletne per-90 statystyki
- Można zbudować **realny player value model**:
  - `Grosicki: xG per 90 = 0.31, xA per 90 = 0.24`
  - Suma wartości zawodników na boisku → jakość składu meczowego
  - Deficyt jakości = realny sygnał predykcyjny

---

## 4. PLAN NA NASTĘPNE KROKI

### KROK 1 (następne działanie): Sanity check danych

Porównać per-sezon statystyki z Flashscore (naszej bazy) z tym co jest na ekstraklasa.org.

- Cel: sprawdzić czy dane są spójne (różnice <5%)
- Jeśli tak → można ufać Flashscore i budować mapowanie
- Jeśli nie → problem fundamentalny, trzeba rozwiązać zanim scrapujemy zawodników

### KROK 2: Sanity check ekstraklasa.org

- Sprawdzić strukturę URL
- Sprawdzić dostępność sezonu 2023/24
- Zobaczyć profil zawodnika i pełny ranking
- Ocenić trudność scrapingu

### KROK 3: Decyzja o zakresie scrapingu

Na podstawie kroku 2 zdecydować co i jak scrapujemy.

### KROK 4: Prototyp scrapera na 1 stronie

Zanim scrape 500 zawodników — sprawdzamy że umiemy pobrać 1.

### KROK 5: Full scraping wszystkich zawodników × 3 sezony

### KROK 6: Mapowanie nazwisk Flashscore ↔ ekstraklasa.org

- Flashscore: "Abramowicz S."
- ekstraklasa.org: prawdopodobnie "Sławomir Abramowicz"
- Tu będzie 90% technicznych problemów

### KROK 7: Player Value Model

- Dla każdego zawodnika: per-90 metryki
- Dla każdego meczu: suma jakości zawodników na boisku
- Feature: deficyt jakości vs typowy skład

### KROK 8: Test korelacji z xG drużyny (właściwy — z jakością, nie ilością)

### KROK 9: Jeśli sygnał → integracja z Poissonem jako korekta lambdy

---

## 5. STAN ARCHITEKTURY

### Struktura projektu (posprzątana)

Ekstraklasa*predictor/
├── db/ekstraklasa.db
├── data/
│ ├── raw/ (surowe źródła API + Flashscore)
│ ├── processed/ (aktualne mvp*, jsony, statusy)
│ └── reports/
├── docs/
│ ├── data_dictionary.md
│ └── handoff.md (ten plik)
├── scripts/
│ ├── audit/ (10 aktywnych audytów)
│ ├── cleaning/ (build_mvp_z_xg.py)
│ ├── db/ (baza.py)
│ ├── model/ (5 produkcyjnych modeli)
│ └── scraping/
│ ├── flashscore/ (scrape_lineups, backfill)
│ └── debug/
└── archive/ (posprzątane, po funkcji: backtests, scripts/, tests, data)

### Zasada porządku (obowiązuje od teraz)

- Nowy plik → od razu do właściwego folderu
- Skrypt nie zadziałał / eksperyment negatywny → od razu do archive/
- Po każdej dużej operacji → commit
- Nie zbieramy śmieci "na potem"

---

## 6. TABELE W BAZIE

### Aktywne (używane przez model produkcyjny)

- `matches` — 918 meczów z pełnymi statystykami drużyn
- `match_coaches` — trenerzy per mecz (potencjał: feature "coach change")

### Zbudowane, ale nie dały sygnału (zostają jako pomost do przyszłego mapowania)

- `lineups` — 37535 rekordów (starterzy + ławka)
- `match_absences` — 4172 rekordy
- `match_substitutions` — 8126 rekordów
- `player_minutes` — 37535 rekordów
- `match_availability` — 1836 rekordów

**Nie usuwamy** — będą potrzebne do mapowania z ekstraklasa.org

---

## 7. PROFIL UŻYTKOWNIKA

- Nieprogramista — pracuje krok po kroku z AI
- Oczekuje pełnych gotowych skryptów (nie fragmentów)
- Komunikuje się po polsku
- AI podejmuje decyzje techniczne i mówi uczciwie gdy coś nie działa
- Interesują go rynki: 1X2, O/U, kartki, rożne
- Rozumie że model wycenia szanse (value bet = model_prob > implied_prob)
- Cel: przewaga nad bukmacherem przez lepszą wycenę szans

---

## 8. STACK TECHNICZNY

Python 3.12
SQLite (db/ekstraklasa.db, 918 meczów)
scikit-learn, scipy, numpy, pandas
Playwright (Firefox) + BeautifulSoup do scrapingu
VS Code + Git (przez interfejs VS Code Source Control)

---

## 9. ZASADY PRACY (uzgodnione)

1. **Jeden krok na raz** — nie biegamy w przód
2. **Testujemy hipotezy przed budową systemu** — jeśli sygnał zerowy → nie budujemy
3. **Porządek po każdym kroku** — commit + archiwum
4. **Uczciwość** — jeśli coś nie działa, AI mówi to wprost
5. **Model wycenia szanse, nie typuje wyniki** — value bet gdy edge > 0
