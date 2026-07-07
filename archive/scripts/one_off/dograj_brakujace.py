import pandas as pd
import time
import re
import urllib.parse
from playwright.sync_api import sync_playwright

# ============================================================
# KONFIGURACJA — URL-e do uzupełnienia
# ============================================================
BRAKUJACE_URL = [
    "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/pogon-szczecin-Um9YwPQ0/szczegoly/statystyki/?mid=hltUZQX2",
    "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=MRhRBEoD",
    "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=pIwRSRAT",
]

PLIK_CSV = "data/flash_2025_26.csv"

# ============================================================
# Kopiuj z scraper_flashscore.py
# ============================================================
MAPA_STAT = {
    "Oczekiwane gole (xG)":               ("xg_gosp", "xg_gosc"),
    "xG na bramkę (xGOT)":                ("xgot_gosp", "xgot_gosc"),
    "Oczekiwane asysty (xA)":             ("xa_gosp", "xa_gosc"),
    "Posiadanie piłki":                    ("posiadanie_gosp", "posiadanie_gosc"),
    "Strzały łącznie":                     ("strzaly_gosp", "strzaly_gosc"),
    "Strzały na bramkę":                   ("celne_gosp", "celne_gosc"),
    "Strzały niecelne":                    ("strzaly_niecelne_gosp", "strzaly_niecelne_gosc"),
    "Strzały zablokowane":                 ("strzaly_zablokowane_gosp", "strzaly_zablokowane_gosc"),
    "Strzały z pola karnego":              ("strzaly_pk_gosp", "strzaly_pk_gosc"),
    "Strzały spoza pola karnego":          ("strzaly_spoza_pk_gosp", "strzaly_spoza_pk_gosc"),
    "Wielkie szanse":                      ("wielkie_szanse_gosp", "wielkie_szanse_gosc"),
    "Rzuty rożne":                         ("rozne_gosp", "rozne_gosc"),
    "Kontakty w polu karnym przeciwnika":  ("kontakty_pk_gosp", "kontakty_pk_gosc"),
    "Spalone":                             ("spalone_gosp", "spalone_gosc"),
    "Rzuty wolne":                         ("rzuty_wolne_gosp", "rzuty_wolne_gosc"),
    "Podania":                             ("podania_gosp", "podania_gosc"),
    "Długie podania":                      ("dlugie_podania_gosp", "dlugie_podania_gosc"),
    "Dośrodkowania":                       ("dosrodkowania_gosp", "dosrodkowania_gosc"),
    "Faule":                               ("faule_gosp", "faule_gosc"),
    "Próby odbioru piłki":                ("odbiory_gosp", "odbiory_gosc"),
    "Obrony bramkarza":                    ("obrony_bramkarza_gosp", "obrony_bramkarza_gosc"),
    "xGot przeciw":                        ("xgot_przeciw_gosp", "xgot_przeciw_gosc"),
    "Zapobiegnięcia utracie gola":         ("zapobiegniecia_gosp", "zapobiegniecia_gosc"),
    "Żółte kartki":                        ("zk_gosp", "zk_gosc"),
    "Czerwone kartki":                     ("czk_gosp", "czk_gosc"),
    "Wygrane pojedynki":                   ("pojedynki_gosp", "pojedynki_gosc"),
    "Wybicia":                             ("wybicia_gosp", "wybicia_gosc"),
    "Przechwyty":                          ("przechwyty_gosp", "przechwyty_gosc"),
    "Błędy skutkujące strzałem":          ("bledy_strzal_gosp", "bledy_strzal_gosc"),
    "Błędy skutkujące golem":             ("bledy_gol_gosp", "bledy_gol_gosc"),
}

NAGLOWKI = {"TOP STATYSTYKI", "STRZAŁY", "ATAK", "PODANIA", "OBRONA",
             "STATYSTYKI BRAMKARZA", "KURSY", "MECZ", "1. POŁOWA", "2. POŁOWA",
             "SZCZEGÓŁY", "STATYSTYKI", "SKŁADY", "STATYSTYKI ZAWODNIKÓW"}

DOMYSLNE_ZERO = {
    "czk_gosp", "czk_gosc", "zk_gosp", "zk_gosc",
    "bledy_strzal_gosp", "bledy_strzal_gosc",
    "bledy_gol_gosp", "bledy_gol_gosc",
    "zapobiegniecia_gosp", "zapobiegniecia_gosc",
}


def wyciagnij_liczbe(tekst):
    tekst = tekst.strip()
    match = re.search(r'\((\d+)/\d+\)', tekst)
    if match:
        return match.group(1)
    match = re.match(r'^-?[\d.]+$', tekst)
    if match:
        return tekst
    return None


def znajdz_wstecz(linie, od):
    j = od - 1
    while j >= max(0, od - 6):
        k = linie[j].strip()
        if k and k not in MAPA_STAT and k not in NAGLOWKI:
            if re.match(r'^\d+%$', k):
                prev = linie[j-1].strip() if j > 0 else ""
                m = re.search(r'\((\d+)/\d+\)', prev)
                if m:
                    return m.group(1)
                return k.replace("%", "")
            wynik = wyciagnij_liczbe(k)
            if wynik is not None:
                return wynik
        j -= 1
    return None


def znajdz_wprzod(linie, od):
    j = od + 1
    while j < min(len(linie), od + 6):
        k = linie[j].strip()
        if k and k not in MAPA_STAT and k not in NAGLOWKI:
            if re.match(r'^\d+%$', k):
                nast = linie[j+1].strip() if j+1 < len(linie) else ""
                m = re.search(r'\((\d+)/\d+\)', nast)
                if m:
                    return m.group(1)
                return k.replace("%", "")
            wynik = wyciagnij_liczbe(k)
            if wynik is not None:
                return wynik
        j += 1
    return None


def parsuj_statystyki(linie):
    statystyki = {}
    przetworzone = set()
    for i, linia in enumerate(linie):
        nazwa = linia.strip()
        if nazwa not in MAPA_STAT or nazwa in przetworzone:
            continue
        klucz_gosp, klucz_gosc = MAPA_STAT[nazwa]
        val_gosp = znajdz_wstecz(linie, i)
        val_gosc = znajdz_wprzod(linie, i)
        if val_gosp is not None and val_gosc is not None:
            statystyki[klucz_gosp] = val_gosp
            statystyki[klucz_gosc] = val_gosc
            przetworzone.add(nazwa)
    for klucz in DOMYSLNE_ZERO:
        if klucz not in statystyki:
            statystyki[klucz] = "0"
    return statystyki


def wyciagnij_meta(linie):
    data_meczu = None
    kolejka = None
    for linia in linie:
        l = linia.strip()
        if data_meczu is None:
            m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', l)
            if m:
                data_meczu = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        if kolejka is None:
            m = re.search(r'KOLEJKA\s+(\d+)', l, re.IGNORECASE)
            if m:
                kolejka = int(m.group(1))
        if data_meczu and kolejka:
            break
    return data_meczu, kolejka


def pobierz_mecz(url, page, debug=True):
    """Pobiera dane jednego meczu. Zwraca dict ze statystykami lub None."""
    print(f"\n{'='*60}")
    print(f"URL: {url}")

    for attempt in range(3):
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(4000)
            try:
                page.click("button#onetrust-accept-btn-handler", timeout=2000)
                page.wait_for_timeout(500)
            except:
                pass

            tekst = page.inner_text("body")
            linie = [l.strip() for l in tekst.split("\n") if l.strip()]

            if debug:
                print(f"\n--- SUROWY TEKST (pierwsze 80 linii) ---")
                for i, l in enumerate(linie[:80]):
                    print(f"  [{i:3d}] {l}")
                print(f"--- KONIEC PODGLĄDU ---\n")

            stat = parsuj_statystyki(linie)
            data_meczu, kolejka = wyciagnij_meta(linie)

            parsed = urllib.parse.urlparse(url)
            mid = urllib.parse.parse_qs(parsed.query).get("mid", [None])[0]

            stat["data_meczu_flash"] = data_meczu
            stat["kolejka_flash"] = kolejka
            stat["url"] = url
            stat["flash_id"] = mid

            print(f"Wynik: kolejka={kolejka}, data={data_meczu}, xG={stat.get('xg_gosp','BRAK')}/{stat.get('xg_gosc','BRAK')}")
            print(f"Pobrano statystyk: {len([k for k in stat if k not in ('data_meczu_flash','kolejka_flash','url','flash_id')])}")

            return stat

        except Exception as e:
            print(f"Próba {attempt+1}/3 błąd: {e}")
            if attempt < 2:
                time.sleep(5)

    return None


def uzupelnij_csv(wyniki):
    """Wpisuje pobrane dane do istniejącego CSV po URL."""
    df = pd.read_csv(PLIK_CSV)
    print(f"\nCSV przed uzupełnieniem: {len(df)} wierszy")

    for wynik in wyniki:
        if wynik is None:
            continue
        url = wynik.get("url")

        maska = df["url"] == url
        if maska.sum() == 0:
            print(f"  UWAGA: nie znaleziono wiersza dla URL: {url}")
            continue

        idx = df[maska].index[0]

        for kolumna, wartosc in wynik.items():
            if kolumna not in df.columns:
                continue
            if wartosc is None:
                continue
            # Konwertuj do typu kolumny
            try:
                dtype = df[kolumna].dtype
                if pd.api.types.is_float_dtype(dtype):
                    wartosc = float(wartosc)
                elif pd.api.types.is_integer_dtype(dtype):
                    wartosc = int(wartosc)
                df.at[idx, kolumna] = wartosc
            except (ValueError, TypeError):
                df.at[idx, kolumna] = wartosc

        print(f"  Uzupełniono wiersz {idx}: {wynik.get('data_meczu_flash')} kolejka {wynik.get('kolejka_flash')} xG={wynik.get('xg_gosp','?')}/{wynik.get('xg_gosc','?')}")

    df.to_csv(PLIK_CSV, index=False, encoding="utf-8-sig")
    print(f"\nZapisano {PLIK_CSV}")

    # Weryfikacja
    brak = df[df["xg_gosp"].isna() | df["kolejka_flash"].isna()]
    print(f"Pozostałe mecze bez danych: {len(brak)}")
    if len(brak) > 0:
        print(brak[["kolejka_flash", "data_meczu_flash", "url"]].to_string())


if __name__ == "__main__":
    wyniki = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "pl-PL,pl;q=0.9"
        })

        for url in BRAKUJACE_URL:
            wynik = pobierz_mecz(url, page, debug=True)
            wyniki.append(wynik)
            time.sleep(2)

        browser.close()

    # Uzupełnij CSV
    uzupelnij_csv(wyniki)