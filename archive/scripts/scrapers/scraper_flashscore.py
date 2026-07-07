import pandas as pd
import time
import re
import urllib.parse
from playwright.sync_api import sync_playwright


def wyciagnij_liczbe(tekst):
    tekst = tekst.strip()
    match = re.search(r'\((\d+)/\d+\)', tekst)
    if match:
        return match.group(1)
    match = re.match(r'^-?[\d.]+$', tekst)
    if match:
        return tekst
    return None


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
    "czk_gosp", "czk_gosc",
    "zk_gosp", "zk_gosc",
    "bledy_strzal_gosp", "bledy_strzal_gosc",
    "bledy_gol_gosp", "bledy_gol_gosc",
    "zapobiegniecia_gosp", "zapobiegniecia_gosc",
}


def wyciagnij_meta_meczu(linie):
    """
    Wyciąga datę i numer kolejki z linii tekstu strony.
    Data format na stronie: "09.08.2025  17:30"  -> zwracamy "2025-08-09"
    Kolejka format: "KOLEJKA 4"                   -> zwracamy 4
    """
    data_meczu = None
    kolejka = None

    for linia in linie:
        linia = linia.strip()

        # Szukaj daty: DD.MM.YYYY (ewentualnie z godziną po białych znakach)
        if data_meczu is None:
            m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', linia)
            if m:
                dzien, miesiac, rok = m.group(1), m.group(2), m.group(3)
                data_meczu = f"{rok}-{miesiac}-{dzien}"

        # Szukaj kolejki: "KOLEJKA X" lub "- KOLEJKA X"
        if kolejka is None:
            m = re.search(r'KOLEJKA\s+(\d+)', linia, re.IGNORECASE)
            if m:
                kolejka = int(m.group(1))

        # Przerwij jak mamy oba
        if data_meczu and kolejka:
            break

    return data_meczu, kolejka


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


def parsuj_statystyki_flashscore(linie):
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


def zbierz_linki(sezon_url):
    print(f"  Zbieram linki z: {sezon_url}")
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "pl-PL,pl;q=0.9"
        })
        page.goto(sezon_url, wait_until="networkidle")
        page.wait_for_timeout(4000)

        try:
            page.click("button#onetrust-accept-btn-handler", timeout=3000)
            page.wait_for_timeout(1000)
        except:
            pass

        klikniecia = 0
        while klikniecia < 30:
            try:
                btn = None
                for selektor in [
                    "a.event__more",
                    "button.event__more",
                    "[class*='event__more']",
                    "[class*='showMore']",
                    "[class*='show-more']",
                ]:
                    btn = page.query_selector(selektor)
                    if btn and btn.is_visible():
                        break
                    btn = None

                if not btn:
                    for el in page.query_selector_all("a, button"):
                        try:
                            tekst = el.inner_text().strip()
                            if "Pokaż więcej" in tekst and el.is_visible():
                                btn = el
                                break
                        except:
                            continue

                if not btn:
                    print(f"  Brak przycisku po {klikniecia} kliknięciach")
                    break

                page.evaluate("el => el.scrollIntoView({block: 'center'})", btn)
                page.wait_for_timeout(800)
                btn.click(timeout=8000)
                klikniecia += 1
                page.wait_for_timeout(2000)

                linki_teraz = page.query_selector_all("a[href*='/mecz/pilka-nozna/']")
                print(f"  Kliknięcie {klikniecia}: {len(linki_teraz)} meczów")

            except Exception as e:
                print(f"  Koniec klikania po {klikniecia} kliknięciach: {e}")
                break

        linki = []
        for el in page.query_selector_all("a[href*='/mecz/pilka-nozna/']"):
            href = el.get_attribute("href") or ""
            if not href.startswith("https://www.flashscore.pl/mecz/pilka-nozna/"):
                continue
            parsed = urllib.parse.urlparse(href)
            mid = urllib.parse.parse_qs(parsed.query).get("mid", [None])[0]
            base = parsed.scheme + "://" + parsed.netloc + parsed.path.rstrip("/")
            if mid:
                url_stat = f"{base}/szczegoly/statystyki/?mid={mid}"
            else:
                url_stat = f"{base}/szczegoly/statystyki/"
            if url_stat not in linki:
                linki.append(url_stat)

        browser.close()

    print(f"  Znaleziono {len(linki)} unikalnych linków")
    return linki


def pobierz_statystyki_sezonu(sezon_url, nazwa_pliku, nazwa_sezonu):
    print(f"\n=== SEZON {nazwa_sezonu} ===")
    linki = zbierz_linki(sezon_url)

    if not linki:
        print("  Brak linków, pomijam")
        return None

    wszystkie = []
    brak_daty = 0
    brak_kolejki = 0

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)

        for idx, url in enumerate(linki):
            print(f"  [{idx+1}/{len(linki)}]", end=" ", flush=True)

            for attempt in range(3):
                try:
                    page = browser.new_page()
                    page.set_extra_http_headers({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept-Language": "pl-PL,pl;q=0.9"
                    })
                    page.goto(url, wait_until="networkidle", timeout=25000)
                    page.wait_for_timeout(3000)
                    try:
                        page.click("button#onetrust-accept-btn-handler", timeout=2000)
                        page.wait_for_timeout(500)
                    except:
                        pass
                    tekst = page.inner_text("body")
                    page.close()

                    linie = [l.strip() for l in tekst.split("\n") if l.strip()]

                    # Wyciągnij statystyki
                    stat = parsuj_statystyki_flashscore(linie)

                    # Wyciągnij datę i kolejkę
                    data_meczu, kolejka = wyciagnij_meta_meczu(linie)

                    if data_meczu is None:
                        brak_daty += 1
                    if kolejka is None:
                        brak_kolejki += 1

                    stat["data_meczu_flash"] = data_meczu
                    stat["kolejka_flash"] = kolejka
                    stat["url"] = url

                    parsed = urllib.parse.urlparse(url)
                    mid = urllib.parse.parse_qs(parsed.query).get("mid", [None])[0]
                    stat["flash_id"] = mid

                    wszystkie.append(stat)
                    print(f"OK kolejka:{kolejka} data:{data_meczu} xG:{stat.get('xg_gosp','?')}/{stat.get('xg_gosc','?')}")
                    time.sleep(0.8)
                    break

                except Exception as e:
                    print(f"Próba {attempt+1} błąd: {e}")
                    try:
                        page.close()
                    except:
                        pass
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"  POMINIĘTO: {url}")

        browser.close()

    df = pd.DataFrame(wszystkie)

    for kol in DOMYSLNE_ZERO:
        if kol in df.columns:
            df[kol] = df[kol].fillna(0)

    df.to_csv(f"data/{nazwa_pliku}", index=False, encoding="utf-8-sig")
    print(f"\n  Zapisano {len(df)} meczów do data/{nazwa_pliku}")
    print(f"  Brak daty: {brak_daty} meczów")
    print(f"  Brak kolejki: {brak_kolejki} meczów")

    braki = df.isnull().sum()
    braki = braki[braki > 0]
    if len(braki) > 0:
        print("  Brakujące dane:")
        print(braki.to_string())
    else:
        print("  Wszystkie dane kompletne!")

    return df


if __name__ == "__main__":
    SEZONY = [
        {
            "url": "https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa-2025-2026/wyniki/",
            "plik": "flash_2025_26.csv",
            "nazwa": "2025/26"
        },
        {
            "url": "https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa-2024-2025/wyniki/",
            "plik": "flash_2024_25.csv",
            "nazwa": "2024/25"
        },
        {
            "url": "https://www.flashscore.pl/pilka-nozna/polska/pko-bp-ekstraklasa-2023-2024/wyniki/",
            "plik": "flash_2023_24.csv",
            "nazwa": "2023/24"
        },
    ]

    for sezon in SEZONY:
        pobierz_statystyki_sezonu(sezon["url"], sezon["plik"], sezon["nazwa"])

    print("\n=== WSZYSTKIE SEZONY POBRANE ===")