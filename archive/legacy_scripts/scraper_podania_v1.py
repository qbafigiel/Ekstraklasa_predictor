import pandas as pd
import time
import re
from playwright.sync_api import sync_playwright

PLIK_FLASH = "data/flash_2025_26.csv"
PLIK_WYNIK = "data/podania_uzup_2025_26.csv"
RECZNE_FLASH_ID = {"MRhRBEoD", "pIwRSRAT"}

MAPA = {
    "Podania":                              ("podania_sk_gosp",       "podania_wszy_gosp",
                                             "podania_sk_gosc",       "podania_wszy_gosc"),
    "Długie podania":                       ("dl_pod_sk_gosp",        "dl_pod_wszy_gosp",
                                             "dl_pod_sk_gosc",        "dl_pod_wszy_gosc"),
    "Dośrodkowania":                        ("dosrod_sk_gosp",        "dosrod_wszy_gosp",
                                             "dosrod_sk_gosc",        "dosrod_wszy_gosc"),
    "Próby odbioru piłki":                  ("odbiory_sk_gosp",       "odbiory_wszy_gosp",
                                             "odbiory_sk_gosc",       "odbiory_wszy_gosc"),
    "Podania w strefę obrony przeciwnika":  ("pod_strefa_sk_gosp",    "pod_strefa_wszy_gosp",
                                             "pod_strefa_sk_gosc",    "pod_strefa_wszy_gosc"),
}


def znajdz_nawias_wstecz(linie, od):
    for j in range(od - 1, max(0, od - 5), -1):
        m = re.search(r'\((\d+)/(\d+)\)', linie[j].strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def znajdz_nawias_wprzod(linie, od):
    for j in range(od + 1, min(len(linie), od + 5)):
        m = re.search(r'\((\d+)/(\d+)\)', linie[j].strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def parsuj(linie):
    wyniki = {}
    przetworzone = set()
    for i, linia in enumerate(linie):
        nazwa = linia.strip()
        if nazwa not in MAPA or nazwa in przetworzone:
            continue
        sk_g, ws_g, sk_gc, ws_gc = MAPA[nazwa]
        sk_gosp, ws_gosp = znajdz_nawias_wstecz(linie, i)
        sk_gosc, ws_gosc = znajdz_nawias_wprzod(linie, i)
        if sk_gosp is not None:
            wyniki[sk_g] = sk_gosp
            wyniki[ws_g] = ws_gosp
        if sk_gosc is not None:
            wyniki[sk_gc] = sk_gosc
            wyniki[ws_gc] = ws_gosc
        przetworzone.add(nazwa)
    return wyniki


def nowa_strona(browser):
    page = browser.new_page()
    page.set_extra_http_headers({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "pl-PL,pl;q=0.9"
    })
    return page


def pobierz_url(page, url, browser):
    """Próbuje pobrać stronę. Przy błędzie czeka coraz dłużej."""
    przerwy = [5, 15, 30]
    for attempt, przerwa in enumerate(przerwy):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)
            try:
                page.click("button#onetrust-accept-btn-handler", timeout=2000)
                page.wait_for_timeout(500)
            except:
                pass
            tekst = page.inner_text("body")
            linie = [l.strip() for l in tekst.split("\n") if l.strip()]
            # Sprawdź czy strona załadowała statystyki (musi być "Podania")
            if "Podania" in linie:
                return linie, page
            else:
                print(f" brak statystyk na stronie (próba {attempt+1})", end="")
                time.sleep(przerwa)
        except Exception as e:
            print(f" timeout próba {attempt+1}: {str(e)[:40]}", end="")
            try:
                page.close()
            except:
                pass
            page = nowa_strona(browser)
            time.sleep(przerwa)
    return None, page


if __name__ == "__main__":
    df_flash = pd.read_csv(PLIK_FLASH)
    do_scrapowania = df_flash[~df_flash["flash_id"].isin(RECZNE_FLASH_ID)].copy()
    print(f"Meczów do scrapowania: {len(do_scrapowania)}")

    # Wznowienie
    try:
        df_juz = pd.read_csv(PLIK_WYNIK)
        juz_pobrane = set(df_juz["flash_id"].dropna().tolist())
        print(f"Już pobrane: {len(juz_pobrane)} — wznawiam od miejsca przerwania")
    except FileNotFoundError:
        df_juz = pd.DataFrame()
        juz_pobrane = set()

    wyniki = []
    restart_co = 50  # restart browsera co N meczów

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = nowa_strona(browser)
        mecze_od_restartu = 0

        for nr, (idx, row) in enumerate(do_scrapowania.iterrows()):
            url = row["url"]
            flash_id = row["flash_id"]

            if flash_id in juz_pobrane:
                continue

            print(f"[{nr+1}/{len(do_scrapowania)}] {flash_id}", end=" ", flush=True)

            # Restart browsera co restart_co meczów
            if mecze_od_restartu >= restart_co:
                print("\n  --- Restart browsera ---")
                try:
                    page.close()
                    browser.close()
                except:
                    pass
                time.sleep(5)
                browser = p.firefox.launch(headless=True)
                page = nowa_strona(browser)
                mecze_od_restartu = 0

            linie, page = pobierz_url(page, url, browser)

            if linie is None:
                print(" POMINIĘTO")
                wyniki.append({"flash_id": flash_id, "url": url})
            else:
                stat = parsuj(linie)
                stat["flash_id"] = flash_id
                stat["url"] = url
                wyniki.append(stat)
                pod = f"{stat.get('podania_sk_gosp','?')}/{stat.get('podania_wszy_gosp','?')}"
                dos = f"{stat.get('dosrod_sk_gosp','?')}/{stat.get('dosrod_wszy_gosp','?')}"
                print(f"podania:{pod} dosrod:{dos}")

            mecze_od_restartu += 1

            # Checkpoint co 30 meczów
            if len(wyniki) % 30 == 0:
                df_tmp = pd.concat([df_juz, pd.DataFrame(wyniki)], ignore_index=True)
                df_tmp.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
                print(f"  --- Checkpoint: {len(df_tmp)} wierszy zapisanych ---")

            time.sleep(1.5)

        try:
            page.close()
            browser.close()
        except:
            pass

    # Finalne zapisanie
    df_nowe = pd.DataFrame(wyniki)
    df_final = pd.concat([df_juz, df_nowe], ignore_index=True)

    # Ręczne mecze
    reczne = [
        {
            "flash_id": "MRhRBEoD",
            "url": "https://www.flashscore.pl/mecz/pilka-nozna/korona-kielce-pp78XcbA/radomiak-radom-zD5nYhAT/szczegoly/statystyki/?mid=MRhRBEoD",
            "podania_sk_gosp": 248, "podania_wszy_gosp": 315,
            "podania_sk_gosc": 355, "podania_wszy_gosc": 428,
            "dl_pod_sk_gosp": 18, "dl_pod_wszy_gosp": 58,
            "dl_pod_sk_gosc": 27, "dl_pod_wszy_gosc": 53,
            "dosrod_sk_gosp": 4, "dosrod_wszy_gosp": 17,
            "dosrod_sk_gosc": 4, "dosrod_wszy_gosc": 15,
            "odbiory_sk_gosp": 10, "odbiory_wszy_gosp": 15,
            "odbiory_sk_gosc": 9, "odbiory_wszy_gosc": 19,
        },
        {
            "flash_id": "pIwRSRAT",
            "url": "https://www.flashscore.pl/mecz/pilka-nozna/arka-gdynia-feEez0Ei/legia-warszawa-K6kUepBs/szczegoly/statystyki/?mid=pIwRSRAT",
            "podania_sk_gosp": 388, "podania_wszy_gosp": 473,
            "podania_sk_gosc": 213, "podania_wszy_gosc": 284,
            "dl_pod_sk_gosp": 34, "dl_pod_wszy_gosp": 48,
            "dl_pod_sk_gosc": 22, "dl_pod_wszy_gosc": 54,
            "dosrod_sk_gosp": 11, "dosrod_wszy_gosp": 37,
            "dosrod_sk_gosc": 6, "dosrod_wszy_gosc": 19,
            "odbiory_sk_gosp": 7, "odbiory_wszy_gosp": 9,
            "odbiory_sk_gosc": 8, "odbiory_wszy_gosc": 15,
        },
    ]
    for r in reczne:
        if r["flash_id"] not in set(df_final["flash_id"].dropna().tolist()):
            df_final = pd.concat([df_final, pd.DataFrame([r])], ignore_index=True)

    df_final.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
    print(f"\nZapisano {len(df_final)} wierszy do {PLIK_WYNIK}")

    ok = df_final["podania_wszy_gosp"].notna().sum() if "podania_wszy_gosp" in df_final.columns else 0
    print(f"Mają podania_wszy_gosp: {ok}/{len(df_final)}")
    brak = df_final[df_final["podania_wszy_gosp"].isna()]
    if len(brak) > 0:
        print(f"Brakuje danych dla {len(brak)} meczów:")
        print(brak[["flash_id", "url"]].to_string())