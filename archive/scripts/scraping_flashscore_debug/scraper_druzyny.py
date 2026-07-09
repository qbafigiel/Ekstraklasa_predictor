import pandas as pd
import time
from playwright.sync_api import sync_playwright

PLIK_FLASH = "data/flash_2025_26.csv"
PLIK_WYNIK = "data/flash_2025_26_druzyny.csv"


def pobierz_druzyny(url):
    """Nowa sesja Firefox dla każdego meczu. Zwraca (gospodarz, gosc) jako pełne nazwy."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "pl-PL,pl;q=0.9"
        })
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(2500)
            try:
                page.click("button#onetrust-accept-btn-handler", timeout=2000)
                page.wait_for_timeout(400)
            except:
                pass

            # Spróbuj przez selektory HTML — najdokładniejsza metoda
            try:
                gosp = page.locator(".duelParticipant__home .participant__participantName").inner_text(timeout=5000).strip()
                gosc = page.locator(".duelParticipant__away .participant__participantName").inner_text(timeout=5000).strip()
                browser.close()
                if gosp and gosc:
                    return gosp, gosc
            except:
                pass

            # Fallback: z tekstu strony — tytuł ma format "Druzyna1 v Druzyna2 DD/MM/YYYY"
            tekst = page.title()
            browser.close()
            if " v " in tekst:
                czesc = tekst.split(" v ")
                gosp = czesc[0].strip()
                gosc = czesc[1].split(" ")[0].strip() + " " + " ".join(czesc[1].split(" ")[1:]).split(" ")[0].strip()
                # Tytuł: "Korona Kielce v Wisła Płock 08/08/2025 Statystyki"
                gosc = czesc[1].rsplit(" ", 2)[0].strip()
                return gosp, gosc

        except Exception as e:
            try:
                browser.close()
            except:
                pass

    return None, None


def pobierz_druzyny_z_tytulu(url):
    """Nowa sesja. Wyciąga drużyny z tytułu strony — najprostsze i najszybsze."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "pl-PL,pl;q=0.9"
        })
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(2000)
            try:
                page.click("button#onetrust-accept-btn-handler", timeout=2000)
            except:
                pass

            # Tytuł strony: "Korona Kielce v Wisła Płock 08/08/2025 Statystyki"
            tytul = page.title().strip()
            browser.close()

            if " v " in tytul:
                # Podziel na "gospodarz" i "gość data Statystyki"
                gosp_czesc, reszta = tytul.split(" v ", 1)
                # Usuń datę i "Statystyki" z końca gościa
                # Format daty: DD/MM/YYYY
                import re
                gosc_czesc = re.sub(r'\s+\d{2}/\d{2}/\d{4}.*$', '', reszta).strip()
                return gosp_czesc.strip(), gosc_czesc.strip()

        except Exception as e:
            try:
                browser.close()
            except:
                pass

    return None, None


if __name__ == "__main__":
    df = pd.read_csv(PLIK_FLASH)
    print(f"Wczytano {len(df)} meczów")

    # Wznowienie
    try:
        df_wynik = pd.read_csv(PLIK_WYNIK)
        juz = set(df_wynik["flash_id"].dropna().tolist())
        print(f"Wznowienie — już pobrane: {len(juz)}")
    except FileNotFoundError:
        df_wynik = df.copy()
        df_wynik["gosp_nazwa"] = None
        df_wynik["gosc_nazwa"] = None
        juz = set()

    for nr, row in df.iterrows():
        flash_id = row["flash_id"]
        url = row["url"]

        if flash_id in juz:
            continue

        print(f"[{nr+1}/{len(df)}] {flash_id}", end=" ", flush=True)

        gosp, gosc = pobierz_druzyny_z_tytulu(url)

        if gosp and gosc:
            df_wynik.loc[df_wynik["flash_id"] == flash_id, "gosp_nazwa"] = gosp
            df_wynik.loc[df_wynik["flash_id"] == flash_id, "gosc_nazwa"] = gosc
            print(f"{gosp} vs {gosc}")
        else:
            print("BRAK")

        # Checkpoint co 30
        if (nr + 1) % 30 == 0:
            df_wynik.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
            print(f"  --- Checkpoint {nr+1} ---")

        time.sleep(0.3)

    df_wynik.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
    print(f"\nZapisano do {PLIK_WYNIK}")

    ok = df_wynik["gosp_nazwa"].notna().sum()
    print(f"Pobrano nazwy: {ok}/{len(df_wynik)}")
    brak = df_wynik[df_wynik["gosp_nazwa"].isna()][["flash_id", "url"]]
    if len(brak):
        print(f"\nBrakuje ({len(brak)}):")
        print(brak.to_string())