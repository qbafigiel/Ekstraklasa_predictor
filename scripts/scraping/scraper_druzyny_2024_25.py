import pandas as pd
import time
import re
from playwright.sync_api import sync_playwright
from pathlib import Path

# Ścieżki zgodne z strukturą projektu
ROOT = Path(__file__).resolve().parents[2]
PLIK_FLASH = ROOT / "data" / "raw" / "flash" / "flash_2024_25.csv"
PLIK_WYNIK = ROOT / "data" / "raw" / "flash" / "flash_2024_25_druzyny.csv"


def pobierz_druzyny_z_tytulu(url: str):
    """Wyciąga gospodarza i gościa z tytułu strony Flashscore."""
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
            except Exception:
                pass

            tytul = page.title().strip()
            browser.close()

            if " v " in tytul:
                gosp_czesc, reszta = tytul.split(" v ", 1)
                # Usuń datę (DD/MM/YYYY) i słowo "Statystyki" z końca
                gosc_czesc = re.sub(r'\s+\d{2}/\d{2}/\d{4}.*$', '', reszta).strip()
                return gosp_czesc.strip(), gosc_czesc.strip()

        except Exception as e:
            try:
                browser.close()
            except Exception:
                pass
    return None, None


def main():
    if not PLIK_FLASH.exists():
        print(f"❌ Brak pliku źródłowego: {PLIK_FLASH}")
        return

    df = pd.read_csv(PLIK_FLASH, encoding="utf-8-sig")
    print(f"Wczytano {len(df)} meczów z {PLIK_FLASH.name}")

    # Wznowienie lub nowy plik
    if PLIK_WYNIK.exists():
        df_wynik = pd.read_csv(PLIK_WYNIK, encoding="utf-8-sig")
        juz = set(df_wynik["flash_id"].dropna().tolist())
        print(f"Wznowienie — już pobrane: {len(juz)}/{len(df)}")
    else:
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
            print(f"✅ {gosp} vs {gosc}")
        else:
            print("❌ BRAK")

        # Checkpoint co 30 meczów
        if (nr + 1) % 30 == 0:
            df_wynik.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
            print(f"  💾 Checkpoint zapisany ({nr+1}/{len(df)})")

        # Flashscore blokuje przy zbyt szybkich requestach
        time.sleep(0.6)

    # Zapis końcowy
    df_wynik.to_csv(PLIK_WYNIK, index=False, encoding="utf-8-sig")
    print(f"\n✅ Zapisano do: {PLIK_WYNIK}")

    ok = df_wynik["gosp_nazwa"].notna().sum()
    print(f"Pobrano nazwy: {ok}/{len(df_wynik)}")
    
    brak = df_wynik[df_wynik["gosp_nazwa"].isna()][["flash_id", "url"]]
    if len(brak):
        print(f"\n⚠️ Brakuje ({len(brak)}):")
        print(brak.to_string())
    else:
        print("\n🎉 Wszystkie mecze mają przypisane drużyny!")


if __name__ == "__main__":
    main()