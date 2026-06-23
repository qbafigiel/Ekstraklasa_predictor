import pandas as pd
import requests
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://production-umpire-api.ekstraklasa.tisagroup.ch/api/v3"


def pobierz_token():
    token = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def on_response(response):
            nonlocal token
            if "umpire-api" in response.url:
                auth = response.request.headers.get("authorization")
                if auth:
                    token = auth

        page.on("response", on_response)
        page.goto("https://www.ekstraklasa.org/terminarz", wait_until="networkidle")
        page.wait_for_timeout(4000)
        browser.close()

    print(f"Token: {'OK' if token else 'BRAK'}")
    return token


def api_get_z_retry(endpoint, headers, max_retry=3):
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(max_retry):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Próba {attempt+1}/{max_retry} nieudana: {e}")
            if attempt < max_retry - 1:
                time.sleep(5)
    return None


def pobierz_id_meczow_kolejki(round_id, headers):
    data = api_get_z_retry(f"rounds/{round_id}?include=matches", headers)
    if not data:
        return []
    mecze = data.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
    return [m["id"] for m in mecze]


def pobierz_dane_meczu(match_id, headers):
    # Info o meczu
    data = api_get_z_retry(
        f"matches/{match_id}?include=home_squad.team.club,away_squad.team.club",
        headers
    )
    if not data:
        return None

    attrs = data["data"]["attributes"]
    included = data.get("included", [])

    clubs = {obj["id"]: obj["attributes"].get("code") for obj in included if obj["type"] == "club"}
    teams = {obj["id"]: obj["relationships"].get("club", {}).get("data", {}).get("id") for obj in included if obj["type"] == "team"}
    squads = {obj["id"]: obj["relationships"].get("team", {}).get("data", {}).get("id") for obj in included if obj["type"] == "squad"}

    rels = data["data"]["relationships"]
    home_squad_id = rels.get("home_squad", {}).get("data", {}).get("id")
    away_squad_id = rels.get("away_squad", {}).get("data", {}).get("id")

    def squad_to_code(sid):
        return clubs.get(teams.get(squads.get(sid)))

    time.sleep(0.2)

    # Statystyki
    stat_data = api_get_z_retry(
        f"statistics?filter%5Bcontext_type_eq%5D=Match&filter%5Bresource_type_eq%5D=Squad&filter%5Bcontext_id_eq%5D={match_id}&include=resource",
        headers
    )
    if not stat_data:
        return None

    stats_by_squad = {}
    for obj in stat_data.get("data", []):
        sid = obj.get("relationships", {}).get("resource", {}).get("data", {}).get("id")
        stats_by_squad[sid] = obj.get("attributes", {}).get("values", {})

    hs = stats_by_squad.get(str(home_squad_id), {})
    as_ = stats_by_squad.get(str(away_squad_id), {})

    return {
        "match_id": match_id,
        "gospodarz": squad_to_code(home_squad_id),
        "gosc": squad_to_code(away_squad_id),
        "gole_gosp": attrs.get("home_squad_score"),
        "gole_gosc": attrs.get("away_squad_score"),
        "data_meczu": attrs.get("start_time", "")[:10] if attrs.get("start_time") else None,
        "posiadanie_gosp": hs.get("ball_possession_percent"),
        "posiadanie_gosc": as_.get("ball_possession_percent"),
        "strzaly_gosp": hs.get("shots"),
        "strzaly_gosc": as_.get("shots"),
        "celne_gosp": hs.get("shots_on_target"),
        "celne_gosc": as_.get("shots_on_target"),
        "strzaly_zablokowane_gosp": hs.get("shots_blocked"),
        "strzaly_zablokowane_gosc": as_.get("shots_blocked"),
        "strzaly_niecelne_gosp": hs.get("shots_off_target"),
        "strzaly_niecelne_gosc": as_.get("shots_off_target"),
        "rozne_gosp": hs.get("corner_kicks"),
        "rozne_gosc": as_.get("corner_kicks"),
        "faule_gosp": hs.get("fouls"),
        "faule_gosc": as_.get("fouls"),
        "spalone_gosp": hs.get("offsides"),
        "spalone_gosc": as_.get("offsides"),
        "zk_gosp": hs.get("yellow_cards"),
        "zk_gosc": as_.get("yellow_cards"),
        "czk_gosp": hs.get("red_cards"),
        "czk_gosc": as_.get("red_cards"),
        "druga_zk_gosp": hs.get("second_yellow_card"),
        "druga_zk_gosc": as_.get("second_yellow_card"),
        "dosrodkowania_gosp": hs.get("crosses"),
        "dosrodkowania_gosc": as_.get("crosses"),
        "dosrodkowania_celne_gosp": hs.get("crosses_accurate"),
        "dosrodkowania_celne_gosc": as_.get("crosses_accurate"),
        "odbiory_gosp": hs.get("tackles_successful"),
        "odbiory_gosc": as_.get("tackles_successful"),
        "podania_gosp": hs.get("passes"),
        "podania_gosc": as_.get("passes"),
        "podania_celne_gosp": hs.get("passes_accurate"),
        "podania_celne_gosc": as_.get("passes_accurate"),
    }


def pobierz_sezon(round_start, round_end, nazwa_pliku, nazwa_sezonu):
    print(f"\n=== POBIERANIE SEZONU {nazwa_sezonu} ===")
    print(f"Rounds: {round_start}-{round_end}\n")

    token = pobierz_token()
    if not token:
        print("Brak tokenu")
        return None

    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Origin": "https://www.ekstraklasa.org",
        "Referer": "https://www.ekstraklasa.org/"
    }

    wszystkie_mecze = []
    nieudane = []

    for round_id in range(round_start, round_end + 1):
        kolejka_nr = round_id - round_start + 1
        print(f"[Kolejka {kolejka_nr}/{round_end - round_start + 1}] round_id={round_id}")

        match_ids = pobierz_id_meczow_kolejki(round_id, headers)
        if not match_ids:
            print(f"  BRAK MECZÓW - kolejka zostanie ponowiona")
            nieudane.append(round_id)
            continue

        print(f"  Meczów: {len(match_ids)}")

        for match_id in match_ids:
            dane = pobierz_dane_meczu(match_id, headers)
            if dane:
                dane["kolejka"] = kolejka_nr
                wszystkie_mecze.append(dane)
                print(f"    OK: {dane['gospodarz']} {dane['gole_gosp']}:{dane['gole_gosc']} {dane['gosc']}")
            else:
                print(f"    BŁĄD: mecz {match_id}")
                nieudane.append(match_id)

        time.sleep(0.5)

    # Ponów nieudane kolejki
    if nieudane:
        print(f"\nPonawiam {len(nieudane)} nieudanych...")
        time.sleep(10)
        for round_id in nieudane:
            if isinstance(round_id, int):
                kolejka_nr = round_id - round_start + 1
                print(f"  Retry kolejka {kolejka_nr}...")
                match_ids = pobierz_id_meczow_kolejki(round_id, headers)
                for match_id in match_ids:
                    dane = pobierz_dane_meczu(match_id, headers)
                    if dane:
                        dane["kolejka"] = kolejka_nr
                        wszystkie_mecze.append(dane)
                        print(f"    OK: {dane['gospodarz']} {dane['gole_gosp']}:{dane['gole_gosc']} {dane['gosc']}")

    df = pd.DataFrame(wszystkie_mecze)
    if not df.empty:
        df = df.sort_values(["kolejka", "match_id"]).reset_index(drop=True)
        df.to_csv(f"data/{nazwa_pliku}", index=False, encoding="utf-8-sig")
        print(f"\nZapisano {len(df)} meczów do data/{nazwa_pliku}")
        print(df.groupby("kolejka").size().to_string())
    return df


if __name__ == "__main__":
    # Sezon 2025/26 - rounds 331-364
    df_2526 = pobierz_sezon(331, 364, "mecze_2025_26.csv", "2025/26")

    # Sprawdź wynik
    if df_2526 is not None:
        print("\n=== WERYFIKACJA BRAKUJĄCYCH DANYCH ===")
        print(df_2526.isnull().sum())