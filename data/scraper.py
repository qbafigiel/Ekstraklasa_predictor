import pandas as pd
import requests
import json
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


def api_get(endpoint, headers):
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def pobierz_id_meczow_kolejki(round_id, headers):
    data = api_get(f"rounds/{round_id}?include=matches", headers)
    mecze = data.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
    return [m["id"] for m in mecze]


def pobierz_info_meczu(match_id, headers):
    data = api_get(
        f"matches/{match_id}?include=home_squad.team.club,away_squad.team.club",
        headers
    )
    attrs = data.get("data", {}).get("attributes", {})
    included = data.get("included", [])

    # Zbierz kluby
    clubs = {obj["id"]: obj["attributes"].get("code") for obj in included if obj["type"] == "club"}
    teams = {obj["id"]: obj["relationships"].get("club", {}).get("data", {}).get("id")
             for obj in included if obj["type"] == "team"}
    squads = {obj["id"]: obj["relationships"].get("team", {}).get("data", {}).get("id")
              for obj in included if obj["type"] == "squad"}

    rels = data.get("data", {}).get("relationships", {})
    home_squad_id = rels.get("home_squad", {}).get("data", {}).get("id")
    away_squad_id = rels.get("away_squad", {}).get("data", {}).get("id")

    def squad_to_club_code(squad_id):
        team_id = squads.get(squad_id)
        club_id = teams.get(team_id)
        return clubs.get(club_id)

    return {
        "match_id": match_id,
        "home_squad_id": home_squad_id,
        "away_squad_id": away_squad_id,
        "gospodarz": squad_to_club_code(home_squad_id),
        "gosc": squad_to_club_code(away_squad_id),
        "gole_gosp": attrs.get("home_squad_score"),
        "gole_gosc": attrs.get("away_squad_score"),
        "data_meczu": attrs.get("start_time", "")[:10] if attrs.get("start_time") else None,
    }


def pobierz_statystyki_meczu(match_id, home_squad_id, away_squad_id, headers):
    endpoint = f"statistics?filter%5Bcontext_type_eq%5D=Match&filter%5Bresource_type_eq%5D=Squad&filter%5Bcontext_id_eq%5D={match_id}&include=resource"
    data = api_get(endpoint, headers)

    stats_by_squad = {}
    for obj in data.get("data", []):
        squad_id = obj.get("relationships", {}).get("resource", {}).get("data", {}).get("id")
        values = obj.get("attributes", {}).get("values", {})
        stats_by_squad[squad_id] = values

    # Wypisz dostępne pola dla pierwszego meczu
    if match_id == "2644":
        print("\n=== DOSTĘPNE POLA STATYSTYK ===")
        for squad_id, values in stats_by_squad.items():
            print(f"Squad {squad_id}: {list(values.keys())}")
        print("===\n")

    home_stats = stats_by_squad.get(str(home_squad_id), {})
    away_stats = stats_by_squad.get(str(away_squad_id), {})

    def get_pair(key):
        return home_stats.get(key), away_stats.get(key)

    posiadanie_gosp, posiadanie_gosc = get_pair("ball_possession")
    strzaly_gosp, strzaly_gosc = get_pair("shots")
    celne_gosp, celne_gosc = get_pair("shots_on_target")
    rozne_gosp, rozne_gosc = get_pair("corner_kicks")
    faule_gosp, faule_gosc = get_pair("fouls")
    spalone_gosp, spalone_gosc = get_pair("offsides")
    zk_gosp, zk_gosc = get_pair("yellow_cards")
    czk_gosp, czk_gosc = get_pair("red_cards")
    dosrodkowania_gosp, dosrodkowania_gosc = get_pair("crosses")
    odbiory_gosp, odbiory_gosc = get_pair("tackles")
    podania_gosp, podania_gosc = get_pair("passes")
    podania_celne_gosp, podania_celne_gosc = get_pair("accurate_passes")

    # Fallback dla nazw które widzieliśmy w danych
    if strzaly_gosp is None:
        strzaly_gosp, strzaly_gosc = get_pair("shots_total")
    if celne_gosp is None:
        celne_gosp, celne_gosc = get_pair("shots_on_goal")

    return {
        "posiadanie_gosp": posiadanie_gosp,
        "posiadanie_gosc": posiadanie_gosc,
        "strzaly_gosp": strzaly_gosp,
        "strzaly_gosc": strzaly_gosc,
        "celne_gosp": celne_gosp,
        "celne_gosc": celne_gosc,
        "rozne_gosp": rozne_gosp,
        "rozne_gosc": rozne_gosc,
        "faule_gosp": faule_gosp,
        "faule_gosc": faule_gosc,
        "spalone_gosp": spalone_gosp,
        "spalone_gosc": spalone_gosc,
        "zk_gosp": zk_gosp,
        "zk_gosc": zk_gosc,
        "czk_gosp": czk_gosp,
        "czk_gosc": czk_gosc,
        "dosrodkowania_gosp": dosrodkowania_gosp,
        "dosrodkowania_gosc": dosrodkowania_gosc,
        "odbiory_gosp": odbiory_gosp,
        "odbiory_gosc": odbiory_gosc,
        "podania_gosp": podania_gosp,
        "podania_gosc": podania_gosc,
        "podania_celne_gosp": podania_celne_gosp,
        "podania_celne_gosc": podania_celne_gosc,
    }


def pobierz_wszystkie_mecze(zapisz_do_csv=True):
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
    round_start = 331
    round_end = 364

    for round_id in range(round_start, round_end + 1):
        kolejka_nr = round_id - round_start + 1
        print(f"\n[Kolejka {kolejka_nr}/34] round_id={round_id}")

        try:
            match_ids = pobierz_id_meczow_kolejki(round_id, headers)
            print(f"  Meczów: {len(match_ids)}")

            for match_id in match_ids:
                try:
                    info = pobierz_info_meczu(match_id, headers)
                    time.sleep(0.2)

                    stat = pobierz_statystyki_meczu(
                        match_id,
                        info["home_squad_id"],
                        info["away_squad_id"],
                        headers
                    )
                    time.sleep(0.2)

                    mecz = {**info, **stat, "kolejka": kolejka_nr}
                    mecz.pop("home_squad_id", None)
                    mecz.pop("away_squad_id", None)
                    wszystkie_mecze.append(mecz)

                    print(f"    OK: {info.get('gospodarz','?')} {info.get('gole_gosp','?')}:{info.get('gole_gosc','?')} {info.get('gosc','?')}")

                except Exception as e:
                    print(f"    BŁĄD mecz {match_id}: {e}")

        except Exception as e:
            print(f"  BŁĄD kolejka: {e}")

    df = pd.DataFrame(wszystkie_mecze)

    if not df.empty:
        df = df.sort_values(["kolejka", "match_id"]).reset_index(drop=True)
        if zapisz_do_csv:
            df.to_csv("data/mecze_2025_26.csv", index=False, encoding="utf-8-sig")
            print(f"\nZapisano {len(df)} meczów do data/mecze_2025_26.csv")

    return df


if __name__ == "__main__":
    print("=== SCRAPING API EKSTRAKLASY 2025/26 ===\n")

    # Najpierw test na jednym meczu żeby zobaczyć dostępne pola
    print("--- TEST: statystyki meczu 2644 (Pogoń vs GKS) ---")
    token = pobierz_token()
    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Origin": "https://www.ekstraklasa.org",
        "Referer": "https://www.ekstraklasa.org/"
    }
    info = pobierz_info_meczu("2644", headers)
    print(f"Mecz: {info['gospodarz']} {info['gole_gosp']}:{info['gole_gosc']} {info['gosc']}")
    stat = pobierz_statystyki_meczu("2644", info["home_squad_id"], info["away_squad_id"], headers)
    print(f"Statystyki: {stat}")

    print("\n--- Pobieranie wszystkich meczów ---")
    df = pobierz_wszystkie_mecze(zapisz_do_csv=True)
    if df is not None and not df.empty:
        print(f"\n=== PODSUMOWANIE ===")
        print(f"Łącznie meczów: {len(df)}")
        print(df[["match_id", "kolejka", "gospodarz", "gosc", "gole_gosp", "gole_gosc"]].head(10))