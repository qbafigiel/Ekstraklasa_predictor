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


def api_get_z_retry(endpoint, headers, max_retry=3):
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(max_retry):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Próba {attempt+1}/{max_retry}: {e}")
            if attempt < max_retry - 1:
                time.sleep(5)
    return None


def znajdz_rundy_sezonu(headers):
    print("Szukam sezonów i ich round_id...\n")
    data = api_get_z_retry(
        "competition_seasons?filter%5Bcompetition_id_eq%5D=1&include=season",
        headers
    )
    if not data:
        return

    print("Dostępne sezony:")
    for obj in data.get("data", []):
        season_id = obj["id"]
        name = obj["attributes"].get("name", "")
        season_rel = obj.get("relationships", {}).get("season", {}).get("data", {})
        print(f"  competition_season_id={season_id} | {name} | season_id={season_rel.get('id')}")

    print()

    # Pobierz stages żeby znaleźć grupy i rundy
    data2 = api_get_z_retry("stages?include=season,groups", headers)
    if not data2:
        return

    print("Dostępne stages (rundy zasadnicze):")
    for obj in data2.get("data", []):
        stage_id = obj["id"]
        name = obj["attributes"].get("name", "")
        season_rel = obj.get("relationships", {}).get("season", {}).get("data", {})
        groups = obj.get("relationships", {}).get("groups", {}).get("data", [])
        print(f"  stage_id={stage_id} | {name} | season_id={season_rel.get('id')} | groups={[g['id'] for g in groups]}")

    print()

    # Pobierz rounds dla każdej grupy
    included = data2.get("included", [])
    group_ids = []
    for obj in included:
        if obj["type"] == "group":
            group_ids.append(obj["id"])

    print("Rounds per group (pierwsze 3 i ostatnie 3):")
    for group_id in group_ids[:5]:
        data3 = api_get_z_retry(
            f"rounds?filter%5Bgroup_id_eq%5D={group_id}&group_id={group_id}",
            headers
        )
        if not data3:
            continue
        rounds = data3.get("data", [])
        if rounds:
            pierwsze = rounds[:3]
            ostatnie = rounds[-3:]
            print(f"\n  Group {group_id} | Łącznie rounds: {len(rounds)}")
            for r in pierwsze:
                print(f"    round_id={r['id']} | {r['attributes'].get('name')} | start={r['attributes'].get('start_date')}")
            if len(rounds) > 6:
                print(f"    ...")
            for r in ostatnie:
                print(f"    round_id={r['id']} | {r['attributes'].get('name')} | start={r['attributes'].get('start_date')}")
        time.sleep(0.3)


def pobierz_id_meczow_kolejki(round_id, headers):
    data = api_get_z_retry(f"rounds/{round_id}?include=matches", headers)
    if not data:
        return []
    mecze = data.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
    return [m["id"] for m in mecze]


def pobierz_dane_meczu(match_id, headers):
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
    nieudane_rounds = []

    for round_id in range(round_start, round_end + 1):
        kolejka_nr = round_id - round_start + 1
        print(f"[Kolejka {kolejka_nr}/{round_end - round_start + 1}] round_id={round_id}")

        match_ids = pobierz_id_meczow_kolejki(round_id, headers)
        if not match_ids:
            print(f"  BRAK MECZÓW")
            nieudane_rounds.append(round_id)
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

        time.sleep(0.5)

    # Ponów nieudane
    if nieudane_rounds:
        print(f"\nPonawiam {len(nieudane_rounds)} kolejek za 10 sekund...")
        time.sleep(10)
        for round_id in nieudane_rounds:
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
        braki = df.isnull().sum()
        braki = braki[braki > 0]
        if len(braki) > 0:
            print("Brakujące dane:")
            print(braki)
        else:
            print("Wszystkie dane kompletne!")
    return df


if __name__ == "__main__":
    token = pobierz_token()
    headers = {
        "Authorization": token,
        "Accept": "application/json",
        "Origin": "https://www.ekstraklasa.org",
        "Referer": "https://www.ekstraklasa.org/"
    }
    import json

    # Sprawdź pełną strukturę danych meczu
    print("=== PEŁNA STRUKTURA MECZU 2644 ===")
    data = api_get_z_retry(
        "matches/2644?include=home_squad.team.club,away_squad.team.club,incidents",
        headers
    )
    print(json.dumps(data["data"]["attributes"], indent=2, ensure_ascii=False))

    # Sprawdź czy są dodatkowe statystyki
    print("\n=== WSZYSTKIE STATYSTYKI MECZU 2644 ===")
    stat_data = api_get_z_retry(
        "statistics?filter%5Bcontext_type_eq%5D=Match&filter%5Bresource_type_eq%5D=Squad&filter%5Bcontext_id_eq%5D=2644&include=resource",
        headers
    )
    for obj in stat_data.get("data", []):
        sid = obj["relationships"]["resource"]["data"]["id"]
        values = obj["attributes"]["values"]
        print(f"Squad {sid}: {json.dumps(values, indent=2, ensure_ascii=False)}")