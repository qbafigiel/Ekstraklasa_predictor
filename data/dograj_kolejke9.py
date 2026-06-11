import pandas as pd
import requests
import time
from playwright.sync_api import sync_playwright

BASE = "https://production-umpire-api.ekstraklasa.tisagroup.ch/api/v3"

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
    return token

token = pobierz_token()
headers = {
    "Authorization": token,
    "Accept": "application/json",
    "Origin": "https://www.ekstraklasa.org",
    "Referer": "https://www.ekstraklasa.org/"
}

match_ids = ["2416", "2417", "2420", "2424", "2418", "2422", "2423", "2419", "2421"]
mecze = []

for match_id in match_ids:
    try:
        r = requests.get(f"{BASE}/matches/{match_id}?include=home_squad.team.club,away_squad.team.club", headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
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

        r2 = requests.get(f"{BASE}/statistics?filter%5Bcontext_type_eq%5D=Match&filter%5Bresource_type_eq%5D=Squad&filter%5Bcontext_id_eq%5D={match_id}&include=resource", headers=headers, timeout=15)
        r2.raise_for_status()
        stat_data = r2.json()
        stats_by_squad = {}
        for obj in stat_data.get("data", []):
            sid = obj.get("relationships", {}).get("resource", {}).get("data", {}).get("id")
            stats_by_squad[sid] = obj.get("attributes", {}).get("values", {})

        hs = stats_by_squad.get(str(home_squad_id), {})
        as_ = stats_by_squad.get(str(away_squad_id), {})

        mecz = {
            "match_id": match_id, "kolejka": 9,
            "gospodarz": squad_to_code(home_squad_id), "gosc": squad_to_code(away_squad_id),
            "gole_gosp": attrs.get("home_squad_score"), "gole_gosc": attrs.get("away_squad_score"),
            "data_meczu": attrs.get("start_time", "")[:10] if attrs.get("start_time") else None,
            "posiadanie_gosp": hs.get("ball_possession_percent"), "posiadanie_gosc": as_.get("ball_possession_percent"),
            "strzaly_gosp": hs.get("shots"), "strzaly_gosc": as_.get("shots"),
            "celne_gosp": hs.get("shots_on_target"), "celne_gosc": as_.get("shots_on_target"),
            "rozne_gosp": hs.get("corner_kicks"), "rozne_gosc": as_.get("corner_kicks"),
            "faule_gosp": hs.get("fouls"), "faule_gosc": as_.get("fouls"),
            "spalone_gosp": hs.get("offsides"), "spalone_gosc": as_.get("offsides"),
            "zk_gosp": hs.get("yellow_cards"), "zk_gosc": as_.get("yellow_cards"),
            "czk_gosp": hs.get("red_cards"), "czk_gosc": as_.get("red_cards"),
            "dosrodkowania_gosp": hs.get("crosses"), "dosrodkowania_gosc": as_.get("crosses"),
            "odbiory_gosp": hs.get("tackles_successful"), "odbiory_gosc": as_.get("tackles_successful"),
            "podania_gosp": hs.get("passes"), "podania_gosc": as_.get("passes"),
            "podania_celne_gosp": hs.get("passes_accurate"), "podania_celne_gosc": as_.get("passes_accurate"),
        }
        mecze.append(mecz)
        print(f"OK: {mecz['gospodarz']} {mecz['gole_gosp']}:{mecz['gole_gosc']} {mecz['gosc']}")
        time.sleep(0.3)

    except Exception as e:
        print(f"BŁĄD mecz {match_id}: {e}")

df_new = pd.DataFrame(mecze)
df_old = pd.read_csv("data/mecze_2025_26.csv")
df_all = pd.concat([df_old, df_new], ignore_index=True).sort_values(["kolejka", "match_id"]).reset_index(drop=True)
df_all.to_csv("data/mecze_2025_26.csv", index=False, encoding="utf-8-sig")
print(f"\nZapisano łącznie {len(df_all)} meczów")
print(df_all.groupby("kolejka").size().to_string())