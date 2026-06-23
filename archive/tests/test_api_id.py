import requests
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

# Lech vs Cracovia kolejka 1 - match_id 2345
r = requests.get(f"{BASE}/matches/2345", headers=headers, timeout=15)
import json
data = r.json()
print("=== WSZYSTKIE POLA MECZU 2345 ===")
print(json.dumps(data["data"]["attributes"], indent=2, ensure_ascii=False))
print("\n=== RELATIONSHIPS ===")
print(json.dumps(data["data"]["relationships"], indent=2, ensure_ascii=False))