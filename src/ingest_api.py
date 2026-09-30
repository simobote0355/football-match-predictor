import json
import time
import requests
from config import RAW_DIR, API_TOKEN, API_BASE_URL, API_LEAGUE, API_SEASONS, PAUSE_SECONDS, MAX_RETRIES


def fetch_season(season):
    """Download the matches from a season. Returns JSON or None."""
    url = f"{API_BASE_URL}/competitions/{API_LEAGUE}/matches"
    headers = {"X-Auth-Token": API_TOKEN}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params={"season": season}, timeout=(10, 30))
        except requests.RequestException as e:
            print(f"Network error (attempt {attempt}): {e}")
            time.sleep(10)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            print("Rate limit hit, waiting 60 s...")
            time.sleep(60)
            continue
        if r.status_code == 403:
            print(f"Season {season} is not available on the free plan.")
            return None
        r.raise_for_status()

    return None


def main():
    if not API_TOKEN:
        raise SystemExit("Missing FOOTBALL_DATA_TOKEN in the .env file")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in API_SEASONS:
        print(f"{season}: downloading...")
        data = fetch_season(season)
        if data is not None:
            path = RAW_DIR / f"{API_LEAGUE}_{season}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            print(f" saved ({len(data['matches'])} matches)")
        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()