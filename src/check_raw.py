import json
import pandas as pd
from config import RAW_DIR, LEAGUE, CSV_SEASONS, API_SEASONS

# CSV
for code in CSV_SEASONS:
    path = RAW_DIR / f"{LEAGUE}_{code}.csv"

    if not path.exists():
        print(f"{code}: MISSING: {path.name}")
        continue

    df = pd.read_csv(path, encoding="latin-1").dropna(subset=["HomeTeam"])
    print(f"{code}: {len(df)} rows | From {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

# API
for season in API_SEASONS:
    path = RAW_DIR / f"{LEAGUE}_{season}.json"

    if not path.exists():
        print(f"{season} (API): MISSING -> {path.name}")
        continue

    matches = json.loads(path.read_text(encoding="utf-8"))["matches"]
    finished = sum(m["status"] == "FINISHED" for m in matches)
    print(f"{season} (API): {len(matches)} matches, {finished} finished")