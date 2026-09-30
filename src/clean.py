import json
import pandas as pd
from config import RAW_DIR, PROCESSED_DIR, LEAGUE, CSV_SEASONS, API_SEASONS


TEAM_MAPPING = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Norwich": "Norwich City",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Leeds": "Leeds United",
    "Brighton": "Brighton & Hove Albion",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
    "Leicester": "Leicester City",
}


def load_csv():
    dfs = []

    for season in CSV_SEASONS:
        path = RAW_DIR / f"{LEAGUE}_{season}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}")

        aux = pd.read_csv(
            path,
            encoding="latin-1"
        ).dropna(subset=["HomeTeam"])

        aux["season"] = season
        aux["played"] = True
        aux["source"] = "csv"

        aux = aux[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "season", "played", "source"]]

        dfs.append(aux)

    return pd.concat(dfs, ignore_index=True)


def load_api():
    dfs = []

    for season in API_SEASONS:
        path = RAW_DIR / f"{LEAGUE}_{season}.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}")

        matches = json.loads(
            path.read_text(encoding="utf-8")
        )["matches"]

        aux = pd.json_normalize(matches)

        aux = aux.rename(columns={
            "utcDate": "Date",
            "homeTeam.name": "HomeTeam",
            "awayTeam.name": "AwayTeam",
            "score.fullTime.home": "FTHG",
            "score.fullTime.away": "FTAG",
            "score.winner": "FTR",
            "status": "status",
        })

        aux["season"] = '2627'
        aux["played"] = aux["status"].eq("FINISHED")
        aux["source"] = "api"

        aux = aux[["Date", "HomeTeam","AwayTeam", "FTHG", "FTAG", "FTR", "season", "played", "source"]]
        aux["FTR"] = aux["FTR"].map({
            "HOME_TEAM": "H",
            "AWAY_TEAM": "A",
            "DRAW": "D",
        })

        dfs.append(aux)

    return pd.concat(dfs, ignore_index=True)


def clean_data(df):
    df.loc[~df["played"], "FTR"] = pd.NA

    df["Date"] = pd.to_datetime(df["Date"].str[:10], format="%Y-%m-%d")

    for column in ["HomeTeam", "AwayTeam"]:
        df[column] = df[column].str.replace(r"\s*(AFC|FC)\b", "", regex=True, case=False).str.strip().replace(TEAM_MAPPING)

    df["FTHG"] = df["FTHG"].astype("Int64")
    df["FTAG"] = df["FTAG"].astype("Int64")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def main():
    df_csv = load_csv()
    df_api = load_api()

    df = pd.concat([df_csv, df_api], ignore_index=True)

    df = clean_data(df)

    df.columns = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'season', 'played', 'source']

    df.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)


if __name__ == "__main__":
    main()