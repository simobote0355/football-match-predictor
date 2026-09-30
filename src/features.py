import numpy as np
import pandas as pd
from config import PROCESSED_DIR

WINDOW = 5      # matches used for rolling feature
MIN_GAMES = 3   # minimum past mataches to compute a rolling value

ROLL_COLS = ['form_5', 'gf_5', 'ga_5', 'venue_form_5', 'rest_days']

FEATURE_COLS = [
    "home_form_5", "away_form_5",
    "home_gf_5", "away_gf_5",
    "home_ga_5", "away_ga_5",
    "home_venue_form_5", "away_venue_form_5",
    "home_rest_days", "away_rest_days",
    "elo_home", "elo_away", "elo_diff",
    "h2h_points",
    "form_diff", "rest_diff",
]


def build_long(df):
    """One row per team per match (2 rows per match)."""
    cols = ['match_id', 'date', 'team', 'opponent', 'goals_for', 'goals_against']

    home = df[['match_id', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals']].copy()
    home.columns = cols
    home['is_home'] = 1

    away = df[["match_id", "date", "away_team", "home_team", "away_goals", "home_goals"]].copy()
    away.columns = cols
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long[['goals_for', 'goals_against']] = long[['goals_for', 'goals_against']].astype(float)

    gf, ga = long['goals_for'], long['goals_against']
    long['points'] = np.select([gf > ga, gf == ga, gf <ga], [3, 1, 0], default=np.nan)

    return long.sort_values(['team', 'date', 'match_id']).reset_index(drop=True)


def past_mean(s):
    """Mean of the previous WINDOW values. shift(1) excludes the current match."""
    return s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean()


def add_rolling_features(long):
    by_team = long.groupby("team")
    long["form_5"] = by_team["points"].transform(past_mean)
    long["gf_5"] = by_team["goals_for"].transform(past_mean)
    long["ga_5"] = by_team["goals_against"].transform(past_mean)
    long["rest_days"] = by_team["date"].diff().dt.days.clip(upper=14)
    long["venue_form_5"] = long.groupby(["team", "is_home"])["points"].transform(past_mean)
    return long


def merge_long_features(df, long):
    """Bring the team-level features back to one row per match."""
    home = long[long["is_home"] == 1][["match_id"] + ROLL_COLS]
    away = long[long["is_home"] == 0][["match_id"] + ROLL_COLS]
    home = home.rename(columns={c: f"home_{c}" for c in ROLL_COLS})
    away = away.rename(columns={c: f"away_{c}" for c in ROLL_COLS})
    return df.merge(home, on="match_id").merge(away, on="match_id")


def add_elo(df, k=20, home_adv=60, start=1500):
    """Elo BEFORE each match. The update happens after saving the value."""
    elo = {}
    pre_home, pre_away = [], []

    for row in df.itertuples():
        elo_h = elo.get(row.home_team, start)
        elo_a = elo.get(row.away_team, start)
        pre_home.append(elo_h)
        pre_away.append(elo_a)

        if not row.played:
            continue

        expected_home = 1 / (1 + 10 ** ((elo_a - (elo_h + home_adv)) / 400))
        actual_home = {"H": 1, "D": 0.5, "A": 0}[row.result]
        change = k * (actual_home - expected_home)
        elo[row.home_team] = elo_h + change
        elo[row.away_team] = elo_a - change

    df["elo_home"] = pre_home
    df["elo_away"] = pre_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    
    return df


def add_h2h(df, n=3):
    """Home team's average points in the last n previous meetings."""
    history = {}
    values = []

    for row in df.itertuples():
        key = tuple(sorted([row.home_team, row.away_team]))
        past = history.get(key, [])[-n:]

        points = []
        for past_home, past_result in past:
            if past_result == "D":
                points.append(1)
            elif (past_result == "H") == (past_home == row.home_team):
                points.append(3)
            else:
                points.append(0)
        values.append(np.mean(points) if points else np.nan)

        if row.played:   # added AFTER computing the value
            history.setdefault(key, []).append((row.home_team, row.result))

    df["h2h_points"] = values
    return df


def build_features(df):
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    df["match_id"] = df.index

    long = add_rolling_features(build_long(df))
    df = merge_long_features(df, long)
    df = add_elo(df)
    df = add_h2h(df)

    df["form_diff"] = df["home_form_5"] - df["away_form_5"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    return df

def main():
    df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    df = build_features(df)
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)

    print(len(df), "rows")
    print(df[FEATURE_COLS].isna().sum())


if __name__ == "__main__":
    main()