import numpy as np
import pandas as pd
from config import PROCESSED_DIR

WINDOW = 5          # matches used for the short rolling features
MIN_GAMES = 3       # minimum past matches to compute a short rolling value
LONG_WINDOW = 10    # matches used for the draw / goals features
LONG_MIN_GAMES = 5

# Team-level columns (computed per team, then copied as home_<col> and away_<col>)
ROLL_COLS = [
    'form_5', 'gf_5', 'ga_5', 'venue_form_5', 'rest_days',      # base
    'prev_ppg', 'is_promoted',                                   # start
    'ppg_season', 'gd_season', 'games_season',                   # table
    'draw_10', 'cs_10', 'fts_10', 'tg_10',                       # draw
    'opp_elo_5', 'overperf_5',                                   # opp
    'elo_trend_5',                                               # elo
    'games_14d', 'games_30d',                                    # calendar
]

# Feature groups. Turn them on one at a time in USE_GROUPS and compare the
# VALIDATION log loss (model.py prints it). Do not choose groups by the test set.
GROUPS = {
    "base": [
        "home_form_5", "away_form_5",
        "home_gf_5", "away_gf_5",
        "home_ga_5", "away_ga_5",
        "home_venue_form_5", "away_venue_form_5",
        "home_rest_days", "away_rest_days",
        "elo_home", "elo_away", "elo_diff",
        "h2h_points",
        "form_diff", "rest_diff",
    ],
    # Where each team starts the season (helps when rolling features are empty)
    "start": [
        "home_prev_ppg", "away_prev_ppg", "prev_ppg_diff",
        "home_is_promoted", "away_is_promoted",
    ],
    # Season-to-date level, and how many games that level is based on
    "table": [
        "home_ppg_season", "away_ppg_season", "ppg_season_diff",
        "home_gd_season", "away_gd_season",
        "home_games_season", "away_games_season",
    ],
    # Draw and low-scoring tendency (last 10 matches)
    "draw": [
        "home_draw_10", "away_draw_10",
        "home_cs_10", "away_cs_10",
        "home_fts_10", "away_fts_10",
        "home_tg_10", "away_tg_10",
        "abs_elo_diff",
    ],
    # Quality of the opponents faced, and results vs what Elo expected
    "opp": [
        "home_opp_elo_5", "away_opp_elo_5",
        "home_overperf_5", "away_overperf_5",
    ],
    # Elo momentum
    "elo": [
        "home_elo_trend_5", "away_elo_trend_5",
    ],
    # Schedule congestion (Premier League matches only)
    "calendar": [
        "home_games_14d", "away_games_14d",
        "home_games_30d", "away_games_30d",
        "is_midweek",
    ],
}

USE_GROUPS = ["base"]   # groups the model uses

FEATURE_COLS = [c for g in USE_GROUPS for c in GROUPS[g]]
ALL_COLS = [c for cols in GROUPS.values() for c in cols]   # used by check_leakage.py


def build_long(df):
    """One row per team per match (2 rows per match). Needs the Elo columns."""
    cols = ['match_id', 'date', 'season', 'team', 'opponent', 'goals_for', 'goals_against',
            'team_elo', 'opp_elo', 'exp_score']

    home = df[['match_id', 'date', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals',
               'elo_home', 'elo_away', 'exp_home']].copy()
    home.columns = cols
    home['is_home'] = 1

    away = df[['match_id', 'date', 'season', 'away_team', 'home_team', 'away_goals', 'home_goals',
               'elo_away', 'elo_home', 'exp_home']].copy()
    away.columns = cols
    away['exp_score'] = 1 - away['exp_score']   # expected score of the away team
    away['is_home'] = 0

    long = pd.concat([home, away], ignore_index=True)
    long[['goals_for', 'goals_against']] = long[['goals_for', 'goals_against']].astype(float)

    gf, ga = long['goals_for'], long['goals_against']
    long['points'] = np.select([gf > ga, gf == ga, gf < ga], [3, 1, 0], default=np.nan)
    long['gd'] = gf - ga

    return long.sort_values(['team', 'date', 'match_id']).reset_index(drop=True)


def past_mean(s, window=WINDOW, min_games=MIN_GAMES):
    """Mean of the previous `window` values. shift(1) excludes the current match."""
    return s.shift(1).rolling(window, min_periods=min_games).mean()


def past_mean_long(s):
    return past_mean(s, LONG_WINDOW, LONG_MIN_GAMES)


def count_recent(dates, days):
    """For each match, how many of the team's matches fall in the previous `days` days."""
    d = dates.values
    before = np.searchsorted(d, d, side="left")                              # matches before this date
    start = np.searchsorted(d, d - np.timedelta64(days, "D"), side="left")   # matches before the window
    return pd.Series(before - start, index=dates.index)


def add_start_features(long):
    """Last season's points per game and promoted flag. Only uses the previous season."""
    seasons = sorted(long["season"].unique())
    prev_of = dict(zip(seasons[1:], seasons[:-1]))
    prev_season = long["season"].map(prev_of)    # NaN in the first season of the data

    season_ppg = long.groupby(["team", "season"])["points"].mean()
    keys = pd.MultiIndex.from_arrays([long["team"], prev_season])
    long["prev_ppg"] = season_ppg.reindex(keys).values

    # Promoted = did not play in the previous season. Unknown (NaN) in the first season.
    long["is_promoted"] = np.where(prev_season.isna(), np.nan, long["prev_ppg"].isna().astype(float))
    return long


def add_rolling_features(long):
    played = long["points"].notna()
    long["score"] = long["points"].map({3: 1.0, 1: 0.5, 0: 0.0})   # Elo-style result
    long["resid"] = long["score"] - long["exp_score"]               # result minus what Elo expected
    long["draw"] = (long["points"] == 1).astype(float).where(played)
    long["clean_sheet"] = (long["goals_against"] == 0).astype(float).where(played)
    long["failed_score"] = (long["goals_for"] == 0).astype(float).where(played)
    long["total_goals"] = long["goals_for"] + long["goals_against"]

    by_team = long.groupby("team")

    # base
    long["form_5"] = by_team["points"].transform(past_mean)
    long["gf_5"] = by_team["goals_for"].transform(past_mean)
    long["ga_5"] = by_team["goals_against"].transform(past_mean)
    long["rest_days"] = by_team["date"].diff().dt.days.clip(upper=14)
    long["venue_form_5"] = long.groupby(["team", "is_home"])["points"].transform(past_mean)

    # draw
    long["draw_10"] = by_team["draw"].transform(past_mean_long)
    long["cs_10"] = by_team["clean_sheet"].transform(past_mean_long)
    long["fts_10"] = by_team["failed_score"].transform(past_mean_long)
    long["tg_10"] = by_team["total_goals"].transform(past_mean_long)

    # opp: quality of past opponents and results vs expectation
    long["opp_elo_5"] = by_team["opp_elo"].transform(past_mean)
    long["overperf_5"] = by_team["resid"].transform(past_mean)

    # elo: Elo before this match minus Elo before the match 5 games ago
    long["elo_trend_5"] = long["team_elo"] - by_team["team_elo"].shift(5)

    # table: season-to-date, only matches before this one
    by_season = long.groupby(["team", "season"])
    long["ppg_season"] = by_season["points"].transform(lambda s: s.shift(1).expanding().mean())
    long["gd_season"] = by_season["gd"].transform(lambda s: s.shift(1).expanding().mean())
    long["games_season"] = by_season["points"].transform(lambda s: s.shift(1).expanding().count())

    # calendar
    long["games_14d"] = by_team["date"].transform(lambda s: count_recent(s, 14))
    long["games_30d"] = by_team["date"].transform(lambda s: count_recent(s, 30))

    return add_start_features(long)


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
    pre_home, pre_away, exp_home = [], [], []

    for row in df.itertuples():
        elo_h = elo.get(row.home_team, start)
        elo_a = elo.get(row.away_team, start)
        expected_home = 1 / (1 + 10 ** ((elo_a - (elo_h + home_adv)) / 400))
        pre_home.append(elo_h)
        pre_away.append(elo_a)
        exp_home.append(expected_home)

        if not row.played:
            continue

        actual_home = {"H": 1, "D": 0.5, "A": 0}[row.result]
        change = k * (actual_home - expected_home)
        elo[row.home_team] = elo_h + change
        elo[row.away_team] = elo_a - change

    df["elo_home"] = pre_home
    df["elo_away"] = pre_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["exp_home"] = exp_home   # Elo's expected score for the home team (used by overperf_5)

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

    df = add_elo(df)    # before the long table: it needs each team's Elo
    df = add_h2h(df)
    long = add_rolling_features(build_long(df))
    df = merge_long_features(df, long)

    df["form_diff"] = df["home_form_5"] - df["away_form_5"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["prev_ppg_diff"] = df["home_prev_ppg"] - df["away_prev_ppg"]
    df["ppg_season_diff"] = df["home_ppg_season"] - df["away_ppg_season"]
    df["abs_elo_diff"] = df["elo_diff"].abs()
    df["is_midweek"] = df["date"].dt.dayofweek.isin([1, 2, 3]).astype(int)   # Tue-Thu
    return df


def main():
    df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
    df = build_features(df)
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)

    print(len(df), "rows")
    print(df[ALL_COLS].isna().sum())


if __name__ == "__main__":
    main()
