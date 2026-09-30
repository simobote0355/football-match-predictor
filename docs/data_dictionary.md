# Data dictionary: matches_clean.csv

One row per Premier League match, sorted chronologically.
Produced by `src/clean.py` from two sources: football-data.co.uk (CSV,
historical seasons) and football-data.org (API, current season).

| Column | Type | Description |
|---|---|---|
| date | date | Match day (no time, no time zone) |
| home_team | text | Home team, name standardized across sources |
| away_team | text | Away team, name standardized across sources |
| home_goals | integer (null if not played) | Home goals at full time |
| away_goals | integer (null if not played) | Away goals at full time |
| result | text (null if not played) | H = home win, D = draw, A = away win |
| season | text | Season code: 2122 = 2021-22, ..., 2526 = 2025-26, 2627 = 2026-27 |
| played | boolean | True if the match has finished |
| source | text | csv (football-data.co.uk) or api (football-data.org) |

## Notes

- Season 2627 comes from the API and is incomplete: rows with
  `played = False` are upcoming matches, the ones the model will predict.
- Unplayed matches have no goals and no result (null values).
- Team names follow a single format (for example "Manchester United",
  "Norwich City"), so the same team is never spelled two ways.
- Rows are sorted by `date`. Features in `features.py` must be computed
  using only rows with an earlier date than the match being predicted.

## Naming of raw files

- CSV files: `PL_<code>.csv`, where the code joins the two years of the
  season (`PL_2122.csv` = 2021-22).
- API file: `PL_2026.json`, where 2026 is the starting year of the season
  (2026-27).