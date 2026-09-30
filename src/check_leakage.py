import pandas as pd
from config import PROCESSED_DIR
from features import build_features, FEATURE_COLS

df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", parse_dates=["date"])
base = build_features(df)

# 1. Pick a played match in the middle of the data
i = base[base["played"]].index[1500]
row = base.loc[i]
print("Match:", row["date"].date(), row["home_team"], "vs", row["away_team"], "->", row["result"])

# 2. Invent a different result for that match only
changed = df.copy()
mask = (
    (changed["date"] == row["date"])
    & (changed["home_team"] == row["home_team"])
    & (changed["away_team"] == row["away_team"])
)
changed.loc[mask, "home_goals"] = 0
changed.loc[mask, "away_goals"] = 5
changed.loc[mask, "result"] = "A"
alt = build_features(changed)

# 3. Its own features must NOT change
same = base.loc[i, FEATURE_COLS].equals(alt.loc[i, FEATURE_COLS])
print("Features of that match unchanged:", same)

# 4. Only LATER matches may change
diff = (base[FEATURE_COLS].fillna(-999) != alt[FEATURE_COLS].fillna(-999)).any(axis=1)
first_changed = base.loc[diff, "date"].min()
print("First match affected:", first_changed.date())
print("All affected matches are later:", first_changed > row["date"])