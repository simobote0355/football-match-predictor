"""Freeze the next matchday's predictions and score them once results exist.

    python src/track_predictions.py log     # append upcoming matches to the log (never overwrites)
    python src/track_predictions.py score   # compare the log against real results

Weekly routine:
    1. python src/ingest_api.py && python src/clean.py && python src/features.py && python src/model.py   (refresh)
    2. python src/track_predictions.py score                                   (score last matchday)
    3. python src/track_predictions.py log                                     (freeze the next one)
    4. git add data/processed/predictions_log.csv && git commit && git push    (BEFORE kickoff)

The commit date is the proof that the predictions were made before the matches.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import PROCESSED_DIR

LOG_PATH = PROCESSED_DIR / "predictions_log.csv"
PRED_PATH = PROCESSED_DIR / "powerbi" / "predictions.csv"

# match_id is just the row number after sorting by date, so it changes if a
# fixture is rescheduled. This key does not.
KEY = ["season", "home_team", "away_team"]
MODELS = {
    "xgboost": ["prob_H", "prob_D", "prob_A"],
    "logistic": ["lr_prob_H", "lr_prob_D", "lr_prob_A"],
}
LOG_COLS = ["logged_at", "season", "date", "home_team", "away_team",
            "prob_H", "prob_D", "prob_A", "lr_prob_H", "lr_prob_D", "lr_prob_A", "pred", "note"]


def log_next_matches(days=7):
    """Append pending matches in the next `days` days that are not logged yet."""
    feats = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    pred = pd.read_csv(PRED_PATH)
    df = feats[["match_id", "date", "season", "home_team", "away_team", "played"]].merge(
        pred.drop(columns=["split", "result"]), on="match_id")

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    # Stale-data guard: if past matches still have no result, features and
    # predictions were built before they were played, so freezing them is wrong.
    overdue = df[(~df["played"]) & (df["date"] < today)]
    if len(overdue) > 0 and "--force" not in sys.argv:
        raise SystemExit(
            f"{len(overdue)} matches dated before today have no result yet.\n"
            "Refresh first: ingest_api.py -> clean.py -> model.py, then run this again.\n"
            "(Use --force only if those matches were postponed.)"
        )

    new = df[(~df["played"]) & (df["date"] > today) & (df["date"] <= today + pd.Timedelta(days=days))].copy()

    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        seen = new[KEY].merge(old[KEY], on=KEY, how="left", indicator=True)["_merge"].values == "both"
        new = new[~seen]   # frozen: a match already in the log is never touched
    else:
        old = pd.DataFrame(columns=LOG_COLS)

    if new.empty:
        print("Nothing new to log.")
        return

    new["logged_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    new["note"] = ""
    out = pd.concat([old, new[LOG_COLS]], ignore_index=True)
    out.to_csv(LOG_PATH, index=False)
    print(f"Logged {len(new)} matches. Commit {LOG_PATH.name} now, before kickoff.")


def wilson_interval(k, n, z=1.96):
    """95% interval for a proportion; works with small n."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def score_table(d):
    """Accuracy and log loss per model for matches that already have a result."""
    y = d["result"].map({"H": 0, "D": 1, "A": 2}).values
    n = len(d)
    rows = []
    for name, cols in MODELS.items():
        p = d[cols].values
        p = p / p.sum(axis=1, keepdims=True)   # CSV rounding: make each row sum exactly 1
        hits = int((p.argmax(axis=1) == y).sum())
        low, high = wilson_interval(hits, n)
        rows.append({"model": name, "n": n, "hits": hits, "accuracy": hits / n,
                     "acc_low": low, "acc_high": high,
                     "log_loss": log_loss(y, p, labels=[0, 1, 2])})
    home = int((y == 0).sum())
    rows.append({"model": "always home", "n": n, "hits": home, "accuracy": home / n,
                 "acc_low": wilson_interval(home, n)[0], "acc_high": wilson_interval(home, n)[1],
                 "log_loss": np.nan})
    rows.append({"model": "uniform 1/3", "n": n, "hits": np.nan, "accuracy": np.nan,
                 "acc_low": np.nan, "acc_high": np.nan, "log_loss": np.log(3)})
    return pd.DataFrame(rows)


def score():
    log = pd.read_csv(LOG_PATH)
    feats = pd.read_csv(PROCESSED_DIR / "features.csv")
    results = feats.loc[feats["played"], KEY + ["result"]]
    d = log.merge(results, on=KEY)   # inner join: only matches that already have a result

    print(f"{len(log)} predictions logged, {len(d)} already played.")
    if d.empty:
        print("No results yet. Run ingest_api.py, clean.py and model.py to refresh them.")
        return

    d["hit"] = d[MODELS["xgboost"]].values.argmax(axis=1) == d["result"].map({"H": 0, "D": 1, "A": 2}).values
    print(d[["date", "home_team", "away_team", "pred", "result", "hit"]].to_string(index=False))
    print()
    print(score_table(d).round(3).to_string(index=False))
    if len(d) < 100:
        print(f"\nOnly {len(d)} matches: intervals are very wide. Judge by log loss once you have 100+.")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "log":
        log_next_matches()
    elif command == "score":
        score()
    else:
        raise SystemExit("Usage: python src/track_predictions.py [log|score]")
