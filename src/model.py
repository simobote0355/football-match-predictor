import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR
from features import FEATURE_COLS

OUT = PROCESSED_DIR / "powerbi"

# Temporal cutoffs. Set TRAIN_START = "2021-07-01" to train on every season.
TRAIN_START = "2021-07-01"   # before this: burn-in (only warms up Elo and form)
VAL_START = "2024-07-01"
TEST_START = "2025-07-01"
LABELS = ["H", "D", "A"]     # class 0, 1, 2


def load_data():
    """Read the features and label every match with its split."""
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None)

    df["split"] = np.select(
        [~df["played"].astype(bool),
         df["date"] < TRAIN_START,
         df["date"] < VAL_START,
         df["date"] < TEST_START],
        ["pending", "burn_in", "train", "val"],
        default="test",
    )
    df["y"] = df["result"].map({"H": 0, "D": 1, "A": 2}).fillna(-1).astype(int)
    return df


def train_models(df):
    """Fit the 3 models on train. Return their probabilities for every match."""
    X, y = df[FEATURE_COLS], df["y"]
    tr = df["split"] == "train"
    va = df["split"] == "val"

    # Baseline: same probabilities for everyone (class frequencies in train)
    freq = np.bincount(y[tr], minlength=3) / tr.sum()

    # Logistic regression (imputer and scaler are fitted on train only)
    lr = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    lr.fit(X[tr], y[tr])

    # XGBoost, stops when the validation log loss stops improving
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
    )
    xgb.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

    return {
        "baseline": np.tile(freq, (len(df), 1)),
        "logistic": lr.predict_proba(X),
        "xgboost": xgb.predict_proba(X),
    }


def evaluate(df, probas):
    """Log loss and accuracy per model and per split."""
    y = df["y"].values
    rows = []
    for name, p in probas.items():
        for split in ["train", "val", "test"]:
            m = (df["split"] == split).values
            rows.append({
                "model": name,
                "split": split,
                "log_loss": log_loss(y[m], p[m], labels=[0, 1, 2]),
                "accuracy": accuracy_score(y[m], p[m].argmax(axis=1)),
            })
    return pd.DataFrame(rows)


def calibration_table(df, proba):
    """Predicted probability vs real frequency on the test set (5 bins per class)."""
    m = (df["split"] == "test").values
    y = df["y"].values[m]
    rows = []
    for k, label in enumerate(LABELS):
        frac, mean_pred = calibration_curve(
            (y == k).astype(int), proba[m, k], n_bins=5, strategy="quantile"
        )
        for i, (mp, f) in enumerate(zip(mean_pred, frac), start=1):
            rows.append({"class": label, "bin": i, "mean_pred": mp, "frac_real": f})
    return pd.DataFrame(rows)


def plot_calibration(calib):
    plt.plot([0, 1], [0, 1], "k--", label="perfecta")
    for label in LABELS:
        part = calib[calib["class"] == label]
        plt.plot(part["mean_pred"], part["frac_real"], marker="o", label=label)
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia real")
    plt.title("Calibración XGBoost (test)")
    plt.legend()
    plt.savefig(OUT / "calibration.png", dpi=150)


def export(df, probas, metrics, calib):
    """Write the CSV files that Power BI will read."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_id = {t: i for i, t in enumerate(teams, start=1)}
    pd.DataFrame({"team_id": team_id.values(), "team": team_id.keys()}).to_csv(OUT / "teams.csv", index=False)

    df["home_team_id"] = df["home_team"].map(team_id)
    df["away_team_id"] = df["away_team"].map(team_id)
    cols_export = ["match_id", "date", "home_team_id", "away_team_id", "home_goals", "away_goals", "result", "played", "split"]
    df[cols_export].to_csv(OUT / "matches.csv", index=False)
    df[["match_id"] + FEATURE_COLS].to_csv(OUT / "features.csv", index=False)

    pred = df[["match_id", "split", "result"]].copy()
    pred[["prob_H", "prob_D", "prob_A"]] = probas["xgboost"]
    pred[["lr_prob_H", "lr_prob_D", "lr_prob_A"]] = probas["logistic"]
    pred["pred"] = np.array(LABELS)[probas["xgboost"].argmax(axis=1)]
    pred.to_csv(OUT / "predictions.csv", index=False)

    metrics.to_csv(OUT / "metrics.csv", index=False)
    calib.to_csv(OUT / "calibration.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    probas = train_models(df)

    metrics = evaluate(df, probas)
    print(metrics.round(4).to_string(index=False))

    calib = calibration_table(df, probas["xgboost"])
    plot_calibration(calib)

    export(df, probas, metrics, calib)
    print("CSV exported to", OUT)


if __name__ == "__main__":
    main()