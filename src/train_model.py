from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


FEATURE_COLUMNS = [
    "home_season_win_pct_before",
    "away_season_win_pct_before",
    "home_last5_win_pct",
    "away_last5_win_pct",
    "home_last5_point_diff",
    "away_last5_point_diff",
    "home_rest_days",
    "away_rest_days",
    "season_win_pct_before_diff",
    "last5_win_pct_diff",
    "last5_point_diff_diff",
    "rest_days_diff",
]


def main():
    features = pd.read_csv("Data/processed/nba_games_features.csv")

    model_df = features.dropna(subset=FEATURE_COLUMNS + ["home_win"]).copy()

    train = model_df[model_df["season"] < "2025-26"]
    test = model_df[model_df["season"] == "2025-26"]

    X_train = train[FEATURE_COLUMNS]
    y_train = train["home_win"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["home_win"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred_class):.3f}")
    print(f"Log loss: {log_loss(y_test, pred_proba):.3f}")
    print(f"Brier score: {brier_score_loss(y_test, pred_proba):.3f}")

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        "models/home_win_logreg.joblib",
    )


if __name__ == "__main__":
    main()