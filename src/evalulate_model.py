from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)


TARGET_COLUMN = "home_win"


def load_model_bundle(model_path: Path) -> tuple[object, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}. "
            "Run src/train_model.py first."
        )

    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        if "model" not in bundle or "feature_columns" not in bundle:
            raise ValueError(
                "Model bundle must contain 'model' and 'feature_columns'."
            )

        return bundle["model"], list(bundle["feature_columns"])

    raise ValueError(
        "Expected a joblib dictionary with 'model' and 'feature_columns'."
    )


def load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {features_path}")

    features = pd.read_csv(features_path)

    if TARGET_COLUMN not in features.columns:
        raise ValueError(f"Feature file is missing target column: {TARGET_COLUMN}")

    return features


def split_evaluation_data(
    features: pd.DataFrame,
    feature_columns: list[str],
    test_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = set(feature_columns).difference(features.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature file is missing model columns: {missing_text}")

    model_df = features.dropna(
        subset=[TARGET_COLUMN, *feature_columns],
    ).copy()

    train = model_df.loc[model_df["season"] < test_season].copy()
    test = model_df.loc[model_df["season"] == test_season].copy()

    if train.empty:
        raise ValueError(f"No training rows found before season {test_season}.")

    if test.empty:
        raise ValueError(f"No test rows found for season {test_season}.")

    return train, test


def calculate_metrics(
    y_true: pd.Series,
    predicted_probability: pd.Series,
    threshold: float,
) -> dict[str, float]:
    predicted_class = (predicted_probability >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predicted_class),
        "log_loss": log_loss(y_true, predicted_probability),
        "brier_score": brier_score_loss(y_true, predicted_probability),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy:    {metrics['accuracy']:.3f}")
    print(f"Log loss:    {metrics['log_loss']:.3f}")
    print(f"Brier score: {metrics['brier_score']:.3f}")


def build_calibration_table(
    y_true: pd.Series,
    predicted_probability: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    calibration = pd.DataFrame(
        {
            "actual": y_true.reset_index(drop=True),
            "predicted_probability": predicted_probability.reset_index(drop=True),
        }
    )

    calibration["probability_bin"] = pd.cut(
        calibration["predicted_probability"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    return (
        calibration.groupby("probability_bin", observed=True)
        .agg(
            games=("actual", "size"),
            avg_predicted_probability=("predicted_probability", "mean"),
            actual_home_win_rate=("actual", "mean"),
        )
        .reset_index(drop=True)
    )


def evaluate_model(
    model_path: Path,
    features_path: Path,
    test_season: str,
    threshold: float,
) -> None:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be greater than 0 and less than 1.")

    model, feature_columns = load_model_bundle(model_path)
    features = load_features(features_path)
    train, test = split_evaluation_data(
        features=features,
        feature_columns=feature_columns,
        test_season=test_season,
    )

    X_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]

    predicted_probability = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=test.index,
        name="home_win_probability",
    )
    predicted_class = (predicted_probability >= threshold).astype(int)

    baseline_probability = pd.Series(
        train[TARGET_COLUMN].mean(),
        index=test.index,
    )

    model_metrics = calculate_metrics(
        y_true=y_test,
        predicted_probability=predicted_probability,
        threshold=threshold,
    )
    baseline_metrics = calculate_metrics(
        y_true=y_test,
        predicted_probability=baseline_probability,
        threshold=threshold,
    )

    print(f"Model: {model_path}")
    print(f"Features: {features_path}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Train rows before {test_season}: {len(train)}")
    print(f"Test rows in {test_season}: {len(test)}")
    print(f"Decision threshold: {threshold:.2f}")

    print_metrics("Model Metrics", model_metrics)
    print_metrics("Baseline Metrics", baseline_metrics)

    print("\nConfusion Matrix")
    print("----------------")
    print(confusion_matrix(y_test, predicted_class))

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, predicted_class, digits=3))

    print("\nCalibration Table")
    print("-----------------")
    print(
        build_calibration_table(
            y_true=y_test,
            predicted_probability=predicted_probability,
        ).to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NBA home-win prediction model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/home_win_logreg.joblib"),
        help="Path to the saved joblib model bundle.",
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=Path("Data/processed/nba_games_features.csv"),
        help="Path to the model feature CSV.",
    )

    parser.add_argument(
        "--test-season",
        default="2025-26",
        help="Season to evaluate on.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for converting predictions to classes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_model(
        model_path=args.model,
        features_path=args.features,
        test_season=args.test_season,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
