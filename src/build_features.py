from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "game_id",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "home_win",
    "PTS_home",
    "PTS_away",
    "PLUS_MINUS_home",
    "PLUS_MINUS_away",
}


TEAM_HISTORY_FEATURES = [
    "games_played_before",
    "season_win_pct_before",
    "season_avg_point_diff_before",
    "season_avg_points_for_before",
    "season_avg_points_against_before",
    "last5_win_pct",
    "last5_point_diff",
    "last5_points_for",
    "last5_points_against",
    "last10_win_pct",
    "last10_point_diff",
    "last10_points_for",
    "last10_points_against",
    "rest_days",
    "is_back_to_back",
]


def load_clean_games(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    LOGGER.info("Loading cleaned games from %s", input_path)

    games = pd.read_csv(
        input_path,
        dtype={
            "game_id": "string",
            "season": "string",
            "home_team": "string",
            "away_team": "string",
        },
    )

    LOGGER.info("Loaded %s games and %s columns", len(games), len(games.columns))

    return games


def validate_required_columns(games: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(games.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The cleaned dataset is missing required columns: {missing_text}"
        )


def standardize_games(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()

    games["game_date"] = pd.to_datetime(
        games["game_date"],
        errors="raise",
    )

    games["home_win"] = pd.to_numeric(
        games["home_win"],
        errors="raise",
    ).astype("int8")

    numeric_columns = [
        "PTS_home",
        "PTS_away",
        "PLUS_MINUS_home",
        "PLUS_MINUS_away",
    ]

    for column in numeric_columns:
        games[column] = pd.to_numeric(
            games[column],
            errors="raise",
        )

    games = games.sort_values(
        ["game_date", "game_id"],
    ).reset_index(drop=True)

    return games


def build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    home = games[
        [
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_win",
            "PTS_home",
            "PTS_away",
            "PLUS_MINUS_home",
        ]
    ].copy()

    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "PTS_home": "points_for",
            "PTS_away": "points_against",
            "PLUS_MINUS_home": "point_diff",
        }
    )

    home["is_home"] = 1
    home["win"] = home["home_win"]

    away = games[
        [
            "game_id",
            "game_date",
            "season",
            "away_team",
            "home_team",
            "home_win",
            "PTS_away",
            "PTS_home",
            "PLUS_MINUS_away",
        ]
    ].copy()

    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "PTS_away": "points_for",
            "PTS_home": "points_against",
            "PLUS_MINUS_away": "point_diff",
        }
    )

    away["is_home"] = 0
    away["win"] = 1 - away["home_win"]

    team_games = pd.concat(
        [home, away],
        ignore_index=True,
    )

    team_games = team_games.sort_values(
        ["season", "team", "game_date", "game_id"],
    ).reset_index(drop=True)

    return team_games


def add_team_history_features(team_games: pd.DataFrame) -> pd.DataFrame:
    team_games = team_games.copy()
    team_group = team_games.groupby(["season", "team"], sort=False)

    team_games["games_played_before"] = team_group.cumcount()

    wins_before = team_group["win"].cumsum() - team_games["win"]
    point_diff_before = (
        team_group["point_diff"].cumsum() - team_games["point_diff"]
    )
    points_for_before = (
        team_group["points_for"].cumsum() - team_games["points_for"]
    )
    points_against_before = (
        team_group["points_against"].cumsum() - team_games["points_against"]
    )

    games_before = team_games["games_played_before"].replace(0, pd.NA)

    team_games["season_win_pct_before"] = wins_before / games_before
    team_games["season_avg_point_diff_before"] = (
        point_diff_before / games_before
    )
    team_games["season_avg_points_for_before"] = (
        points_for_before / games_before
    )
    team_games["season_avg_points_against_before"] = (
        points_against_before / games_before
    )

    team_games["last5_win_pct"] = rolling_mean_before(team_group["win"], 5)
    team_games["last5_point_diff"] = rolling_mean_before(
        team_group["point_diff"],
        5,
    )
    team_games["last5_points_for"] = rolling_mean_before(
        team_group["points_for"],
        5,
    )
    team_games["last5_points_against"] = rolling_mean_before(
        team_group["points_against"],
        5,
    )

    team_games["last10_win_pct"] = rolling_mean_before(team_group["win"], 10)
    team_games["last10_point_diff"] = rolling_mean_before(
        team_group["point_diff"],
        10,
    )
    team_games["last10_points_for"] = rolling_mean_before(
        team_group["points_for"],
        10,
    )
    team_games["last10_points_against"] = rolling_mean_before(
        team_group["points_against"],
        10,
    )

    team_games["rest_days"] = team_group["game_date"].diff().dt.days
    team_games["is_back_to_back"] = team_games["rest_days"].eq(1).astype("int8")

    return team_games


def rolling_mean_before(
    grouped_series: pd.core.groupby.SeriesGroupBy,
    window: int,
) -> pd.Series:
    return grouped_series.transform(
        lambda series: series.shift(1).rolling(
            window,
            min_periods=1,
        ).mean()
    )


def merge_team_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    features = games[
        [
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_win",
        ]
    ].copy()

    merge_columns = ["game_id", "team", *TEAM_HISTORY_FEATURES]

    home_features = (
        team_games.loc[team_games["is_home"].eq(1), merge_columns]
        .copy()
        .add_prefix("home_")
    )

    away_features = (
        team_games.loc[team_games["is_home"].eq(0), merge_columns]
        .copy()
        .add_prefix("away_")
    )

    features = features.merge(
        home_features,
        left_on=["game_id", "home_team"],
        right_on=["home_game_id", "home_team"],
        how="left",
        validate="one_to_one",
    )

    features = features.merge(
        away_features,
        left_on=["game_id", "away_team"],
        right_on=["away_game_id", "away_team"],
        how="left",
        validate="one_to_one",
    )

    features = features.drop(
        columns=["home_game_id", "away_game_id"],
        errors="ignore",
    )

    return features


def add_matchup_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    difference_features = [
        "games_played_before",
        "season_win_pct_before",
        "season_avg_point_diff_before",
        "season_avg_points_for_before",
        "season_avg_points_against_before",
        "last5_win_pct",
        "last5_point_diff",
        "last5_points_for",
        "last5_points_against",
        "last10_win_pct",
        "last10_point_diff",
        "last10_points_for",
        "last10_points_against",
        "rest_days",
        "is_back_to_back",
    ]

    for feature in difference_features:
        features[f"{feature}_diff"] = (
            features[f"home_{feature}"] - features[f"away_{feature}"]
        )

    return features


def validate_features(features: pd.DataFrame) -> None:
    if features.empty:
        raise ValueError("The feature dataset is empty.")

    if features["game_id"].duplicated().any():
        raise ValueError("The feature dataset contains duplicated game IDs.")

    if not features["home_win"].isin([0, 1]).all():
        raise ValueError("home_win contains values other than 0 and 1.")

    missing_team_features = features[
        [
            "home_games_played_before",
            "away_games_played_before",
            "home_season_win_pct_before",
            "away_season_win_pct_before",
        ]
    ].isna()

    if missing_team_features.all(axis=1).any():
        raise ValueError(
            "At least one game failed to receive both home and away features."
        )


def order_columns(features: pd.DataFrame) -> pd.DataFrame:
    first_columns = [
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "home_win",
    ]

    existing_first_columns = [
        column
        for column in first_columns
        if column in features.columns
    ]

    remaining_columns = [
        column
        for column in features.columns
        if column not in existing_first_columns
    ]

    return features[existing_first_columns + remaining_columns]


def build_features(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    games = load_clean_games(input_path)

    validate_required_columns(games)

    games = standardize_games(games)

    team_games = build_team_games(games)
    team_games = add_team_history_features(team_games)

    features = merge_team_features(games, team_games)
    features = add_matchup_features(features)
    features = order_columns(features)

    validate_features(features)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info(
        "Saved %s feature rows to %s",
        len(features),
        output_path,
    )

    LOGGER.info(
        "Missing values by feature:\n%s",
        features.isna().sum().loc[lambda counts: counts.gt(0)].to_string(),
    )

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe pre-game NBA modeling features from cleaned games."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/processed/nba_games_clean.csv"),
        help="Path to the cleaned game-level CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/processed/nba_games_features.csv"),
        help="Path for the model-ready feature CSV.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    build_features(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
