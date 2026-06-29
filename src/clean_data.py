from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "SEASON",
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "PTS",
    "PLUS_MINUS",
}


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    LOGGER.info("Loading raw data from %s", input_path)

    df = pd.read_csv(
        input_path,
        dtype={
            "GAME_ID": "string",
            "TEAM_ID": "string",
            "SEASON_ID": "string",
        },
    )

    LOGGER.info("Loaded %s rows and %s columns", len(df), len(df.columns))

    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The raw dataset is missing required columns: {missing_text}"
        )


def standardize_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    string_columns = [
        "SEASON",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "MATCHUP",
        "WL",
    ]

    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].str.upper()
    df["WL"] = df["WL"].str.upper()

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"],
        errors="raise",
    )

    numeric_columns = [
        "MIN",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
        "PLUS_MINUS",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        LOGGER.warning(
            "Removing %s completely duplicated rows",
            duplicate_count,
        )
        df = df.drop_duplicates().copy()

    return df


def validate_team_rows_per_game(df: pd.DataFrame) -> None:
    rows_per_game = df.groupby("GAME_ID").size()
    invalid_games = rows_per_game[rows_per_game != 2]

    if not invalid_games.empty:
        sample = invalid_games.head(10).to_dict()

        raise ValueError(
            "Some games do not contain exactly two team rows. "
            f"Number of invalid games: {len(invalid_games)}. "
            f"Sample: {sample}"
        )


def handle_missing_shooting_percentages(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    percentage_pairs = [
        ("FG_PCT", "FGA"),
        ("FG3_PCT", "FG3A"),
        ("FT_PCT", "FTA"),
    ]

    for percentage_column, attempt_column in percentage_pairs:
        if (
            percentage_column not in df.columns
            or attempt_column not in df.columns
        ):
            continue

        expected_missing = (
            df[percentage_column].isna()
            & df[attempt_column].eq(0)
        )

        df.loc[expected_missing, percentage_column] = 0.0

        unexpected_missing = (
            df[percentage_column].isna()
            & df[attempt_column].gt(0)
        )

        if unexpected_missing.any():
            bad_rows = df.loc[
                unexpected_missing,
                [
                    "GAME_ID",
                    "TEAM_ABBREVIATION",
                    attempt_column,
                    percentage_column,
                ],
            ].head(10)

            raise ValueError(
                f"{percentage_column} is missing even though "
                f"{attempt_column} is greater than zero.\n"
                f"{bad_rows.to_string(index=False)}"
            )

    return df


def add_location_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_home"] = df["MATCHUP"].str.contains(
        "vs.",
        regex=False,
        na=False,
    )

    home_rows_per_game = df.groupby("GAME_ID")["is_home"].transform("sum")

    df["is_neutral"] = home_rows_per_game.ne(1)

    neutral_game_count = df.loc[
        df["is_neutral"],
        "GAME_ID",
    ].nunique()

    if neutral_game_count > 0:
        LOGGER.warning(
            "Found %s games without exactly one identifiable home row",
            neutral_game_count,
        )

    return df


def remove_neutral_site_games(df: pd.DataFrame) -> pd.DataFrame:
    neutral_game_ids = df.loc[
        df["is_neutral"],
        "GAME_ID",
    ].unique()

    if len(neutral_game_ids) > 0:
        LOGGER.warning(
            "Excluding %s neutral or ambiguous-location games",
            len(neutral_game_ids),
        )

    return df.loc[~df["is_neutral"]].copy()


def validate_game_results(df: pd.DataFrame) -> None:
    wins_per_game = (
        df.assign(is_win=df["WL"].eq("W"))
        .groupby("GAME_ID")["is_win"]
        .sum()
    )

    invalid_win_counts = wins_per_game[wins_per_game != 1]

    if not invalid_win_counts.empty:
        raise ValueError(
            f"{len(invalid_win_counts)} games do not contain exactly "
            "one winning team."
        )

    game_metadata = df.groupby("GAME_ID").agg(
        date_count=("GAME_DATE", "nunique"),
        season_count=("SEASON", "nunique"),
        team_count=("TEAM_ID", "nunique"),
    )

    invalid_metadata = game_metadata[
        (game_metadata["date_count"] != 1)
        | (game_metadata["season_count"] != 1)
        | (game_metadata["team_count"] != 2)
    ]

    if not invalid_metadata.empty:
        raise ValueError(
            f"{len(invalid_metadata)} games have inconsistent dates, "
            "seasons, or team IDs."
        )

    if "PLUS_MINUS" in df.columns:
        plus_minus_sums = df.groupby("GAME_ID")["PLUS_MINUS"].sum()
        invalid_plus_minus = plus_minus_sums[
            ~plus_minus_sums.fillna(0).eq(0)
        ]

        if not invalid_plus_minus.empty:
            raise ValueError(
                f"{len(invalid_plus_minus)} games have team PLUS_MINUS "
                "values that do not sum to zero."
            )


def convert_to_game_level(df: pd.DataFrame) -> pd.DataFrame:
    home = df.loc[df["is_home"]].copy()
    away = df.loc[~df["is_home"]].copy()

    columns_to_exclude = {
        "GAME_ID",
        "is_home",
        "is_neutral",
    }

    home_rename_map = {
        column: f"{column}_home"
        for column in home.columns
        if column not in columns_to_exclude
    }

    away_rename_map = {
        column: f"{column}_away"
        for column in away.columns
        if column not in columns_to_exclude
    }

    home = home.rename(columns=home_rename_map)
    away = away.rename(columns=away_rename_map)

    home = home.drop(
        columns=["is_home", "is_neutral"],
        errors="ignore",
    )

    away = away.drop(
        columns=["is_home", "is_neutral"],
        errors="ignore",
    )

    games = home.merge(
        away,
        on="GAME_ID",
        how="inner",
        validate="one_to_one",
    )

    return games


def simplify_game_columns(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()

    rename_map = {
        "GAME_ID": "game_id",
        "GAME_DATE_home": "game_date",
        "SEASON_home": "season",
        "SEASON_ID_home": "season_id",
        "TEAM_ID_home": "home_team_id",
        "TEAM_ID_away": "away_team_id",
        "TEAM_ABBREVIATION_home": "home_team",
        "TEAM_ABBREVIATION_away": "away_team",
        "TEAM_NAME_home": "home_team_name",
        "TEAM_NAME_away": "away_team_name",
    }

    games = games.rename(columns=rename_map)

    redundant_columns = [
        "GAME_DATE_away",
        "SEASON_away",
        "SEASON_ID_away",
        "MATCHUP_home",
        "MATCHUP_away",
    ]

    games = games.drop(
        columns=redundant_columns,
        errors="ignore",
    )

    return games


def create_target(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()

    games["home_win"] = (
        games["PTS_home"] > games["PTS_away"]
    ).astype("int8")

    target_from_wl = games["WL_home"].eq("W").astype("int8")

    if not games["home_win"].equals(target_from_wl):
        inconsistent_games = games.loc[
            games["home_win"] != target_from_wl,
            [
                "game_id",
                "home_team",
                "away_team",
                "PTS_home",
                "PTS_away",
                "WL_home",
                "WL_away",
            ],
        ]

        raise ValueError(
            "The target calculated from points does not match WL_home.\n"
            f"{inconsistent_games.head(10).to_string(index=False)}"
        )

    return games


def validate_cleaned_games(games: pd.DataFrame) -> None:
    if games.empty:
        raise ValueError("The cleaned game dataset is empty.")

    if games["game_id"].duplicated().any():
        raise ValueError("The cleaned dataset contains duplicated game IDs.")

    if games["game_date"].isna().any():
        raise ValueError("The cleaned dataset contains missing game dates.")

    if games["home_team_id"].eq(games["away_team_id"]).any():
        raise ValueError(
            "At least one game has the same home and away team."
        )

    if not games["home_win"].isin([0, 1]).all():
        raise ValueError("home_win contains values other than 0 and 1.")

    if games["WL_home"].eq(games["WL_away"]).any():
        raise ValueError(
            "At least one game has identical home and away WL values."
        )

    expected_margin = games["PTS_home"] - games["PTS_away"]

    if not expected_margin.eq(games["PLUS_MINUS_home"]).all():
        raise ValueError(
            "PTS_home - PTS_away does not always equal PLUS_MINUS_home."
        )

    if not games["PLUS_MINUS_home"].eq(
        -games["PLUS_MINUS_away"]
    ).all():
        raise ValueError(
            "Home and away PLUS_MINUS values are not always opposites."
        )


def order_columns(games: pd.DataFrame) -> pd.DataFrame:
    first_columns = [
        "game_id",
        "game_date",
        "season",
        "season_id",
        "home_team_id",
        "home_team",
        "home_team_name",
        "away_team_id",
        "away_team",
        "away_team_name",
        "home_win",
    ]

    existing_first_columns = [
        column
        for column in first_columns
        if column in games.columns
    ]

    remaining_columns = [
        column
        for column in games.columns
        if column not in existing_first_columns
    ]

    return games[
        existing_first_columns + remaining_columns
    ]


def clean_nba_data(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    df = load_raw_data(input_path)

    validate_required_columns(df)

    df = standardize_values(df)
    df = remove_exact_duplicates(df)

    validate_team_rows_per_game(df)

    df = handle_missing_shooting_percentages(df)
    df = add_location_columns(df)
    df = remove_neutral_site_games(df)

    # Recheck after removing ambiguous games.
    validate_team_rows_per_game(df)
    validate_game_results(df)

    games = convert_to_game_level(df)
    games = simplify_game_columns(games)
    games = create_target(games)

    games = games.sort_values(
        ["game_date", "game_id"],
    ).reset_index(drop=True)

    validate_cleaned_games(games)

    games = order_columns(games)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    games.to_csv(
        output_path,
        index=False,
    )

    LOGGER.info(
        "Saved %s cleaned games to %s",
        len(games),
        output_path,
    )

    LOGGER.info(
        "Games by season:\n%s",
        games.groupby("season")["game_id"].nunique().to_string(),
    )

    return games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean NBA team-game data and convert it to one row per game."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/nba_games_2022_2026.csv"),
        help="Path to the raw NBA CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Data/processed/nba_games_clean.csv"),
        help="Path for the cleaned game-level CSV.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    clean_nba_data(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()