from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


# ----------------------------
# File locations
# ----------------------------
OLD_DATA_PATH = Path("data/raw/nba_games_2022_2025.csv")
HOLDOUT_PATH = Path("data/raw/nba_games_2025_26.csv")
COMBINED_PATH = Path("data/processed/nba_games_2022_2026.csv")


# ----------------------------
# Make sure folders exist
# ----------------------------
HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Load the previous three seasons
# ----------------------------
old_games = pd.read_csv(
    OLD_DATA_PATH,
    dtype={"GAME_ID": "string"}
)

old_games["GAME_DATE"] = pd.to_datetime(
    old_games["GAME_DATE"],
    errors="coerce"
)


# ----------------------------
# Download the 2025-26 season
# ----------------------------
print("Downloading the 2025-26 regular season...")

response = leaguegamelog.LeagueGameLog(
    season="2025-26",
    season_type_all_star="Regular Season",
    player_or_team_abbreviation="T",
    timeout=60
)

games_2025_26 = response.get_data_frames()[0]

games_2025_26["SEASON"] = "2025-26"

games_2025_26["GAME_DATE"] = pd.to_datetime(
    games_2025_26["GAME_DATE"],
    errors="coerce"
)

games_2025_26["GAME_ID"] = games_2025_26["GAME_ID"].astype("string")


# ----------------------------
# Save 2025-26 separately
# ----------------------------
games_2025_26.to_csv(
    HOLDOUT_PATH,
    index=False
)


# ----------------------------
# Combine all four seasons
# ----------------------------
all_games = pd.concat(
    [old_games, games_2025_26],
    ignore_index=True
)

# Each game appears once for each team, so the unique row
# identifier should be GAME_ID + TEAM_ID.
all_games = all_games.drop_duplicates(
    subset=["GAME_ID", "TEAM_ID"],
    keep="last"
)

all_games = all_games.sort_values(
    ["GAME_DATE", "GAME_ID", "TEAM_ID"]
).reset_index(drop=True)


# ----------------------------
# Save the combined dataset
# ----------------------------
all_games.to_csv(
    COMBINED_PATH,
    index=False
)


# ----------------------------
# Inspect the results
# ----------------------------
print("\nFinished!")
print(f"2025-26 holdout saved to: {HOLDOUT_PATH}")
print(f"Combined data saved to: {COMBINED_PATH}")

print("\nRows by season:")
print(all_games.groupby("SEASON").size())

print("\nUnique games by season:")
print(all_games.groupby("SEASON")["GAME_ID"].nunique())

print("\nDuplicate team-game rows:")
print(
    all_games.duplicated(
        subset=["GAME_ID", "TEAM_ID"]
    ).sum()
)