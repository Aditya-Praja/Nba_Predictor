from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


# Three complete NBA seasons ending from 2023 through 2025
seasons = ["2022-23", "2023-24", "2024-25"]

season_dataframes = []

for season in seasons:
    print(f"Downloading {season}...")

    response = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T"
    )

    season_df = response.get_data_frames()[0]

    # Add a clear season label
    season_df["SEASON"] = season

    season_dataframes.append(season_df)

    # Pause between requests
    time.sleep(1)


# Combine all three seasons
games = pd.concat(
    season_dataframes,
    ignore_index=True
)

# Convert the date column into datetime format
games["GAME_DATE"] = pd.to_datetime(
    games["GAME_DATE"],
    errors="coerce"
)

# Sort games chronologically
games = games.sort_values(
    by="GAME_DATE"
).reset_index(drop=True)

# Save all seasons in one CSV
games.to_csv(
    "Data/raw/nba_games_2022_2025.csv",
    index=False
)

print("\nDownload complete")
print("Shape:", games.shape)
print("Seasons:", games["SEASON"].unique())
print(games.head())