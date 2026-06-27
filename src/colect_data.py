from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

seasons = ["2023-2024", 
           "2025,2025",
           "2025-2026"]

response = leaguegamelog.LeagueGameLog(season=seasons[0], season_type_all_star="Regular Season", player_or_team_abbreviation="T")
season_df = response.get_data_frames()[0]

