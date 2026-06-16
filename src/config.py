import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SPORTS = [
    "americanfootball_nfl",
    "basketball_nba",
    "soccer_epl",
    "baseball_mlb",
    "icehockey_nhl",
]

NUM_CUSTOMERS = 2000
SIMULATION_DAYS = 180
