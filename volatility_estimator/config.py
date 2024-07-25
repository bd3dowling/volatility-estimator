import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

# Load default env-vars from .env
load_dotenv()

# Access env-vars for central access
RAW_FILE_NAME_PATTERN = os.getenv("RAW_FILE_NAME_PATTERN", "")
START_TIME = datetime.time.fromisoformat(os.getenv("START_TIME", "00:00:00"))
END_TIME = datetime.time.fromisoformat(os.getenv("END_TIME", "23:59:59"))
NUM_TRADING_DAYS = int(os.getenv("NUM_TRADING_DAYS", 252))
DATA_PATH = Path(os.getenv("DATA_PATH", "data"))

# Derivative configs
LOAD_DATA_PATH = DATA_PATH / "load"
CLEAN_DATA_PATH = DATA_PATH / "clean"
CLEAN_PRICE_PATH = CLEAN_DATA_PATH / "prices"
HIST_VOL_PATH = CLEAN_DATA_PATH / "historical_volatility"
