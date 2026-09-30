import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

LEAGUE = "PL"

# --- CSV (football-data.co.uk): historia fija, descargada a mano ---
# Código de la temporada como en la página: "2122" = 2021-22
CSV_SEASONS = ["2122", "2223", "2324", "2425", "2526"]

# --- API (football-data.org): temporada en curso ---
API_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
API_BASE_URL = "https://api.football-data.org/v4"
API_LEAGUE = LEAGUE
API_SEASONS = [2026]   # año de inicio: 2026 = temporada 2026-27

PAUSE_SECONDS = 7
MAX_RETRIES = 3