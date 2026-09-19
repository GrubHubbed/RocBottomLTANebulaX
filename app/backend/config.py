import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
EXCEL_DIR = STORAGE_DIR / "excel"
MODELS_DIR = STORAGE_DIR / "models"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"
STANDIN_MODELS_DIR = BASE_DIR / "standin_models"

# Ensure storage directories exist
for directory in [STORAGE_DIR, EXCEL_DIR, MODELS_DIR, PREDICTIONS_DIR, STANDIN_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# App Configuration
APP_TITLE = "LTA Smart Depot OCC"
APP_VERSION = "2.0.0 (NebulaX PS3)"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8080))
