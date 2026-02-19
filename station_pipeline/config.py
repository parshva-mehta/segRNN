from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CONTINUOUS_FEATURES = [
    "feel",
    "relh",
    "tmpf",
    "vsby",
    "sknt",
    "mslp",
    "p01i",
    "alti",
    "dwpf",
    "drct",
]

PLACEHOLDERS = ["M", "T", "", "NaN", "NULL", "None"]

DEFAULT_START_DAYS_BACK = 365
DEFAULT_N_STEPS_IN = 24
DEFAULT_N_STEPS_OUT = 1
DEFAULT_TEST_SPLIT = 0.2

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
IEM_GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
IEM_ASOS_REQUEST_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

