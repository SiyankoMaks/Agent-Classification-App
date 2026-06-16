from pathlib import Path

from dotenv import load_dotenv
import os


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset_deviant_50_final.xlsx"
MODEL_PATH = BASE_DIR / "models" / "deviant_risk_model.joblib"

POLZA_API_KEY = os.getenv("POLZA_API_KEY", "")
POLZA_MODEL = os.getenv("POLZA_MODEL", "deepseek/deepseek-v4-flash")
POLZA_BASE_URL = os.getenv("POLZA_BASE_URL", "https://polza.ai/api/v1")
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "500"))
