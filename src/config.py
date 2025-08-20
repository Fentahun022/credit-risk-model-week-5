from pathlib import Path

# --- DIRECTORIES ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "data.csv"
# --- MLFLOW CONFIGURATION ---

MLFLOW_TRACKING_URI_TRAIN = "http://127.0.0.1:5000"
MLFLOW_TRACKING_URI_API = f"sqlite:///{DATABASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_B5W5_Production"
REGISTERED_MODEL_NAME = "CreditRiskModel"
# --- MODEL PARAMETERS ---

RANDOM_STATE = 42
TEST_SIZE = 0.2
KMEANS_N_CLUSTERS = 3
MODEL_PARAMS = {"random_state": RANDOM_STATE}

MIN_SCORE = 300
MAX_SCORE = 850

