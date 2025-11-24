from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "fake_news_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "verinewsia_model.joblib"

# Parámetros
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MAX_ITER = 1000
