import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_dataset(path: str | None = None):
    """
    Carga el dataset de noticias falsas.
    """
    if path is None:
        path = DATA_PATH

    df = pd.read_csv(path)

    X = df["text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
