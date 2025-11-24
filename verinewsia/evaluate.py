import joblib
from .data_loader import load_dataset
from .agent import VeriNewsIAAgent
from .config import MODEL_PATH

def main():
    X_train, X_test, y_train, y_test = load_dataset()

    model = joblib.load(MODEL_PATH)
    agent = VeriNewsIAAgent(model=model)

    print("=== Evaluación del modelo cargado ===")
    agent.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
