import joblib
from pathlib import Path

from .data_loader import load_dataset
from .agent import VeriNewsIAAgent
from .config import MODEL_DIR, MODEL_PATH

def main():
    X_train, X_test, y_train, y_test = load_dataset()

    # Antes:
    # agent = VeriNewsIAAgent()

    # Mejor:
    agent = VeriNewsIAAgent(model_type="svm")  # "logreg", "svm" o "rf"
    agent.train(X_train, y_train)

    print("=== Evaluación en conjunto de prueba ===")
    agent.evaluate(X_test, y_test)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(agent.model, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()
