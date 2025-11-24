import joblib
from .agent import VeriNewsIAAgent
from .config import MODEL_PATH

LABEL_MAP = {
    0: "NOTICIA FALSA",
    1: "NOTICIA VERDADERA"
}

def main():
    model = joblib.load(MODEL_PATH)
    agent = VeriNewsIAAgent(model=model)

    print("=== VeriNewsIA CLI ===")
    print("Escribe una noticia (o 'salir' para terminar):")

    while True:
        text = input("> ")
        if text.lower() in ("salir", "exit", "quit"):
            break

        pred = agent.predict([text])[0]
        try:
            proba = agent.predict_proba([text])[0]
            print(f"Predicción: {LABEL_MAP.get(pred, pred)}")
            print(f"Probabilidades: {proba}")
        except Exception:
            print(f"Predicción: {LABEL_MAP.get(pred, pred)}")

if __name__ == "__main__":
    main()
