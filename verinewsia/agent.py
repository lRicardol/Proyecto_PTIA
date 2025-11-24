from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from .config import MAX_FEATURES, NGRAM_RANGE, MAX_ITER


def build_classifier(model_type: str = "logreg"):
    """
    Crea un clasificador supervisado según el tipo indicado.

    model_type:
        - "logreg"  : Regresión logística (buena baseline para texto).
        - "svm"     : SVM lineal (suele funcionar muy bien en texto).
        - "rf"      : Random Forest (ensamble de árboles).
    """
    model_type = model_type.lower()

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=MAX_ITER,
            class_weight="balanced",
            n_jobs=None,
        )
    elif model_type == "svm":
        clf = LinearSVC(
            class_weight="balanced"
        )
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=NGRAM_RANGE
            )),
            ("clf", clf),
        ]
    )
    return pipe


class VeriNewsIAAgent:
    """
    Agente inteligente para detección de noticias falsas.

    Percibe: texto de la noticia.
    Razona: transforma texto a espacio vectorial (TF-IDF) y aplica un clasificador.
    Actúa: devuelve etiqueta (0=falsa, 1=verdadera) y métricas de rendimiento.
    """

    def __init__(self, model=None, model_type: str = "logreg"):
        if model is None:
            self.model_type = model_type
            self.model = build_classifier(model_type=model_type)
        else:
            self.model_type = model_type
            self.model = model

    def train(self, X_train, y_train):
        """Entrena el modelo"""
        self.model.fit(X_train, y_train)

    def predict(self, texts):
        """Predice etiquetas para uno o varios textos."""
        return self.model.predict(texts)

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en un conjunto de prueba.
        Retorna la exactitud y muestra un reporte de clasificación.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Modelo: {self.model_type}")
        print(f"Exactitud: {acc:.2%}")
        print(classification_report(y_test, y_pred))
        return acc
