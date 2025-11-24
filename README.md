# Proyecto_PTIA
#### Ricardo Ayala
#### Allan Contreras

### VeriNewsIA

Sistema inteligente para detección de noticias falsas usando
Procesamiento de Lenguaje Natural y aprendizaje supervisado.

## 1. Descripción general

**VeriNewsIA** es un prototipo desarrollado para el curso  
**Principios y Tecnologías de Inteligencia Artificial (PTIA)**.

El sistema implementa un **agente de IA** capaz de:

- Recibir el **texto de una noticia** como entrada.
- Transformar el texto a una representación numérica usando **TF-IDF**.
- Aplicar un **modelo de aprendizaje supervisado** (Logistic Regression, SVM o Random Forest).
- Clasificar la noticia como:

> `0` → NOTICIA FALSA  
> `1` → NOTICIA VERDADERA  

El enfoque combina conceptos vistos en clase: aprendizaje supervisado, PLN, métricas de evaluación, manejo de overfitting y diseño de agentes inteligentes.

---

## 2. Objetivo del proyecto

Diseñar e implementar un prototipo de sistema de IA capaz de **clasificar noticias como verdaderas o falsas**, alcanzando un rendimiento razonable en un dataset pequeño y demostrando:

- Uso correcto del **ciclo de vida de un modelo de ML** (datos → entrenamiento → evaluación).
- Aplicación de **técnicas de PLN** básicas.
- Comparación de **distintos modelos supervisados**.
- Diseño de un **agente de IA** claro y documentado.

---

## 3. Arquitectura del agente VeriNewsIA

El agente se implementa en la clase `VeriNewsIAAgent`:

1. **Percepción**  
   - Entrada: texto crudo de la noticia (`str`).
   - Módulo: `TfidfVectorizer` convierte el texto en un vector numérico.

2. **Razonamiento**  
   - Módulo de clasificación configurable:
     - `LogisticRegression`
     - `LinearSVC` (SVM lineal)
     - `RandomForestClassifier`  
   - Se utiliza `class_weight="balanced"` para lidiar con posibles desbalances entre clases.

3. **Acción**  
   - Devuelve la **etiqueta predicha** (`0` ó `1`).
   - Durante la evaluación, calcula **accuracy** y muestra un `classification_report` con:
     - precision  
     - recall  
     - f1-score  
     - support  

Todo esto se encapsula en un **Pipeline de scikit-learn**:  

`texto → TF-IDF → clasificador → predicción`.

---

## 4. Estructura del proyecto

```text
Proyecto_PTIA/
├─ data/
│  └─ fake_news_dataset.csv        # Dataset sintético de noticias
├─ models/
│  └─ verinewsia_model.joblib      # Modelo entrenado (se genera después)
├─ verinewsia/
│  ├─ __init__.py
│  ├─ config.py                    # Rutas y parámetros globales
│  ├─ data_loader.py               # Carga y separación train/test
│  ├─ agent.py                     # Agente VeriNewsIA (IA principal)
│  ├─ train.py                     # Script de entrenamiento
│  ├─ evaluate.py                  # Script de evaluación del modelo guardado
│  └─ cli.py                       # Interfaz de línea de comandos
├─ requirements.txt
└─ README.md


## Estructura básica

- `data/fake_news_dataset.csv`: dataset con columnas `text`, `label`.
- `verinewsia/train.py`: entrena el modelo.
- `verinewsia/cli.py`: prueba el modelo escribiendo noticias por consola.

## Uso

```bash
pip install -r requirements.txt
python -m verinewsia.train
python -m verinewsia.cli


---

## 3. Qué te falta para correrlo YA

1. Crear la carpeta del proyecto con esta estructura.  
2. Poner tu dataset en `data/fake_news_dataset.csv` con columnas:
   - `text` → texto de la noticia  
   - `label` → 0/1 o fake/real (ajusta el `LABEL_MAP` si es texto).  
3. Instalar dependencias:

```bash
pip install -r requirements.txt

python -m verinewsia.train

python -m verinewsia.cli
