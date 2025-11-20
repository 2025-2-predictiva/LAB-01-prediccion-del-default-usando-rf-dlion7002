# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
# def clean_data(df):
#     df = df.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}


# # 1. Importación de librerías y configuración general

# %%
import os
import json
import gzip
import pickle
from typing import Tuple, Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Rutas y constantes
DATA_DIR = "files/input"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"
TARGET_COL = "default"
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


# %% [markdown]
# # 2. Funciones de carga y limpieza de datos

# %%
def read_zip_csv(filename: str) -> pd.DataFrame:
    """Lee un CSV comprimido en ZIP desde la carpeta de entrada."""
    full_path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(full_path, compression="zip", index_col=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las transformaciones de limpieza requeridas en el enunciado."""
    df = df.rename(columns={"default payment next month": TARGET_COL})
    df = df.drop(columns=["ID"])

    # Filtrar información no disponible
    mask_valid = (df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)
    df = df.loc[mask_valid].copy()

    # Agrupar EDUCATION > 4 en categoría 'others' (4)
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"] < 4, 4)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X (características) e y (objetivo)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


# %% [markdown]
# # 3. Construcción del pipeline y búsqueda de hiperparámetros

# %%
def build_pipeline() -> Pipeline:
    """Crea el pipeline de preprocesamiento + modelo."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="passthrough",
    )

    rf = RandomForestClassifier(random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", rf),
        ]
    )
    return pipe


def build_search(pipe: Pipeline) -> GridSearchCV:
    """Envuelve el pipeline en un GridSearchCV con validación cruzada."""
    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 5, 10, 20],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )
    return search


# %% [markdown]
# # 4. Funciones para guardar el modelo y calcular métricas

# %%
def dump_model(model, path: str) -> None:
    """Guarda el modelo entrenado en un archivo comprimido con gzip."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def build_metrics_record(dataset: str, y_true, y_pred) -> Dict:
    """Diccionario con métricas principales para un dataset."""
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def build_confusion_record(dataset: str, y_true, y_pred) -> Dict:
    """Diccionario con la matriz de confusión en el formato solicitado."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(tn),
            "predicted_1": int(fp),
        },
        "true_1": {
            "predicted_0": int(fn),
            "predicted_1": int(tp),
        },
    }


def save_metrics(records: List[Dict], path: str) -> None:
    """Guarda una lista de diccionarios (métricas y matrices) en JSON line-by-line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# %% [markdown]
# # 5. Función principal: entrenamiento, evaluación y guardado

# %%
def main() -> None:
    # 1. Cargar datasets
    train_raw = read_zip_csv("train_data.csv.zip")
    test_raw = read_zip_csv("test_data.csv.zip")

    # 2. Limpiar datos
    train_clean = clean_data(train_raw)
    test_clean = clean_data(test_raw)

    # 3. Separar X e y
    X_train, y_train = split_features_target(train_clean)
    X_test, y_test = split_features_target(test_clean)

    # 4. Construir pipeline y búsqueda de hiperparámetros
    pipeline = build_pipeline()
    search = build_search(pipeline)

    # 5. Entrenar (incluye validación cruzada)
    search.fit(X_train, y_train)

    # 6. Guardar modelo entrenado
    dump_model(search, MODEL_PATH)

    # 7. Predicciones para train y test
    y_train_pred = search.predict(X_train)
    y_test_pred = search.predict(X_test)

    # 8. Métricas y matrices de confusión
    train_metrics = build_metrics_record("train", y_train, y_train_pred)
    test_metrics = build_metrics_record("test", y_test, y_test_pred)

    train_cm = build_confusion_record("train", y_train, y_train_pred)
    test_cm = build_confusion_record("test", y_test, y_test_pred)

    # 9. Guardar todo en metrics.json
    records = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics(records, METRICS_PATH)


# %% [markdown]
# # 6. Punto de entrada

# %%
if __name__ == "__main__":
    main()

# %%
