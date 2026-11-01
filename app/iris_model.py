"""Модуль работы с моделью: загрузка и предсказание."""
from pathlib import Path

import numpy as np
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.keras"

# Порядок классов как в sklearn.datasets.load_iris
CLASS_NAMES = ["Ирис щетинистый (setosa)",
               "Ирис разноцветный (versicolor)",
               "Ирис виргинский (virginica)"]

_model = keras.models.load_model(MODEL_PATH)


def predict_iris(sepal_length: float, sepal_width: float,
                 petal_length: float, petal_width: float) -> str:
    x = np.array([[sepal_length, sepal_width, petal_length, petal_width]],
                 dtype="float32")
    probs = _model.predict(x, verbose=0)[0]
    return CLASS_NAMES[int(np.argmax(probs))]
