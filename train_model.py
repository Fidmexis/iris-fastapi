"""Обучает простую нейросеть на ирисах Фишера и сохраняет её в model/iris_model.keras.
Если модель уже обучена в вашем ноутбуке, вместо этого скрипта в ноутбуке выполните:
    model.save("iris_model.keras")
и положите файл в папку model/.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность на тесте: {acc:.3f}")

model.save("model/iris_model.keras")
print("Модель сохранена в model/iris_model.keras")
