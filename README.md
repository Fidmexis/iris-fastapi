# Классификатор ирисов Фишера (FastAPI + Keras)

## Запуск
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py            # создаёт model/iris_model.keras
uvicorn app.main:app --reload
```
Откройте http://127.0.0.1:8000

Пример для проверки: 5.1, 3.5, 1.4, 0.2 → setosa.
