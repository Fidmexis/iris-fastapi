from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.iris_model import predict_iris

app = FastAPI(title="Классификатор ирисов Фишера")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    result = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    values = {
        "sepal_length": sepal_length, "sepal_width": sepal_width,
        "petal_length": petal_length, "petal_width": petal_width,
    }
    return templates.TemplateResponse(
        request, "index.html", {"result": result, "values": values}
    )
