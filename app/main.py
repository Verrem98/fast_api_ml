from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import mod_predict

app = FastAPI()


class TextIn(BaseModel):
    text: str
    model: str
    labels: dict


class PredictionOutput(BaseModel):
    prediction: str


@app.get("/")
def home():
    return {"test": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: TextIn):
    prediction = mod_predict(payload.model, payload.text)
    print(payload.labels)
    return {'prediction':payload.labels[str(prediction)]}
