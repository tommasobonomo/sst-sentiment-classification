import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from settings import ClassifierType
from src.helpers import get_inference_pipeline

app = FastAPI(
    title="Sentiment classifier API"
)


class SentimentRequest(BaseModel):
    phrase: str
    model: ClassifierType


class SentimentResponse(BaseModel):
    sentiment: str
    score: float


@app.get("/sentiment")
def get_sentiment_of_sentence(request: SentimentRequest) -> SentimentResponse:
    """API endpoint that takes a sentence and evaluates if it carries a positive or negative sentiment"""
    inference_pipeline = get_inference_pipeline(request.model)

    # Prepare input and predict
    X = np.array([request.phrase])
    y_pred = inference_pipeline.predict(X).squeeze()

    if y_pred[1] >= y_pred[0]:
        sentiment = "positive"
    else:
        sentiment = "negative"

    score = y_pred[1]

    return SentimentResponse(sentiment=sentiment, score=score)
