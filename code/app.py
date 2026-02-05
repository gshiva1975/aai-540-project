from fastapi import FastAPI, HTTPException
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI()

# ---- Load model at startup (fail fast) ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

classifier = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    framework="pt"
)

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
def predict(payload: dict):
    text = payload.get("text")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")

    return classifier([text])

