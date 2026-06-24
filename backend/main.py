import time
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Added the missing AutoModelForSequenceClassification import here
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, pipeline
from newspaper import Article as NewsArticle
from newspaper import Config as NewsConfig
import nltk

# Point NLTK strictly to the internal baked-in folder
nltk.data.path = ["/app/nltk_data"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

# Standardizing absolute paths mapped inside the docker context
MODEL_DIR = "/app/model_weights"
SENTIMENT_DIR = "/app/sentiment_model"

# Corrected path references to match variables perfectly
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_DIR)

# Explicitly instantiating the missing sentiment analysis task pipeline
sentiment_task = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=-1 # Forces CPU inference allocation
)

class Article(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: str

def run_pipeline_inference(raw_text: str):
    start_time = time.time()
    sentiment_result = sentiment_task(raw_text[:512])[0]

    with torch.no_grad():
        inputs = tokenizer("summarize: " + raw_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = model.generate(inputs["input_ids"], num_beams=4, max_new_tokens=120, min_new_tokens=30, early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.0)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)

    end_time = time.time()
    return {
        "summary": summary,
        "metadata": {
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "input_tokens": inputs["input_ids"].shape[1],
            "device": device,
            "sentiment": sentiment_result["label"],
            "score": round(sentiment_result["score"], 4)
        }
    }

@app.get("/")
async def root(): return {"status": "healthy"}

@app.post("/generate")
async def generate_summary(article: Article):
    if not article.text.strip(): raise HTTPException(status_code=400, detail="Empty text stream.")
    return run_pipeline_inference(article.text)

@app.post("/scrape")
async def scrape_and_summarize(payload: ScrapeRequest):
    target_url = payload.url.strip()
    if not target_url.startswith(("http://", "https://")): raise HTTPException(status_code=400, detail="Invalid URI.")
    try:
        config = NewsConfig()
        config.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        config.request_timeout = 10
        article = NewsArticle(target_url, config=config)
        article.download()
        article.parse()
        extracted_text = article.text.strip()
        if len(extracted_text) < 150: raise HTTPException(status_code=400, detail="Text density too low.")
        return run_pipeline_inference(extracted_text[:4500])
    except Exception as err: raise HTTPException(status_code=500, detail=str(err))