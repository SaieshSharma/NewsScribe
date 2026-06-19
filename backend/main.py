import time
import os
import torch
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
from newspaper import Article as NewsArticle

app = FastAPI()

# Enable CORS for React frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL CONFIGURATION & LOADING ---
LOCAL_MODEL_PATH = "./model_weights"
SENTIMENT_MODEL_PATH = "./sentiment_model"
device = "cpu"

if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
    print(f"--- 🏰 Loading trained T5 model from local storage: {LOCAL_MODEL_PATH} ---")
    model_source = LOCAL_MODEL_PATH
else:
    print("--- 🌐 Local weights empty. Defaulting to 't5-small' cloud configuration ---")
    model_source = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_source)
model = T5ForConditionalGeneration.from_pretrained(model_source).to(device)

if os.path.exists(SENTIMENT_MODEL_PATH) and os.listdir(SENTIMENT_MODEL_PATH):
    print(f"--- 🏰 Loading local Sentiment Model from {SENTIMENT_MODEL_PATH} ---")
    sentiment_source = SENTIMENT_MODEL_PATH
else:
    sentiment_source = "distilbert-base-uncased-finetuned-sst-2-english"

sentiment_task = pipeline("sentiment-analysis", model=sentiment_source, device=-1)

# --- SCHEMAS ---
class Article(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: str

# --- CORE UTILITY WORKFLOWS ---
def run_pipeline_inference(raw_text: str):
    """Encapsulates the core analytical execution layer for both workflows."""
    start_time = time.time()
    
    # Safe guard truncation for basic sentiment input layer
    sentiment_result = sentiment_task(raw_text[:512])[0]

    with torch.no_grad():
        inputs = tokenizer(
            "summarize: " + raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        output = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_new_tokens=80,
            min_new_tokens=15,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )

        summary = tokenizer.decode(output[0], skip_special_tokens=True)

    end_time = time.time()
    
    return {
        "summary": summary,
        "metadata": {
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "input_tokens": inputs["input_ids"].shape[1],
            "device": device,
            "model": model.config._name_or_path,
            "sentiment": sentiment_result["label"],
            "score": round(sentiment_result["score"], 4)
        }
    }

# --- CONTROLLER ROUTING ---
@app.get("/")
async def root():
    return {"status": "healthy", "engine": "NewsScribe Core"}

@app.post("/generate")
async def generate_summary(article: Article):
    if not article.text.strip():
        raise HTTPException(status_code=400, detail="Input text stream cannot be empty.")
    return run_pipeline_inference(article.text)

@app.post("/scrape")
async def scrape_and_summarize(payload: ScrapeRequest):
    target_url = payload.url.strip()
    if not target_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid global URI scheme detected.")

    try:
        # Newspaper4k handles downloading, user-agent masking, and HTML cleaning out of the box
        article = NewsArticle(target_url)
        article.download()
        article.parse()
        
        extracted_text = article.text.strip()

        if len(extracted_text) < 150:
            raise HTTPException(status_code=400, detail="Extracted text density too low. The domain might be behind a hard paywall.")

        # Truncate string gracefully to protect deep learning memory context bounds
        sanitized_input = extracted_text[:4500]
        
        # Pass variables cleanly down into your existing inference loop function
        return run_pipeline_inference(sanitized_input)
        
    except Exception as general_err:
        raise HTTPException(status_code=500, detail=f"Internal extraction processing failure: {str(general_err)}")
    target_url = payload.url.strip()
    if not target_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid global URI scheme detected.")

    try:
        async with httpx.AsyncClient() as client:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsScribe/2.0"}
            response = await client.get(target_url, headers=headers, timeout=12.0)
            
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Source server returned validation status: {response.status_code}")

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Deconstruct unneeded HTML trees
        for junk in soup(["script", "style", "nav", "footer", "header", "aside"]):
            junk.decompose()

        # Capture text paragraphs
        paragraphs = soup.find_all('p')
        extracted_text = " ".join([p.get_text() for p in paragraphs]).strip()

        if len(extracted_text) < 150:
            raise HTTPException(status_code=400, detail="Extracted body element content density too low to yield a robust summary.")

        # Cap token input length gracefully to preserve CPU memory
        return run_pipeline_inference(extracted_text[:4500])

    except httpx.RequestError as net_err:
        raise HTTPException(status_code=500, detail=f"Network gateway transport failure: {str(net_err)}")
    except Exception as general_err:
        raise HTTPException(status_code=500, detail=f"Internal extraction processing failure: {str(general_err)}")