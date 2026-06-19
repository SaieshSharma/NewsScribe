import time
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from newspaper import Article as NewsArticle
from newspaper import Config as NewsConfig
import nltk

# --- NLTK RUNTIME INITIALIZATION LAYER ---
nltk_data_dir = "/app/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cpu"

FINETUNED_T5_HUB = "SaieshSharma/newsscribe-t5"
FINETUNED_SENTIMENT_HUB = "SaieshSharma/newsscribe-sentiment"

print("Initializing models from Hugging Face Registry Hub...")

# Load your custom fine-tuned T5 Summarizer engine
tokenizer = T5Tokenizer.from_pretrained(FINETUNED_T5_HUB)
model = T5ForConditionalGeneration.from_pretrained(FINETUNED_T5_HUB).to(device)
print("🏰 Fine-tuned T5 core matrix engine loaded successfully.")

# Load your custom sentiment analysis classification engine
sentiment_task = pipeline("sentiment-analysis", model=FINETUNED_SENTIMENT_HUB, device=-1)
print("🏰 Fine-tuned Sentiment analytical layer loaded successfully.")

# --- SCHEMAS ---
class Article(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: str

# --- CORE UTILITY WORKFLOWS ---
def run_pipeline_inference(raw_text: str):
    start_time = time.time()
    
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
            max_new_tokens=120,      
            min_new_tokens=30,
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
        # Founder Architecture Patch: Inject desktop browser headers to bypass news anti-bot walls
        config = NewsConfig()
        config.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        config.request_timeout = 10
        
        article = NewsArticle(target_url, config=config)
        article.download()
        article.parse()
        
        extracted_text = article.text.strip()

        if len(extracted_text) < 150:
            raise HTTPException(status_code=400, detail="Extracted text density too low. Content may be guarded by a strict script paywall.")

        sanitized_input = extracted_text[:4500]
        return run_pipeline_inference(sanitized_input)
        
    except HTTPException as http_err:
        raise http_err
    except Exception as general_err:
        raise HTTPException(status_code=500, detail=f"Data parser processing failure: {str(general_err)}")