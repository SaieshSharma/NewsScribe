
import time
import os
import torch

# --- HARD HARDWARE THROTTLING FOR CPU CONTEXTS ---
# Prevents PyTorch from creating core thrashing on tight EC2 virtual instances
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from newspaper import Article as NewsArticle
from newspaper import Config as NewsConfig
from bs4 import BeautifulSoup
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

MODEL_DIR = "/app/model_weights"
SENTIMENT_DIR = "/app/sentiment_model"

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.config.use_cache = True

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_DIR)

class Article(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: str

def run_pipeline_inference(raw_text: str):
    start_time = time.time()
    
    # 1. LIGHTWEIGHT MANUALLY STEPPED SENTIMENT INFERENCE
    truncated_input_text = raw_text[:384] # Tightened for faster sentiment encoding processing
    sentiment_inputs = sentiment_tokenizer(
        truncated_input_text,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )
    sentiment_inputs.pop("token_type_ids", None)

    with torch.no_grad():
        sentiment_outputs = sentiment_model(**sentiment_inputs)
        probabilities = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)[0]
        prediction_index = torch.argmax(probabilities).item()
        
        sentiment_label = sentiment_model.config.id2label[prediction_index]
        sentiment_score = probabilities[prediction_index].item()

        # 2. HIGH-SPEED LEAN DECODING FOR T5 SUMMARIZER
        # Tightened max_length window from 512 down to a fast 320 block matrix frame
        inputs = tokenizer(
            "summarize: " + raw_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=320 
        ).to(device)
        
        output = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            num_beams=1,             
            use_cache=True,          
            max_new_tokens=70,       # Compact and punchy headlines/summaries process significantly quicker
            min_new_tokens=20, 
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
            "sentiment": sentiment_label,
            "score": round(sentiment_score, 4)
        }
    }

@app.get("/")
async def root(): 
    return {"status": "healthy"}

@app.post("/generate")
async def generate_summary(article: Article):
    if not article.text.strip(): 
        raise HTTPException(status_code=400, detail="Empty text stream.")
    return run_pipeline_inference(article.text)

@app.post("/scrape")
async def scrape_and_summarize(payload: ScrapeRequest):
    target_url = payload.url.strip()
    if not target_url.startswith(("http://", "https://")): 
        raise HTTPException(status_code=400, detail="Invalid URI.")
    try:
        config = NewsConfig()
        config.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        config.request_timeout = 15
        
        article = NewsArticle(target_url, config=config)
        article.download()
        article.parse()
        
        extracted_text = article.text.strip()
        
        bad_phrases = ["covered South Asia", "written by", "subscriber only", "follow us on"]
        is_junk = any(phrase in extracted_text for phrase in bad_phrases)
        
        if is_junk or len(extracted_text) < 400:
            soup = BeautifulSoup(article.html, "html.parser")
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.extract()
                
            paragraphs = soup.find_all("p")
            valid_blocks = []
            
            for p in paragraphs:
                p_text = p.get_text().strip()
                if len(p_text) > 50 and not any(phrase in p_text for phrase in bad_phrases):
                    valid_blocks.append(p_text)
                    
            if valid_blocks:
                extracted_text = " ".join(valid_blocks)

        if len(extracted_text) < 150: 
            raise HTTPException(status_code=400, detail="Unable to safely parse main news body from this layout.")
            
        return run_pipeline_inference(extracted_text[:4000])
        
    except Exception as err: 
        raise HTTPException(status_code=500, detail=str(err))