# Setup

These steps run the React frontend and FastAPI backend locally. The backend needs local model artifacts before inference can work.

## Prerequisites

- Python 3.11 recommended
- Node.js 18+
- T5 model files in `backend/model_weights/`
- Sentiment model files in `backend/sentiment_model/`

The sentiment model can be downloaded with:

```bash
cd backend
python download_sentiment.py
```

## Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

If the frontend cannot reach the backend, create `frontend/.env`:

```text
VITE_API_URL=http://127.0.0.1:8000
```

## Docker Backend

```bash
cd backend
docker build -t newsscribe-backend .
docker run -p 8000:8000 ^
  -v %cd%\model_weights:/app/model_weights ^
  -v %cd%\sentiment_model:/app/sentiment_model ^
  newsscribe-backend
```

On macOS/Linux, replace `%cd%` with `$(pwd)` and `^` with `\`.

## Common Issues

| Issue | Check |
| --- | --- |
| Backend fails on startup | Confirm `backend/model_weights` and `backend/sentiment_model` contain compatible model files. |
| Frontend request fails | Confirm `VITE_API_URL` points to the backend port. |
| URL scraping fails | Some sites block scraping, hide content behind JavaScript, or require a subscription. |
| Slow generation | The app runs on CPU by default; current settings favor stability over maximum summary quality. |
