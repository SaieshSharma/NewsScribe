FROM python:3.11-slim

WORKDIR /app

# Zero Margin Error Patch: Install C++ system runtime libs required for SentencePiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies matrix
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and freeze NLTK assets directly into the image layers
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); nltk.download('punkt_tab', download_dir='/app/nltk_data')"

# Copy application script
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]