FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first — requires PyTorch's own index URL
# sentence-transformers>=5.1.0 requires torch>=2.6.0
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY qa-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models (cached in models_cache volume — subsequent restarts are instant)
# 1. BAAI/bge-m3: embedding model (1024 dim, multilingual)
# 2. cross-encoder/ms-marco-MiniLM-L-6-v2: reranker (22M params)
RUN python -c \
    "from sentence_transformers import SentenceTransformer, CrossEncoder; \
     SentenceTransformer('BAAI/bge-m3'); \
     CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY qa-agent/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
