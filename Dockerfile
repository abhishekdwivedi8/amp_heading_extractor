# Minimal Dockerfile for PDF heading extraction - Adobe Round 1A
FROM --platform=linux/amd64 python:3.10-slim

# Handle architecture compatibility
RUN if [ "$(uname -m)" != "x86_64" ]; then \
        echo "Warning: Non-AMD64 architecture detected"; \
        export FALLBACK_MODE=true; \
    fi

WORKDIR /app

# Install system dependencies with error handling
RUN apt-get update && apt-get install -y \
    libmupdf-dev || echo "Warning: libmupdf-dev not available" && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models to prevent evaluation-time downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Create models directory and cache
RUN mkdir -p /app/models
RUN python -c "import sentence_transformers; model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables for optimized performance
ENV PYTHONUNBUFFERED=1
ENV DOCKER_CONTAINER=true
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV OFFLINE_MODE=true
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Default command
CMD ["python", "process_pdfs.py"]