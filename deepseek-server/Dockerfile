FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_VERBOSE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install dependencies in stages with verbose output for debugging
RUN pip install --no-cache-dir -v fastapi uvicorn pydantic requests && \
    pip install --no-cache-dir -v torch && \
    pip install --no-cache-dir -v transformers && \
    pip install --no-cache-dir -v accelerate safetensors sentencepiece protobuf einops

# Copy application code
COPY app.py .
COPY download_model.py .

# Create model cache directory
RUN mkdir -p /app/model_cache /app/logs

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
