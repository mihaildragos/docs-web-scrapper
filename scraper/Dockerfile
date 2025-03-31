FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    chromium \
    chromium-driver \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/root

# Create and set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser
RUN playwright install chromium
RUN playwright install-deps

# Pre-download PaddleOCR models
RUN mkdir -p /root/.paddleocr
COPY download_paddle_models.py .
RUN python download_paddle_models.py

# Copy application code
COPY *.py .
COPY lib/ ./lib/

# Create directories for output
RUN mkdir -p /app/docs /app/docs/debug

# Command to run the application (will be overridden by docker-compose)
CMD ["python", "workflow_scraper.py", "--help"]
