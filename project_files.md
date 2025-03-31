# Project Files

## LICENSE

```LICENSE
MIT License

Copyright (c) 2023 Mihail Musea

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. ```

## README.md

```md
# AI-Enhanced Workflow Documentation Scraper

An advanced documentation scraping tool that combines browser automation, OCR technology, and AI language models to intelligently extract and organize documentation content from websites.

## New Workflow-Based Architecture

The scraper now uses a modular workflow-based architecture that divides the scraping process into distinct steps:

1. **Navigation Structure Extraction**: Scans the page and extracts the navigation structure or sitemap
2. **Visual Analysis**: Analyzes the non-navigational structure of the page visually with PaddleOCR
3. **Selector Finding**: Uses DeepSeek LLM to determine optimal HTML selectors for content extraction
4. **Content Extraction**: Uses the selectors with BeautifulSoup to extract structured content
5. **Content Processing**: Feeds the extracted content to DeepSeek for coherent document organization
6. **Traversal Planning**: DeepSeek determines the next logical pages to scrape based on content relationships

## Features

- **Intelligent Content Extraction**: Uses DeepSeek LLM to identify and extract relevant content
- **Visual Recognition**: Leverages PaddleOCR to recognize text from visual elements
- **SPA Support**: Handles JavaScript-heavy and single-page applications
- **CloudFlare Bypass**: Built-in solution for websites protected by CloudFlare
- **Hierarchical Output**: Preserves the original documentation structure
- **Markdown Conversion**: Converts HTML content to clean, readable markdown
- **Docker-based**: Runs in containerized environments for portability and isolation
- **Modular Architecture**: Each step is implemented in a separate module for easy customization

## Project Structure

```
docs-web-scraper/
├── LICENSE                   # MIT License file
├── README.md                 # This documentation
├── docker-compose.yml        # Container orchestration
├── scrape.sh                 # Main execution script
│
├── deepseek-server/          # AI language model service
│   ├── Dockerfile            # Container definition for DeepSeek
│   ├── app.py                # FastAPI server for the LLM
│   ├── download_model.py     # Script to download the model
│   └── requirements.txt      # Python dependencies
│
└── scraper/                  # Documentation scraper service
    ├── Dockerfile            # Container definition for scraper
    ├── workflow_scraper.py   # Main workflow-based scraper implementation
    ├── download_paddle_models.py # Pre-downloads OCR models
    ├── requirements.txt      # Python dependencies
    ├── config_example.json   # Example configuration file
    │
    └── lib/                  # Modular workflow components
        ├── __init__.py       # Package initialization
        ├── navigation_extractor.py # Navigation structure extraction
        ├── visual_analyzer.py      # Visual page analysis with OCR
        ├── selector_finder.py      # AI-powered selector discovery
        ├── content_extractor.py    # Content extraction with BeautifulSoup
        ├── content_processor.py    # AI-powered content processing
        └── traversal_planner.py    # Intelligent traversal planning
```

## Prerequisites

- Docker and Docker Compose
- Git
- 8GB+ RAM allocated to Docker
- Stable internet connection (for downloading models)
- (Optional) NVIDIA GPU with CUDA support for faster AI inference

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/docs-web-scraper.git
   cd docs-web-scraper
   ```

2. **Create required directories**:

   ```bash
   mkdir -p output
   mkdir -p deepseek-server/logs
   mkdir -p session_data
   ```

3. **Build the Docker containers**:

   ```bash
   docker-compose build
   ```

   This process may take some time, as it:

   - Builds the DeepSeek LLM server container
   - Builds the scraper container
   - Pre-downloads PaddleOCR models

4. **Make the script executable**:
   ```bash
   chmod +x scrape.sh
   ```

## Usage Instructions

### Basic Usage

```bash
./scrape.sh https://docs.example.com
```

This will scrape the specified documentation site with default parameters and save the results to the `./output` directory.

### Advanced Usage

```bash
./scrape.sh https://shopify.dev/docs/apps/build --depth 3 --delay 2 --debug --max-pages 50
```

### Available Options

- `--output`, `-o`: Output directory (default: "docs")
- `--depth`, `-d`: Maximum crawl depth (default: 3)
- `--delay`, `-w`: Delay between requests in seconds (default: 1.0)
- `--llm`: DeepSeek LLM endpoint (default: "http://deepseek:8000")
- `--visible`: Run browser in visible mode (not headless)
- `--debug`: Enable debug mode (save intermediate files)
- `--max-pages`: Maximum number of pages to scrape (default: 100)
- `--solve-captcha`: Open browser on your Mac to solve CloudFlare CAPTCHA first

## CloudFlare Bypass Instructions

Many documentation sites use CloudFlare protection that requires solving a CAPTCHA. The scraper includes functionality to handle this case:

1. **Run with the solve-captcha option**:

   ```bash
   ./scrape.sh https://docs.example.com --solve-captcha
   ```

2. **Solve the CAPTCHA in the browser window** that opens on your machine

3. **Wait for confirmation** that the session has been saved

4. **Continue with scraping** using the authenticated session:
   ```bash
   ./scrape.sh https://docs.example.com --depth 3 --debug
   ```

The authenticated session will be stored in `./session_data/cloudflare_session.json` and automatically mounted into the container for subsequent scraping runs.

## Workflow Architecture Details

### 1. Navigation Extraction

The `NavigationExtractor` identifies the site's navigation structure by:

- Analyzing common navigation elements (menus, sidebars, etc.)
- Detecting and processing sitemaps
- Extracting links from JavaScript-powered menus

### 2. Visual Analysis

The `VisualAnalyzer` performs visual analysis using PaddleOCR to:

- Extract text from screenshots
- Identify visual sections (header, main content, sidebar, footer)
- Detect tables and structured content visually

### 3. Selector Finding

The `SelectorFinder` uses DeepSeek LLM to:

- Analyze HTML structure and visual analysis data
- Determine optimal CSS selectors for different content types
- Validate selectors against the actual HTML

### 4. Content Extraction

The `ContentExtractor` uses BeautifulSoup with the optimal selectors to:

- Extract main content, navigation, code blocks, tables, figures, and notes
- Convert content to structured format
- Extract metadata from the page

### 5. Content Processing

The `ContentProcessor` uses DeepSeek LLM to:

- Organize extracted content into a coherent document
- Improve formatting and structure
- Create a comprehensive document structure

### 6. Traversal Planning

The `TraversalPlanner` uses DeepSeek LLM to:

- Analyze content relationships
- Determine the logical next pages to scrape
- Prioritize pages based on relevance and structure

## Advanced Configuration

For advanced customization, you can edit the configuration example in `scraper/config_example.json`.

### Using a Different LLM Model

To use a different DeepSeek model, edit the `docker-compose.yml` file:

```yaml
services:
  deepseek:
    environment:
      - MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct # Change this line
```

Available models:

- `deepseek-ai/deepseek-llm-7b-base` (default)
- `deepseek-ai/deepseek-llm-7b-chat`
- `deepseek-ai/deepseek-coder-6.7b-instruct` (good for code documentation)
- `deepseek-ai/deepseek-llm-1.3b-base` (for low-memory systems)

## Output Format

The scraper generates:

- Markdown files for each documentation page
- A hierarchical directory structure mirroring the website
- Debug files for each step in the process when debug mode is enabled

Each markdown file includes:

- The original title
- Source URL
- Processed content with preserved formatting

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:

   - Increase Docker's available memory (8GB+ recommended)
   - Use a smaller model (`deepseek-ai/deepseek-llm-1.3b-base`)
   - Reduce `MAX_LENGTH` and `MAX_NEW_TOKENS` in `docker-compose.yml`

2. **CloudFlare Detection**:

   - Use the `--solve-captcha` option to manually solve the CAPTCHA
   - Increase delay between requests with `--delay 2` or higher

3. **Scraper Not Finding Content**:

   - Enable debug mode with `--debug` to see what's being captured
   - Use `--visible` to watch the browser's activity
   - Check the debug files to see the selectors being used

4. **PaddleOCR Model Download Fails**:

   - The scraper will retry downloads automatically
   - If it persistently fails, check network connectivity
   - Consider downloading the models manually according to PaddleOCR documentation

5. **DeepSeek Model Not Loading**:
   - Check Docker logs with `docker-compose logs deepseek`
   - Verify you have enough disk space for model files
   - Try a smaller model if memory is constrained

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created by Mihail Dragos - Feel free to contribute by submitting pull requests or opening issues on GitHub.
```

## deepseek-server/Dockerfile

```deepseek-server/Dockerfile
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
```

## deepseek-server/app.py

```py
#!/usr/bin/env python3
"""
DeepSeek API Server - Simplified for Mac compatibility

A FastAPI server that provides a REST API for interacting with the DeepSeek LLM.
"""

import os
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/deepseek-llm-1.3b-base")
DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU for Mac
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek API",
    description="REST API for interacting with the DeepSeek language model",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model state
model = None
tokenizer = None
generator = None
is_model_loaded = False
model_loading = False


class ModelLoadError(Exception):
    """Exception raised when model loading fails"""

    pass


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE
    do_sample: Optional[bool] = True
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0


class GenerateResponse(BaseModel):
    text: str
    finish_reason: str
    generated_tokens: int
    elapsed_time: float


async def load_model_async():
    """Load the model in background - simplified for macOS compatibility"""
    global model, tokenizer, generator, is_model_loaded, model_loading

    if is_model_loaded or model_loading:
        return

    model_loading = True
    try:
        logger.info(f"Loading model {MODEL_ID}...")
        start_time = time.time()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Simple model loading for Mac compatibility
        logger.info(f"Loading model on {DEVICE}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map={"": DEVICE},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Create generator pipeline
        logger.info("Creating text generation pipeline")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
        )

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f} seconds")
        is_model_loaded = True

    except Exception as e:
        model_loading = False
        logger.error(f"Failed to load model: {str(e)}")
        raise ModelLoadError(f"Failed to load model: {str(e)}")

    model_loading = False


@app.on_event("startup")
async def startup_event():
    """Start model loading when the API server starts"""
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model_async)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "model": MODEL_ID, "loaded": is_model_loaded}


@app.get("/status")
async def status():
    """Get model status"""
    return {
        "model_id": MODEL_ID,
        "is_loaded": is_model_loaded,
        "is_loading": model_loading,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate text based on a prompt"""
    global model, tokenizer, generator, is_model_loaded

    if not is_model_loaded:
        if not model_loading:
            # Start loading if not already loading
            background_tasks.add_task(load_model_async)
        raise HTTPException(
            status_code=503, detail="Model is still loading. Please try again later."
        )

    try:
        start_time = time.time()

        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
        }

        # Generate text
        outputs = generator(request.prompt, **gen_kwargs)

        # Extract generated text
        generated_text = outputs[0]["generated_text"]

        # Remove the prompt from the generated text
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt) :]

        elapsed_time = time.time() - start_time

        return GenerateResponse(
            text=generated_text,
            finish_reason="stop",
            generated_tokens=len(tokenizer.encode(generated_text)),
            elapsed_time=elapsed_time,
        )

    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
```

## deepseek-server/download_model.py

```py
#!/usr/bin/env python3
"""
Model Download Script - Simplified for Mac compatibility

This script pre-downloads the DeepSeek model to avoid downloading it at runtime.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get the model ID from environment or use default
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/deepseek-llm-1.3b-base")
print(f"Downloading model: {MODEL_ID}")

# Download the tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained("./model_cache")

# Download the model with appropriate configuration for Mac
print("Downloading model for CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": "cpu"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Save the model
model.save_pretrained("./model_cache")
print("Model downloaded successfully!")
```

## deepseek-server/requirements.txt

```txt
# Base requirements
fastapi>=0.95.0
uvicorn>=0.22.0
pydantic>=2.0.0
requests>=2.28.0

# Model dependencies (simplified for Mac)
transformers>=4.31.0
torch>=2.0.0
accelerate>=0.20.0
safetensors>=0.3.1
sentencepiece>=0.1.97
protobuf>=4.23.0
einops>=0.6.0

# Optional optimizations - comment out if causing issues
bitsandbytes>=0.39.0
peft>=0.4.0```

## docker-compose.yml

```yml
services:
  deepseek:
    build:
      context: ./deepseek-server
      dockerfile: Dockerfile
    image: deepseek-api:latest
    container_name: deepseek-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_ID=deepseek-ai/deepseek-llm-1.3b-base
      - DEVICE=cpu
      - MAX_LENGTH=1024
      - MAX_NEW_TOKENS=512
      - TEMPERATURE=0.7
      - DEBUG=false
    volumes:
      - deepseek_models:/app/model_cache
      - ./deepseek-server/logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 180s

  scraper:
    build:
      context: ./scraper
      dockerfile: Dockerfile
    image: doc-scraper:latest
    container_name: doc-scraper
    depends_on:
      - deepseek
    volumes:
      - ./output:/app/docs
      - paddle_models:/root/.paddleocr
    environment:
      - LLM_ENDPOINT=http://deepseek:8000
    command: ["python", "workflow_scraper.py", "--help"]

volumes:
  paddle_models:
    name: paddle_models
  deepseek_models:
    name: deepseek_models
```

## scrape.sh

```sh
#!/bin/bash
# scrape.sh - Updated script that uses the workflow-based scraper

# Display help information
function show_help {
    echo "Usage: $0 <url> [options]"
    echo ""
    echo "Options:"
    echo "  --output, -o DIR       Output directory (default: docs)"
    echo "  --depth, -d NUM        Maximum crawl depth (default: 3)"
    echo "  --delay, -w SEC        Delay between requests in seconds (default: 1.0)"
    echo "  --llm URL              DeepSeek LLM endpoint (default: http://deepseek:8000)"
    echo "  --visible              Run browser in visible mode (not headless)"
    echo "  --debug                Enable debug mode (save intermediate files)"
    echo "  --max-pages NUM        Maximum number of pages to scrape (default: 100)"
    echo "  --solve-captcha        Open browser on your Mac to solve CAPTCHA first"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 https://docs.example.com --depth 2 --debug"
    echo "  $0 https://docs.example.com --solve-captcha"
    exit 0
}

# Check if no arguments or help requested
if [ "$#" -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

# Extract the URL (first argument)
URL="$1"
shift

# Process arguments
DOCKER_ARGS=""
SOLVE_CAPTCHA=0

while [ "$#" -gt 0 ]; do
    case "$1" in
    --output | -o)
        DOCKER_ARGS="$DOCKER_ARGS --output $2"
        shift 2
        ;;
    --depth | -d)
        DOCKER_ARGS="$DOCKER_ARGS --depth $2"
        shift 2
        ;;
    --delay | -w)
        DOCKER_ARGS="$DOCKER_ARGS --delay $2"
        shift 2
        ;;
    --llm)
        DOCKER_ARGS="$DOCKER_ARGS --llm $2"
        shift 2
        ;;
    --visible)
        DOCKER_ARGS="$DOCKER_ARGS --visible"
        shift
        ;;
    --debug)
        DOCKER_ARGS="$DOCKER_ARGS --debug"
        shift
        ;;
    --max-pages)
        DOCKER_ARGS="$DOCKER_ARGS --max-pages $2"
        shift 2
        ;;
    --solve-captcha)
        SOLVE_CAPTCHA=1
        shift
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        ;;
    esac
done

# Ensure the DeepSeek server is running
echo "Ensuring DeepSeek server is running..."
docker-compose up -d deepseek

# Wait for DeepSeek to be ready
echo "Waiting for DeepSeek server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/status || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "DeepSeek server is ready!"
        break
    fi
    echo "Waiting for DeepSeek server... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: DeepSeek server did not become ready in time"
    exit 1
fi

# If --solve-captcha flag is set, solve the CAPTCHA on the host machine
if [ $SOLVE_CAPTCHA -eq 1 ]; then
    echo "Opening browser on your Mac to solve CAPTCHA..."

    # Check if the host browser script exists
    if [ ! -f "solve_captcha.py" ]; then
        echo "Creating solver script..."
        cat >solve_captcha.py <<'PYTHON_SCRIPT'
#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not found. Installing required packages...")
    os.system("pip install playwright")
    os.system("playwright install chromium")
    from playwright.sync_api import sync_playwright

def solve_cloudflare(url, timeout=300):
    print("\n" + "="*80)
    print("CloudFlare Protection Detector")
    print("="*80)
    print(f"Opening browser window for you to solve the CAPTCHA for: {url}")
    print("Please complete the verification in the browser window.")
    print("The browser will automatically close once the page fully loads.")
    print("="*80 + "\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="domcontentloaded")
            print("Waiting for you to solve the CAPTCHA...")
            
            def is_cloudflare_bypassed():
                cloudflare_text = page.locator("text=Cloudflare").count()
                captcha_text = page.locator("text=Please verify").count()
                challenge_text = page.locator("text=Just a moment").count()
                return cloudflare_text == 0 and captcha_text == 0 and challenge_text == 0
            
            start_time = time.time()
            while not is_cloudflare_bypassed():
                if time.time() - start_time > timeout:
                    print("Timed out waiting for CloudFlare bypass.")
                    return None
                time.sleep(1)
            
            print("CloudFlare challenge bypassed! Saving session...")
            page.wait_for_timeout(3000)
            
            storage_dir = Path("session_data")
            storage_dir.mkdir(exist_ok=True)
            
            storage_file = storage_dir / "cloudflare_session.json"
            context.storage_state(path=str(storage_file))
            
            print(f"Session saved to: {storage_file}")
            return storage_file
            
        finally:
            browser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solve_captcha.py <url>")
        sys.exit(1)
        
    url = sys.argv[1]
    session_file = solve_cloudflare(url)
    
    if session_file:
        print("\nSuccess! You can now mount this session file in the Docker container.")
    else:
        print("\nFailed to bypass CloudFlare protection. Please try again.")
        sys.exit(1)
PYTHON_SCRIPT

        chmod +x solve_captcha.py
    fi

    # Run the solver on the host (your Mac)
    python solve_captcha.py "$URL"

    # Check if the session file was created
    if [ -f "session_data/cloudflare_session.json" ]; then
        echo "CAPTCHA solved successfully!"
    else
        echo "Failed to solve CAPTCHA. Please try again."
        exit 1
    fi
fi

# Run the workflow-based scraper
echo "Starting workflow-based scraper for URL: $URL"

# Build the final command
# Mount the session directory if it exists
if [ -d "session_data" ]; then
    CMD="docker-compose run -v \"$(pwd)/session_data:/app/session_data\" --rm scraper python workflow_scraper.py \"$URL\" $DOCKER_ARGS"
else
    CMD="docker-compose run --rm scraper python workflow_scraper.py \"$URL\" $DOCKER_ARGS"
fi

# Execute the command
eval $CMD

echo "Scraping completed. Results saved to ./output directory."
```

## scraper/Dockerfile

```scraper/Dockerfile
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
```

## scraper/advanced_doc_scraper.py

```py
#!/usr/bin/env python3
"""
Advanced Documentation Scraper with AI Integration

Features:
- Handles SPAs with JavaScript rendering
- Uses PaddleOCR for visual content recognition
- Integrates with DeepSeek LLM for intelligent content extraction
- Combines HTML structure and visual analysis
"""

import os
import time
import json
import argparse
from urllib.parse import urljoin, urlparse
import re
import numpy as np
from typing import Dict, Any

# Web scraping & browser automation
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page
import html2text

# Visual AI
from paddleocr import PaddleOCR


class AIEnhancedScraper:
    def __init__(
        self,
        base_url: str,
        output_dir: str = "docs",
        max_depth: int = 1,
        delay: float = 1.0,
        headless: bool = True,
        llm_endpoint: str = "http://deepseek:8000",  # Updated to use service name in Docker network
        debug: bool = False,
    ):
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.delay = delay
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.debug = debug

        self.visited_urls = set()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

        # Initialize browser
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)

        # Initialize OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.body_width = 0

    def __del__(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except:
            pass

    def scrape(self):
        """Start the scraping process"""
        print(f"Starting to scrape {self.base_url}")

        try:
            context = self.browser.new_context(viewport={"width": 1280, "height": 1024})
            page = context.new_page()
            self._scrape_page(page, self.base_url, 0)
        finally:
            context.close()

        print(
            f"Scraping complete. {len(self.visited_urls)} pages scraped to {self.output_dir}/"
        )

    def _scrape_page(self, page: Page, url: str, depth: int):
        """Scrape a single page and its links up to max_depth"""
        if url in self.visited_urls or depth > self.max_depth:
            return

        print(f"Scraping: {url}")
        self.visited_urls.add(url)

        # Be polite - add a delay between requests
        time.sleep(self.delay)

        try:
            # Navigate to the page and wait for it to load completely
            page.goto(url, wait_until="networkidle")

            # Extract page title
            title = page.title()

            # Get the HTML content after JavaScript execution
            html_content = page.content()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Take a screenshot for visual analysis
            screenshot_path = (
                os.path.join(
                    self.output_dir, "debug", f"{self._url_to_filename(url)}.png"
                )
                if self.debug
                else None
            )
            if self.debug:
                page.screenshot(path=screenshot_path, full_page=True)

            # Extract text from the page using OCR if needed
            visual_content = {}
            if self.debug and screenshot_path and os.path.exists(screenshot_path):
                visual_content = self._extract_visual_content(screenshot_path)

            # Use LLM to identify important content sections
            main_content = self._extract_content_with_ai(soup, visual_content, url)

            # Convert to markdown
            markdown = self.html2text.handle(str(main_content))

            # Remove excess newlines
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)

            # Save to file
            self._save_markdown(url, title, markdown)

            # Scrape linked pages if not at max depth
            if depth < self.max_depth:
                self._scrape_links(page, soup, url, depth)

        except Exception as e:
            print(f"Error processing {url}: {e}")

    def _extract_visual_content(self, screenshot_path: str) -> Dict[str, Any]:
        """Extract visual content using PaddleOCR"""
        if not os.path.exists(screenshot_path):
            return {}

        results = self.ocr.ocr(screenshot_path)
        if results is None or len(results) == 0:
            return {"text_blocks": []}

        visual_data = {"text_blocks": [], "tables": []}

        # Process OCR results
        for idx, line in enumerate(results[0]):
            if line is None:
                continue

            position = line[0]
            text = line[1][0]
            confidence = line[1][1]

            visual_data["text_blocks"].append(
                {
                    "id": idx,
                    "position": position,
                    "text": text,
                    "confidence": confidence,
                }
            )

        return visual_data

    def _extract_content_with_ai(
        self, soup: BeautifulSoup, visual_content: Dict[str, Any], url: str
    ) -> str:
        """Use LLM to identify and extract relevant content"""
        # First attempt with common selectors
        for selector in [
            "main",
            "article",
            "div.content",
            "div.documentation",
            ".markdown-body",
        ]:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 100:
                return content

        # If no standard selectors worked, use the LLM for content identification
        try:
            # Prepare input for the LLM
            # Get the simplified HTML structure
            html_structure = self._get_simplified_html(soup)

            # Create a prompt for the LLM
            prompt = f"""
            You are an AI assistant helping with web scraping. Analyze this HTML structure and identify the main content
            elements that contain documentation. Return a list of CSS selectors that would extract the main content.
            Avoid selecting navigation, headers, footers, and sidebars.
            
            URL: {url}
            
            HTML Structure:
            {html_structure[:2000]}  # Limiting size to avoid token limits
            
            Visual elements detected:
            {json.dumps(visual_content)[:1000] if visual_content else "No visual data available"}
            
            Return only a JSON array of CSS selectors, ordered by priority.
            """

            # Get selectors from LLM
            response = self._call_llm_api(prompt)

            try:
                selectors = json.loads(response)
                if not isinstance(selectors, list):
                    selectors = [selectors]
            except:
                # Fallback if response isn't valid JSON
                selectors = [
                    s.strip()
                    for s in response.split("\n")
                    if s.strip() and not s.startswith("#")
                ]

            # Try each selector
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # Combine all matching elements
                    content_html = "".join(str(el) for el in elements)
                    if len(content_html) > 100:  # Ensure we got substantial content
                        return content_html

        except Exception as e:
            print(f"Error using LLM for content extraction: {e}")

        # Fallback: remove obvious non-content elements
        for nav in soup.select("nav, header, footer, .sidebar, .menu, .navigation"):
            nav.decompose()

        return str(soup.body)

    def _call_llm_api(self, prompt: str) -> str:
        """Call the DeepSeek LLM API"""
        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={"prompt": prompt, "max_new_tokens": 1024, "temperature": 0.3},
                timeout=60,
            )

            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print(f"LLM API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return ""

    def _get_simplified_html(self, soup: BeautifulSoup) -> str:
        """Create a simplified version of the HTML for LLM analysis"""
        structure = []

        def process_element(element, depth=0):
            if not hasattr(element, "name") or not element.name:
                return

            # Get element attributes
            attrs = {}
            for attr in ["id", "class", "role"]:
                if element.has_attr(attr):
                    attrs[attr] = element[attr]

            # Create element description
            el_desc = {"tag": element.name, "depth": depth, "attrs": attrs}

            # Add text length if element has direct text
            text = element.get_text(strip=True)
            if text:
                # Only add text info if it's direct text and not all from children
                direct_text = "".join(
                    t.strip() for t in element.find_all(text=True, recursive=False)
                )
                if direct_text:
                    el_desc["text_length"] = len(direct_text)
                    if len(direct_text) < 50:
                        el_desc["text"] = direct_text

            structure.append(el_desc)

            # Process children
            for child in element.children:
                if hasattr(child, "name") and child.name:
                    process_element(child, depth + 1)

        process_element(soup.body)
        return json.dumps(structure, indent=2)

    def _scrape_links(
        self, page: Page, soup: BeautifulSoup, current_url: str, depth: int
    ):
        """Find and scrape links on the page"""
        links = page.evaluate(
            """
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(href => !href.startsWith('#'));
                return Array.from(new Set(links));  // Remove duplicates
            }
        """
        )

        for href in links:
            # Handle relative URLs
            full_url = urljoin(current_url, href)

            # Make sure URL is from the same domain
            if not self._is_same_domain(full_url):
                continue

            # Scrape the linked page
            self._scrape_page(page, full_url, depth + 1)

    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is from the same domain as base_url"""
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        return parsed_base.netloc == parsed_url.netloc

    def _url_to_filename(self, url: str) -> str:
        """Convert a URL to a valid filename"""
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "index"

        # Clean path and create appropriate filename
        clean_path = re.sub(r"[^a-zA-Z0-9/]", "_", path)
        if path.endswith("/"):
            clean_path += "index"

        return clean_path

    def _save_markdown(self, url: str, title: str, content: str):
        """Save markdown content to file"""
        clean_path = self._url_to_filename(url)
        filename = f"{clean_path}.md"

        # Create subdirectories if needed
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"Source: {url}\n\n")
            f.write(content)


def main():
    parser = argparse.ArgumentParser(description="AI-Enhanced Documentation Scraper")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--output", "-o", default="docs", help="Output directory")
    parser.add_argument(
        "--depth", "-d", type=int, default=1, help="Maximum crawl depth"
    )
    parser.add_argument(
        "--delay",
        "-w",
        type=float,
        default=1.0,
        help="Delay between requests (seconds)",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("LLM_ENDPOINT", "http://deepseek:8000"),
        help="DeepSeek LLM endpoint",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Run browser in visible mode (not headless)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (save screenshots, etc.)",
    )

    args = parser.parse_args()

    scraper = AIEnhancedScraper(
        base_url=args.url,
        output_dir=args.output,
        max_depth=args.depth,
        delay=args.delay,
        headless=not args.visible,
        llm_endpoint=args.llm,
        debug=args.debug,
    )

    try:
        scraper.scrape()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {e}")


if __name__ == "__main__":
    main()
```

## scraper/download_paddle_models.py

```py
#!/usr/bin/env python3
"""
Script to pre-download PaddleOCR models to avoid downloading them at runtime.
"""
from paddleocr import PaddleOCR
import os
import time
import sys

# Set the HOME environment variable to ensure PaddleOCR saves models in the right place
os.environ["HOME"] = "/root"


def download_models():
    print("Pre-downloading PaddleOCR models...")
    try:
        # Initialize PaddleOCR - this will download the models
        # Use timeout to prevent hanging if download servers are slow
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=True)

        # Force model loading to trigger downloads
        test_img = os.path.join(os.path.dirname(__file__), "test_image.png")

        # Create a small test image if it doesn't exist
        if not os.path.exists(test_img):
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (100, 30), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Hello", fill=(0, 0, 0))
            img.save(test_img)

        # Run inference to ensure models are downloaded
        result = ocr.ocr(test_img, cls=True)
        print("Model test successful!")

        print("PaddleOCR models downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False


# Try to download models with retry logic
max_attempts = 3
for attempt in range(max_attempts):
    print(f"Download attempt {attempt+1}/{max_attempts}")
    if download_models():
        sys.exit(0)
    time.sleep(5)  # Wait before retrying

sys.exit(1)  # Exit with error if all attempts failed
```

## scraper/requirements.txt

```txt
beautifulsoup4==4.12.2
playwright==1.39.0
requests==2.31.0
html2text==2020.1.16
paddleocr==2.7.0
paddlepaddle==2.5.2
numpy==1.24.3
Pillow==10.0.1
python-dateutil==2.8.2
lxml==4.9.3
```
