#!/bin/bash
# scrape.sh - Updated script with improved model loading wait logic

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
    echo "  --max-pages NUM        Maximum pages to scrape (default: 100)"
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
MAX_RETRIES=60 # Increased from 30 to 60
RETRY_COUNT=0
MODEL_LOADED=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/status || echo "000")

    if [ "$STATUS" = "200" ]; then
        # Check if model is actually loaded
        MODEL_STATUS=$(curl -s http://localhost:8000/status)
        if echo "$MODEL_STATUS" | grep -q '"is_loaded": true'; then
            echo "DeepSeek server is ready with model loaded!"
            MODEL_LOADED=1
            break
        else
            echo "DeepSeek server is responding but model is still loading... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        fi
    else
        echo "Waiting for DeepSeek server to respond... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 10 # Increased from 5 to 10 seconds
done

if [ $MODEL_LOADED -eq 0 ]; then
    echo "Warning: DeepSeek model may not be fully loaded yet. Proceeding anyway, but you might encounter errors."
    echo "You may want to cancel (Ctrl+C) and try again later, or adjust memory allocation for Docker."
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to cancel..."
    read -r
fi

# Solve CAPTCHA if needed (existing code)
if [ $SOLVE_CAPTCHA -eq 1 ]; then
    # (CAPTCHA solving code unchanged)
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
