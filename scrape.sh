#!/bin/bash
# scrape.sh - Convenience script for running the document scraper

# Check if a URL was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <url> [additional options]"
    echo "Example: $0 https://docs.example.com --depth 2"
    exit 1
fi

# Extract the URL (first argument)
URL="$1"
shift

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

# Run the scraper with all arguments
echo "Starting scraper for URL: $URL"
docker-compose run --rm scraper python advanced_doc_scraper.py "$URL" "$@"

echo "Scraping completed. Results saved to ./output directory."
