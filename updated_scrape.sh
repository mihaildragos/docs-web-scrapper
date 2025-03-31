#!/bin/bash
# scrape.sh - Enhanced script with CloudFlare bypass support

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
    echo "  --debug                Enable debug mode (save screenshots, etc.)"
    echo "  --workers NUM          Maximum number of worker threads (default: 4)"
    echo "  --session FILE         Use saved browser session (for CloudFlare bypass)"
    echo "  --solve-captcha        Open browser to manually solve CAPTCHA first"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 https://docs.example.com --depth 2 --debug"
    echo "  $0 https://docs.example.com --solve-captcha"
    echo "  $0 https://docs.example.com --session session_data/cloudflare_session.json"
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
SESSION_FILE=""

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
    --workers)
        # We'll check if the workers argument is supported
        WORKERS_ARG="--workers $2"
        shift 2
        ;;
    --session)
        SESSION_FILE="$2"
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

# If --solve-captcha flag is set, solve the CAPTCHA first
if [ $SOLVE_CAPTCHA -eq 1 ]; then
    echo "Opening browser to solve CAPTCHA..."
    SESSION_FILE=$(docker-compose run --rm scraper python solve_cloudflare.py "$URL" | grep -o "session_data/cloudflare_session.json")

    if [ -z "$SESSION_FILE" ]; then
        echo "Failed to solve CAPTCHA. Please try again."
        exit 1
    fi

    echo "CAPTCHA solved successfully! Session saved to $SESSION_FILE"
fi

# Add session file to arguments if provided
if [ ! -z "$SESSION_FILE" ]; then
    DOCKER_ARGS="$DOCKER_ARGS --session $SESSION_FILE"
    echo "Using session file: $SESSION_FILE"
fi

# Check if the container supports the necessary arguments
echo "Checking container version..."
HELP_OUTPUT=$(docker-compose run --rm scraper python advanced_doc_scraper.py --help)
SUPPORTS_WORKERS=$(echo "$HELP_OUTPUT" | grep -c "\-\-workers")
SUPPORTS_SESSION=$(echo "$HELP_OUTPUT" | grep -c "\-\-session")

# Run the scraper with appropriate arguments
echo "Starting scraper for URL: $URL"

# Build the final command
CMD="docker-compose run --rm scraper python advanced_doc_scraper.py \"$URL\" $DOCKER_ARGS"

# Add workers argument if supported
if [ "$SUPPORTS_WORKERS" -gt 0 ] && [ ! -z "$WORKERS_ARG" ]; then
    echo "Using workers argument: $WORKERS_ARG"
    CMD="$CMD $WORKERS_ARG"
elif [ ! -z "$WORKERS_ARG" ]; then
    echo "Container version doesn't support workers argument, ignoring it"
fi

# Check if session support is available
if [ ! -z "$SESSION_FILE" ] && [ "$SUPPORTS_SESSION" -eq 0 ]; then
    echo "WARNING: Container version doesn't support session files."
    echo "You may need to update the container with the latest advanced_doc_scraper.py version."
fi

# Execute the command
eval $CMD

echo "Scraping completed. Results saved to ./output directory."
