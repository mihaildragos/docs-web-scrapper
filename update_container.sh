#!/bin/bash
# update_container.sh - Update the scraper container with the latest script

echo "Updating documentation scraper container..."

# Ensure the updated script is in the scraper directory
if [ ! -f "scraper/advanced_doc_scraper.py" ]; then
    echo "Error: scraper/advanced_doc_scraper.py not found"
    echo "Please ensure you've placed the new script in the scraper directory"
    exit 1
fi

# Stop running containers
echo "Stopping any running containers..."
docker-compose down

# Rebuild the scraper container
echo "Rebuilding the scraper container..."
docker-compose build scraper

echo "Update complete! You can now run ./scrape.sh with the --workers argument."
echo "Example: ./scrape.sh https://docs.shopify.com/apps --depth 4 --debug --workers 8"
