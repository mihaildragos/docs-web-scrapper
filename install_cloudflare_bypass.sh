#!/bin/bash
# install_cloudflare_bypass.sh - Install the CloudFlare bypass functionality

echo "Installing CloudFlare bypass functionality..."

# Create directory structure
mkdir -p scraper/session_data

# Copy the files
echo "Adding solve_cloudflare.py to the scraper container..."
cp solve_cloudflare.py scraper/

# Update the scrape.sh script
echo "Updating scrape.sh with session support..."
cp -f scrape.sh scrape.sh.backup
cp updated_scrape.sh scrape.sh
chmod +x scrape.sh

echo "Updating the container..."
docker-compose build scraper

echo "==============================================================="
echo "Installation complete!"
echo "==============================================================="
echo ""
echo "To use the CloudFlare bypass:"
echo ""
echo "1. Solve the CAPTCHA once:"
echo "   ./scrape.sh https://docs.shopify.com/apps --solve-captcha"
echo ""
echo "2. Then use the saved session for scraping:"
echo "   ./scrape.sh https://docs.shopify.com/apps --depth 4 --debug --session session_data/cloudflare_session.json"
echo ""
echo "The session will remain valid for some time (typically hours to days)."
echo "==============================================================="
