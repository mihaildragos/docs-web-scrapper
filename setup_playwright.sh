#!/bin/bash
# setup_playwright.sh - Manual setup for Playwright

echo "==================== Playwright Manual Setup ===================="
echo "This script will install Playwright and its dependencies for"
echo "solving CloudFlare CAPTCHAs on your Mac."
echo "=============================================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install playwright with increased timeout
echo "Installing Playwright (this may take a while)..."
pip install --default-timeout=120 playwright

# Install browser
echo "Installing Chromium browser (this may take a while)..."
python -m playwright install chromium --with-deps

# Create directories
echo "Creating session directory..."
mkdir -p session_data

echo "=============================================================="
echo "Installation complete! You can now run:"
echo "./scrape.sh https://docs.shopify.com/apps --solve-captcha"
echo "=============================================================="
