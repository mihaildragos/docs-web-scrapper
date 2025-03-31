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
