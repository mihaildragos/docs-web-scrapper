#!/usr/bin/env python3
"""
Script to handle CloudFlare protection in the documentation scraper.
This adds interactive CAPTCHA solving and session persistence.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Playwright for browser automation
from playwright.sync_api import sync_playwright, Page, Browser


def solve_cloudflare(url, timeout=300):
    """
    Open a browser window to manually solve the CloudFlare challenge,
    then save the authenticated cookies for reuse.

    Args:
        url: The URL to visit and solve the CAPTCHA for
        timeout: Maximum time to wait for manual solving (seconds)

    Returns:
        Path to the saved storage state file
    """
    print("\n" + "=" * 80)
    print("CloudFlare Protection Detected!")
    print("=" * 80)
    print(f"Opening browser window for you to solve the CAPTCHA for: {url}")
    print("Please complete the verification in the browser window.")
    print("The browser will automatically close once the page fully loads.")
    print("=" * 80 + "\n")

    with sync_playwright() as p:
        # Launch a visible browser for manual interaction
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        try:
            # Navigate to the URL
            page.goto(url, wait_until="domcontentloaded")

            # Print instructions
            print("Waiting for you to solve the CAPTCHA...")

            # Function to check if the page has bypassed CloudFlare
            def is_cloudflare_bypassed():
                # Check if we're still on the CloudFlare page
                cloudflare_text = page.locator("text=Cloudflare").count()
                captcha_text = page.locator("text=Please verify").count()
                challenge_text = page.locator("text=Just a moment").count()

                # If none of these elements exist, we've likely bypassed CloudFlare
                return (
                    cloudflare_text == 0 and captcha_text == 0 and challenge_text == 0
                )

            # Wait for the user to solve the CAPTCHA and the page to load
            start_time = time.time()
            while not is_cloudflare_bypassed():
                if time.time() - start_time > timeout:
                    print("Timed out waiting for CloudFlare bypass.")
                    return None
                time.sleep(1)

            # Wait an extra moment for the page to fully load
            print("CloudFlare challenge bypassed! Saving session...")
            page.wait_for_timeout(3000)  # Wait 3 seconds

            # Save the authenticated session
            storage_dir = Path("session_data")
            storage_dir.mkdir(exist_ok=True)

            storage_file = storage_dir / "cloudflare_session.json"
            context.storage_state(path=str(storage_file))

            print(f"Session saved to: {storage_file}")
            return storage_file

        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Solve CloudFlare CAPTCHA and save the session"
    )
    parser.add_argument("url", help="URL to visit and solve the CAPTCHA for")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum time to wait for manual solving (seconds)",
    )

    args = parser.parse_args()

    session_file = solve_cloudflare(args.url, args.timeout)
    if session_file:
        print("\nSuccess! Now you can run the scraper with the authenticated session:")
        print(f"./scrape.sh {args.url} --session {session_file}\n")
    else:
        print("\nFailed to bypass CloudFlare protection. Please try again.")


if __name__ == "__main__":
    main()
