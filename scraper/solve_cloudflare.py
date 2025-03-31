#!/usr/bin/env python3
"""
Improved CloudFlare solver with robust Playwright installation
and network timeout handling.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path


def install_dependencies():
    """Install required dependencies with increased timeout and verbosity"""
    print("Installing Playwright with increased timeout...")
    try:
        # First attempt: install with higher timeout
        subprocess.run(
            ["pip", "install", "playwright", "--timeout", "120", "-v"], check=True
        )

        # Install browsers with higher timeout
        print("Installing Playwright browsers...")
        subprocess.run(
            ["python", "-m", "playwright", "install", "chromium", "--with-deps"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        print("\nAutomatic installation failed. Let's try manual steps:")
        print("\n1. First, let's try installing with a different mirror:")
        try:
            subprocess.run(
                [
                    "pip",
                    "install",
                    "playwright",
                    "-i",
                    "https://pypi.tuna.tsinghua.edu.cn/simple",
                ],
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            print("\n2. Please try manual installation:")
            print("   a. Open a new terminal")
            print("   b. Run: pip install playwright")
            print("   c. Run: playwright install chromium")
            print("\nAfter installation completes, run this script again.")
            return False


def solve_cloudflare(url, timeout=300):
    """
    Open a browser window to manually solve the CloudFlare challenge,
    then save the authenticated cookies for reuse.
    """
    # Try importing playwright, install if necessary
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        success = install_dependencies()
        if not success:
            return None

        # Try importing again after installation
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("Failed to import Playwright even after installation.")
            print("Please try installing manually and run again.")
            return None

    print("\n" + "=" * 80)
    print("CloudFlare Protection Detector")
    print("=" * 80)
    print(f"Opening browser window for you to solve the CAPTCHA for: {url}")
    print("Please complete the verification in the browser window.")
    print("The browser will automatically close once the page fully loads.")
    print("=" * 80 + "\n")

    with sync_playwright() as p:
        try:
            # Launch a visible browser for manual interaction
            browser = p.chromium.launch(headless=False)
        except Exception as e:
            print(f"Error launching browser: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Chromium was installed correctly")
            print("2. Try running: playwright install chromium --with-deps")
            print(
                "3. If you have a firewall or VPN, check if it's blocking browser launch"
            )
            return None

        context = browser.new_context()
        page = context.new_page()

        try:
            # Navigate to the URL
            print(f"Navigating to {url}...")
            try:
                page.goto(url, timeout=60000)  # Longer timeout (60 seconds)
            except Exception as e:
                print(f"Navigation error: {e}")
                print("The site may be slow or blocking automated access.")
                print("Continuing anyway to see if we can interact with what loaded...")

            # Print instructions
            print("Waiting for you to solve the CAPTCHA...")

            # Function to check if the page has bypassed CloudFlare
            def is_cloudflare_bypassed():
                try:
                    # Check if we're still on the CloudFlare page
                    cloudflare_text = page.locator("text=Cloudflare").count()
                    captcha_text = page.locator("text=Please verify").count()
                    challenge_text = page.locator("text=Just a moment").count()

                    # If none of these elements exist, we've likely bypassed CloudFlare
                    return (
                        cloudflare_text == 0
                        and captcha_text == 0
                        and challenge_text == 0
                    )
                except Exception:
                    # If we can't check, assume we're still on the challenge page
                    return False

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

        except Exception as e:
            print(f"Error during CAPTCHA solving: {e}")
            return None
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
