#!/usr/bin/env python3
"""
Workflow-based Documentation Scraper

An advanced workflow-based documentation scraper that combines visual analysis,
AI-powered selector finding, content extraction, and intelligent traversal planning.

Example usage:
    python workflow_scraper.py https://docs.example.com --output docs --depth 3
"""

import os
import sys
import time
import json
import argparse
import random
import requests
from collections import deque
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Any, Set, Deque, Tuple

# Playwright for browser automation
from playwright.sync_api import sync_playwright, Page, Browser

# Import the workflow modules
from lib.navigation_extractor import NavigationExtractor
from lib.visual_analyzer import VisualAnalyzer
from lib.selector_finder import SelectorFinder
from lib.content_extractor import ContentExtractor
from lib.content_processor import ContentProcessor
from lib.traversal_planner import TraversalPlanner


class WorkflowScraper:
    def __init__(
        self,
        base_url: str,
        output_dir: str = "docs",
        max_depth: int = 3,
        delay: float = 1.0,
        headless: bool = True,
        llm_endpoint: str = "http://deepseek:8000",
        debug: bool = False,
        max_pages: int = 100,
    ):
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.delay = delay
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.debug = debug
        self.max_pages = max_pages

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

        # Initialize browser
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)

        # Track visited and queued URLs
        self.visited_urls = set()
        self.urls_queue = deque()  # (url, depth, priority, source_url)

        # Initialize workflow components
        self.navigation_extractor = NavigationExtractor(
            debug=debug, output_dir=output_dir
        )
        self.visual_analyzer = VisualAnalyzer(debug=debug, output_dir=output_dir)
        self.selector_finder = SelectorFinder(
            llm_endpoint=llm_endpoint, debug=debug, output_dir=output_dir
        )
        self.content_extractor = ContentExtractor(
            base_url=base_url, debug=debug, output_dir=output_dir
        )
        self.content_processor = ContentProcessor(
            llm_endpoint=llm_endpoint, debug=debug, output_dir=output_dir
        )
        self.traversal_planner = TraversalPlanner(
            base_url=base_url,
            llm_endpoint=llm_endpoint,
            debug=debug,
            output_dir=output_dir,
        )

        # Save site metadata
        self.site_metadata = {
            "base_url": base_url,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pages": {},
        }

    def __del__(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except:
            pass

    def scrape(self):
        """Start the scraping process"""
        print(f"Starting workflow-based scrape of {self.base_url}")

        # Add starting URL to queue
        self.urls_queue.append(
            (self.base_url, 0, 10, None)
        )  # (url, depth, priority, source_url)

        # Process URLs in the queue
        pages_scraped = 0

        try:
            # Create a persistent browser context
            context = self.browser.new_context(viewport={"width": 1280, "height": 1024})

            # Try to load the session data if it exists
            if os.path.exists("session_data/cloudflare_session.json"):
                try:
                    with open("session_data/cloudflare_session.json", "r") as f:
                        print("Loading saved CloudFlare session...")
                        session_data = json.load(f)
                        context.add_cookies(session_data.get("cookies", []))
                except Exception as e:
                    print(f"Error loading session data: {e}")

            # Create a new page
            page = context.new_page()

            # Process the queue
            while self.urls_queue and pages_scraped < self.max_pages:
                # Get next URL from queue
                url, depth, priority, source_url = self.urls_queue.popleft()

                # Skip if already visited or beyond max depth
                if url in self.visited_urls or depth > self.max_depth:
                    continue

                # Process the page
                success = self._process_page(page, url, depth, source_url)

                if success:
                    pages_scraped += 1
                    print(
                        f"Processed {pages_scraped}/{self.max_pages} pages ({len(self.urls_queue)} in queue)"
                    )

                # Be polite - add a delay between requests
                time.sleep(self.delay)

        finally:
            # Save site metadata
            self.site_metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.site_metadata["pages_scraped"] = pages_scraped

            with open(os.path.join(self.output_dir, "site_metadata.json"), "w") as f:
                json.dump(self.site_metadata, f, indent=2)

            # Close browser
            try:
                context.close()
            except:
                pass

        print(f"Scraping complete. {pages_scraped} pages scraped to {self.output_dir}/")

    def _process_page(self, page: Page, url: str, depth: int, source_url: str) -> bool:
        """Process a single page using the workflow"""
        print(f"\n[Depth {depth}] Processing: {url}")

        try:
            # Mark URL as visited
            self.visited_urls.add(url)

            # Step 1: Navigate to URL and extract navigation structure
            start_time = time.time()
            print(f"  Step 1: Navigating to URL and extracting navigation...")

            # Navigate to the page and wait for it to load
            page.goto(url, wait_until="networkidle", timeout=60000)

            # Extract navigation structure
            navigation = self.navigation_extractor.extract_navigation(page, url)

            navigation_time = time.time() - start_time
            print(f"    ✓ Navigation extracted in {navigation_time:.2f}s")

            # Step 2: Analyze the page visually
            start_time = time.time()
            print(f"  Step 2: Analyzing page visually...")

            # Analyze the page structure
            visual_analysis = self.visual_analyzer.analyze_page_structure(page, url)

            visual_time = time.time() - start_time
            print(f"    ✓ Visual analysis completed in {visual_time:.2f}s")

            # Step 3: Find HTML selectors for content extraction
            start_time = time.time()
            print(f"  Step 3: Finding HTML selectors...")

            # Get the HTML content
            html_content = page.content()

            # Find selectors using DeepSeek
            selectors = self.selector_finder.find_content_selectors(
                url, html_content, visual_analysis
            )

            selector_time = time.time() - start_time
            print(f"    ✓ Selectors found in {selector_time:.2f}s")

            # Step 4: Extract content using the selectors
            start_time = time.time()
            print(f"  Step 4: Extracting content...")

            # Extract content
            extracted_content = self.content_extractor.extract_content(
                url, html_content, selectors
            )

            extraction_time = time.time() - start_time
            print(f"    ✓ Content extracted in {extraction_time:.2f}s")

            # Step 5: Process and organize the content
            start_time = time.time()
            print(f"  Step 5: Processing content...")

            # Process the content
            processed_content = self.content_processor.process_content(
                url, extracted_content
            )

            processing_time = time.time() - start_time
            print(f"    ✓ Content processed in {processing_time:.2f}s")

            # Save the processed content
            self._save_content(url, processed_content)

            # Step 6: Plan the traversal to determine next pages
            if depth < self.max_depth:
                start_time = time.time()
                print(f"  Step 6: Planning traversal...")

                # Plan the traversal
                next_urls = self.traversal_planner.plan_traversal(
                    url, extracted_content, processed_content
                )

                # Add next URLs to the queue
                self._add_urls_to_queue(next_urls, depth + 1, url)

                traversal_time = time.time() - start_time
                print(
                    f"    ✓ Traversal planned in {traversal_time:.2f}s with {len(next_urls)} candidate URLs"
                )

            # Update site metadata for this page
            self.site_metadata["pages"][url] = {
                "title": extracted_content.get("title", ""),
                "depth": depth,
                "source_url": source_url,
                "word_count": extracted_content.get("main_content", {}).get(
                    "word_count", 0
                ),
                "processing_times": {
                    "navigation": navigation_time,
                    "visual": visual_time,
                    "selector": selector_time,
                    "extraction": extraction_time,
                    "processing": processing_time,
                    "total": navigation_time
                    + visual_time
                    + selector_time
                    + extraction_time
                    + processing_time,
                },
            }

            return True

        except Exception as e:
            print(f"Error processing {url}: {e}")

            # Add to metadata as failed
            self.site_metadata["pages"][url] = {
                "error": str(e),
                "depth": depth,
                "source_url": source_url,
                "status": "failed",
            }

            return False

    def _add_urls_to_queue(
        self, next_urls: List[Dict[str, Any]], depth: int, source_url: str
    ):
        """Add URLs to the processing queue"""
        # Convert list to a priority queue
        urls_to_add = []

        for url_info in next_urls:
            url = url_info.get("url", "")
            priority = url_info.get("priority", 5)

            # Skip if already visited or in queue
            if url in self.visited_urls or any(url == u[0] for u in self.urls_queue):
                continue

            urls_to_add.append((url, depth, priority, source_url))

        # Sort by priority (highest first)
        urls_to_add.sort(key=lambda x: x[2], reverse=True)

        # Add to queue
        for url_info in urls_to_add:
            self.urls_queue.append(url_info)

    def _save_content(self, url: str, processed_content: Dict[str, Any]):
        """Save processed content to file"""
        # Create filename from URL
        filename = self._url_to_filename(url)

        # Ensure directory exists
        filepath = os.path.join(self.output_dir, f"{filename}.md")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write content to file
        with open(filepath, "w", encoding="utf-8") as f:
            title = processed_content.get("title", "Untitled")
            f.write(f"# {title}\n\n")
            f.write(f"Source: {url}\n\n")
            f.write(processed_content.get("processed_markdown", ""))

    def _url_to_filename(self, url: str) -> str:
        """Convert a URL to a valid filename"""
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "index"

        # Clean path and create appropriate filename
        import re

        clean_path = re.sub(r"[^a-zA-Z0-9/]", "_", path)
        if path.endswith("/"):
            clean_path += "index"

        return clean_path

    def call_llm_with_retry(self, prompt, max_retries=8, initial_backoff=3):
        """
        Call the DeepSeek LLM API with exponential backoff retry logic and robust error handling
        
        Args:
            prompt (str): The prompt to send to the LLM
            max_retries (int): Maximum number of retry attempts
            initial_backoff (int): Initial backoff time in seconds
            
        Returns:
            str: The LLM response text or empty string on failure
        """
        retry_count = 0
        backoff = initial_backoff
        
        # Function to log the retry attempt
        def log_retry(reason, attempt, wait_time):
            print(f"LLM API retry ({attempt}/{max_retries}): {reason} - waiting {wait_time:.1f}s...")
        
        while retry_count < max_retries:
            try:
                # Add a timeout that increases with retry attempts for especially slow servers
                timeout = 30 + (retry_count * 30)  # Start with 30s, add 30s each retry
                
                response = requests.post(
                    f"{self.llm_endpoint}/generate",
                    json={
                        "prompt": prompt, 
                        "max_new_tokens": 1024, 
                        "temperature": 0.3
                    },
                    timeout=timeout,
                )
                
                if response.status_code == 200:
                    return response.json().get("text", "")
                    
                # Handle various error conditions
                elif response.status_code == 503:
                    if "Model is still loading" in response.text:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Add jitter to avoid thundering herd problem
                            jitter = random.uniform(0, 0.1 * backoff)
                            sleep_time = backoff + jitter
                            
                            log_retry("Model still loading", retry_count, sleep_time)
                            time.sleep(sleep_time)
                            
                            # Exponential backoff with cap
                            backoff = min(backoff * 2, 60)  # Cap at 60 seconds
                        else:
                            print(f"Max retries ({max_retries}) reached waiting for LLM to load")
                            return ""
                    else:
                        # Other 503 errors
                        retry_count += 1
                        sleep_time = backoff + random.uniform(0, 0.1 * backoff)
                        log_retry(f"Service unavailable", retry_count, sleep_time)
                        time.sleep(sleep_time)
                        backoff = min(backoff * 2, 60)
                        
                elif response.status_code >= 500:
                    # Server errors
                    retry_count += 1
                    sleep_time = backoff + random.uniform(0, 0.1 * backoff)
                    log_retry(f"Server error {response.status_code}", retry_count, sleep_time)
                    time.sleep(sleep_time)
                    backoff = min(backoff * 2, 60)
                    
                else:
                    # Client errors (400s) or unexpected responses
                    print(f"LLM API error: {response.status_code} - {response.text}")
                    if retry_count < max_retries - 1:  # Try a few times even for client errors
                        retry_count += 1
                        sleep_time = backoff
                        log_retry(f"Client error {response.status_code}", retry_count, sleep_time)
                        time.sleep(sleep_time)
                        backoff = min(backoff * 1.5, 30)  # Slower backoff for client errors
                    else:
                        # Give up on client errors after some attempts
                        return ""
                    
            except requests.exceptions.Timeout:
                # Handle timeouts specially - the server might be overloaded
                retry_count += 1
                sleep_time = backoff * 2 + random.uniform(0, 0.1 * backoff)
                log_retry("Request timeout", retry_count, sleep_time)
                time.sleep(sleep_time)
                backoff = min(backoff * 2, 90)  # Longer backoff for timeouts
                
            except requests.exceptions.ConnectionError:
                # Connection errors could mean the server is down or restarting
                retry_count += 1
                sleep_time = backoff * 3 + random.uniform(0, 0.1 * backoff)
                log_retry("Connection error - server might be down", retry_count, sleep_time)
                time.sleep(sleep_time)
                backoff = min(backoff * 2, 120)  # Even longer backoff for connection errors
                
            except Exception as e:
                # Other unexpected errors
                print(f"Unexpected error calling LLM API: {e}")
                retry_count += 1
                sleep_time = backoff + random.uniform(0, backoff * 0.1)
                time.sleep(sleep_time)
                backoff *= 2
        
        # Return empty string after all retries failed
        print(f"All {max_retries} retries exhausted. Using fallback behavior.")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="AI-Enhanced Workflow Documentation Scraper"
    )
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--output", "-o", default="docs", help="Output directory")
    parser.add_argument(
        "--depth", "-d", type=int, default=3, help="Maximum crawl depth"
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
        help="Enable debug mode (save intermediate files)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=100, help="Maximum number of pages to scrape"
    )

    args = parser.parse_args()

    scraper = WorkflowScraper(
        base_url=args.url,
        output_dir=args.output,
        max_depth=args.depth,
        delay=args.delay,
        headless=not args.visible,
        llm_endpoint=args.llm,
        debug=args.debug,
        max_pages=args.max_pages,
    )

    try:
        scraper.scrape()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {e}")


if __name__ == "__main__":
    main()
