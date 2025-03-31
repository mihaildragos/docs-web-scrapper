#!/usr/bin/env python3
"""
Advanced Documentation Scraper with AI Integration

Features:
- Handles SPAs with JavaScript rendering
- Uses PaddleOCR for visual content recognition
- Integrates with DeepSeek LLM for intelligent content extraction
- Combines HTML structure and visual analysis
"""

import os
import time
import json
import argparse
from urllib.parse import urljoin, urlparse
import re
import numpy as np
from typing import Dict, Any

# Web scraping & browser automation
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page
import html2text

# Visual AI
from paddleocr import PaddleOCR


class AIEnhancedScraper:
    def __init__(
        self,
        base_url: str,
        output_dir: str = "docs",
        max_depth: int = 1,
        delay: float = 1.0,
        headless: bool = True,
        llm_endpoint: str = "http://deepseek:8000",  # Updated to use service name in Docker network
        debug: bool = False,
    ):
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.delay = delay
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.debug = debug

        self.visited_urls = set()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

        # Initialize browser
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)

        # Initialize OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.body_width = 0

    def __del__(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except:
            pass

    def scrape(self):
        """Start the scraping process"""
        print(f"Starting to scrape {self.base_url}")

        try:
            context = self.browser.new_context(viewport={"width": 1280, "height": 1024})
            page = context.new_page()
            self._scrape_page(page, self.base_url, 0)
        finally:
            context.close()

        print(
            f"Scraping complete. {len(self.visited_urls)} pages scraped to {self.output_dir}/"
        )

    def _scrape_page(self, page: Page, url: str, depth: int):
        """Scrape a single page and its links up to max_depth"""
        if url in self.visited_urls or depth > self.max_depth:
            return

        print(f"Scraping: {url}")
        self.visited_urls.add(url)

        # Be polite - add a delay between requests
        time.sleep(self.delay)

        try:
            # Navigate to the page and wait for it to load completely
            page.goto(url, wait_until="networkidle")

            # Extract page title
            title = page.title()

            # Get the HTML content after JavaScript execution
            html_content = page.content()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Take a screenshot for visual analysis
            screenshot_path = (
                os.path.join(
                    self.output_dir, "debug", f"{self._url_to_filename(url)}.png"
                )
                if self.debug
                else None
            )
            if self.debug:
                page.screenshot(path=screenshot_path, full_page=True)

            # Extract text from the page using OCR if needed
            visual_content = {}
            if self.debug and screenshot_path and os.path.exists(screenshot_path):
                visual_content = self._extract_visual_content(screenshot_path)

            # Use LLM to identify important content sections
            main_content = self._extract_content_with_ai(soup, visual_content, url)

            # Convert to markdown
            markdown = self.html2text.handle(str(main_content))

            # Remove excess newlines
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)

            # Save to file
            self._save_markdown(url, title, markdown)

            # Scrape linked pages if not at max depth
            if depth < self.max_depth:
                self._scrape_links(page, soup, url, depth)

        except Exception as e:
            print(f"Error processing {url}: {e}")

    def _extract_visual_content(self, screenshot_path: str) -> Dict[str, Any]:
        """Extract visual content using PaddleOCR"""
        if not os.path.exists(screenshot_path):
            return {}

        results = self.ocr.ocr(screenshot_path)
        if results is None or len(results) == 0:
            return {"text_blocks": []}

        visual_data = {"text_blocks": [], "tables": []}

        # Process OCR results
        for idx, line in enumerate(results[0]):
            if line is None:
                continue

            position = line[0]
            text = line[1][0]
            confidence = line[1][1]

            visual_data["text_blocks"].append(
                {
                    "id": idx,
                    "position": position,
                    "text": text,
                    "confidence": confidence,
                }
            )

        return visual_data

    def _extract_content_with_ai(
        self, soup: BeautifulSoup, visual_content: Dict[str, Any], url: str
    ) -> str:
        """Use LLM to identify and extract relevant content"""
        # First attempt with common selectors
        for selector in [
            "main",
            "article",
            "div.content",
            "div.documentation",
            ".markdown-body",
        ]:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 100:
                return content

        # If no standard selectors worked, use the LLM for content identification
        try:
            # Prepare input for the LLM
            # Get the simplified HTML structure
            html_structure = self._get_simplified_html(soup)

            # Create a prompt for the LLM
            prompt = f"""
            You are an AI assistant helping with web scraping. Analyze this HTML structure and identify the main content
            elements that contain documentation. Return a list of CSS selectors that would extract the main content.
            Avoid selecting navigation, headers, footers, and sidebars.
            
            URL: {url}
            
            HTML Structure:
            {html_structure[:2000]}  # Limiting size to avoid token limits
            
            Visual elements detected:
            {json.dumps(visual_content)[:1000] if visual_content else "No visual data available"}
            
            Return only a JSON array of CSS selectors, ordered by priority.
            """

            # Get selectors from LLM
            response = self._call_llm_api(prompt)

            try:
                selectors = json.loads(response)
                if not isinstance(selectors, list):
                    selectors = [selectors]
            except:
                # Fallback if response isn't valid JSON
                selectors = [
                    s.strip()
                    for s in response.split("\n")
                    if s.strip() and not s.startswith("#")
                ]

            # Try each selector
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # Combine all matching elements
                    content_html = "".join(str(el) for el in elements)
                    if len(content_html) > 100:  # Ensure we got substantial content
                        return content_html

        except Exception as e:
            print(f"Error using LLM for content extraction: {e}")

        # Fallback: remove obvious non-content elements
        for nav in soup.select("nav, header, footer, .sidebar, .menu, .navigation"):
            nav.decompose()

        return str(soup.body)

    def _call_llm_api(self, prompt: str) -> str:
        """Call the DeepSeek LLM API"""
        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={"prompt": prompt, "max_new_tokens": 1024, "temperature": 0.3},
                timeout=60,
            )

            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print(f"LLM API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return ""

    def _get_simplified_html(self, soup: BeautifulSoup) -> str:
        """Create a simplified version of the HTML for LLM analysis"""
        structure = []

        def process_element(element, depth=0):
            if not hasattr(element, "name") or not element.name:
                return

            # Get element attributes
            attrs = {}
            for attr in ["id", "class", "role"]:
                if element.has_attr(attr):
                    attrs[attr] = element[attr]

            # Create element description
            el_desc = {"tag": element.name, "depth": depth, "attrs": attrs}

            # Add text length if element has direct text
            text = element.get_text(strip=True)
            if text:
                # Only add text info if it's direct text and not all from children
                direct_text = "".join(
                    t.strip() for t in element.find_all(text=True, recursive=False)
                )
                if direct_text:
                    el_desc["text_length"] = len(direct_text)
                    if len(direct_text) < 50:
                        el_desc["text"] = direct_text

            structure.append(el_desc)

            # Process children
            for child in element.children:
                if hasattr(child, "name") and child.name:
                    process_element(child, depth + 1)

        process_element(soup.body)
        return json.dumps(structure, indent=2)

    def _scrape_links(
        self, page: Page, soup: BeautifulSoup, current_url: str, depth: int
    ):
        """Find and scrape links on the page"""
        links = page.evaluate(
            """
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(href => !href.startsWith('#'));
                return Array.from(new Set(links));  // Remove duplicates
            }
        """
        )

        for href in links:
            # Handle relative URLs
            full_url = urljoin(current_url, href)

            # Make sure URL is from the same domain
            if not self._is_same_domain(full_url):
                continue

            # Scrape the linked page
            self._scrape_page(page, full_url, depth + 1)

    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is from the same domain as base_url"""
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        return parsed_base.netloc == parsed_url.netloc

    def _url_to_filename(self, url: str) -> str:
        """Convert a URL to a valid filename"""
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "index"

        # Clean path and create appropriate filename
        clean_path = re.sub(r"[^a-zA-Z0-9/]", "_", path)
        if path.endswith("/"):
            clean_path += "index"

        return clean_path

    def _save_markdown(self, url: str, title: str, content: str):
        """Save markdown content to file"""
        clean_path = self._url_to_filename(url)
        filename = f"{clean_path}.md"

        # Create subdirectories if needed
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"Source: {url}\n\n")
            f.write(content)


def main():
    parser = argparse.ArgumentParser(description="AI-Enhanced Documentation Scraper")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--output", "-o", default="docs", help="Output directory")
    parser.add_argument(
        "--depth", "-d", type=int, default=1, help="Maximum crawl depth"
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
        help="Enable debug mode (save screenshots, etc.)",
    )

    args = parser.parse_args()

    scraper = AIEnhancedScraper(
        base_url=args.url,
        output_dir=args.output,
        max_depth=args.depth,
        delay=args.delay,
        headless=not args.visible,
        llm_endpoint=args.llm,
        debug=args.debug,
    )

    try:
        scraper.scrape()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {e}")


if __name__ == "__main__":
    main()
