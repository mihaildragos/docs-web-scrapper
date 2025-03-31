#!/usr/bin/env python3
"""
Structure-Aware Documentation Scraper

An advanced documentation scraper that:
1. Discovers the site structure including hidden menus
2. Creates an index of all available documentation
3. Uses AI to intelligently extract content with proper context
4. Maintains the hierarchical structure of the documentation
"""

import os
import sys
import time
import json
import argparse
from urllib.parse import urljoin, urlparse, unquote
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import yaml
from collections import defaultdict

# Web scraping & browser automation
import requests
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import (
    sync_playwright,
    Page,
    Browser,
    TimeoutError as PlaywrightTimeoutError,
)
import html2text

# Visual AI
from paddleocr import PaddleOCR
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DocScraper")


# Define data structures
@dataclass
class NavItem:
    """Navigation item in the documentation structure"""

    title: str
    url: str
    level: int
    children: List["NavItem"] = None
    parent: Optional["NavItem"] = None
    extracted: bool = False
    selector_path: Optional[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: "NavItem") -> None:
        child.parent = self
        self.children.append(child)

    def get_full_path(self) -> List[str]:
        """Get the full hierarchical path to this item"""
        if self.parent is None:
            return [self.title]
        return self.parent.get_full_path() + [self.title]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "title": self.title,
            "url": self.url,
            "level": self.level,
            "extracted": self.extracted,
            "selector_path": self.selector_path,
            "children": [child.to_dict() for child in self.children],
        }


class StructureDiscoveryService:
    """Service for discovering the structure of a documentation site"""

    def __init__(self, page: Page, base_url: str, ocr: PaddleOCR, debug: bool = False):
        self.page = page
        self.base_url = base_url
        self.ocr = ocr
        self.debug = debug
        self.visited_urls = set()
        self.discovered_nav_items = {}  # url -> NavItem
        self.root = NavItem("Root", base_url, 0)
        self.discovered_nav_items[base_url] = self.root

    def discover_structure(self) -> NavItem:
        """Discover the entire site structure"""
        logger.info(f"Starting structure discovery from {self.base_url}")
        self._process_url(self.base_url, self.root)
        return self.root

    def _process_url(self, url: str, parent_item: NavItem) -> None:
        """Process a URL to extract its structure and navigation elements"""
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        logger.info(f"Discovering structure for: {url}")

        # Navigate to the page
        try:
            self.page.goto(url, wait_until="networkidle", timeout=30000)
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout while loading {url}, continuing with partial page")
            self.page.wait_for_load_state("domcontentloaded")

        # Take screenshot for OCR if in debug mode
        screenshot_path = None
        if self.debug:
            os.makedirs("debug", exist_ok=True)
            url_hash = hashlib.md5(url.encode()).hexdigest()
            screenshot_path = f"debug/{url_hash}.png"
            self.page.screenshot(path=screenshot_path, full_page=True)

        # 1. Expand hidden navigation elements
        self._expand_hidden_menus()

        # 2. Extract navigation structure using both HTML and OCR
        nav_elements = self._extract_navigation_elements()
        if screenshot_path:
            ocr_nav_elements = self._extract_navigation_from_ocr(screenshot_path)
            # Merge OCR and HTML navigation results
            nav_elements = self._merge_navigation_sources(
                nav_elements, ocr_nav_elements
            )

        # 3. Create NavItems and link them to parent
        for nav_element in nav_elements:
            nav_url = nav_element["url"]
            # Normalize and validate URL
            full_url = urljoin(url, nav_url)
            parsed = urlparse(full_url)

            # Skip external links, anchors and non-HTML resources
            if (
                parsed.netloc and parsed.netloc != urlparse(self.base_url).netloc
            ) or not self._is_valid_doc_url(full_url):
                continue

            # Create the NavItem if it doesn't exist
            if full_url not in self.discovered_nav_items:
                nav_item = NavItem(
                    title=nav_element["title"],
                    url=full_url,
                    level=parent_item.level + 1,
                )
                self.discovered_nav_items[full_url] = nav_item
                parent_item.add_child(nav_item)

                # Don't go too deep
                if nav_item.level <= 3:  # Limit hierarchy depth
                    self._process_url(full_url, nav_item)

    def _expand_hidden_menus(self) -> None:
        """Expand collapsible/hidden navigation menus"""
        # Common classes and patterns for expandable menus
        expand_selectors = [
            ".toggle-button",
            ".expand-button",
            ".menu-toggle",
            ".sidebar-toggle",
            "[aria-expanded='false']",
            ".nav-dropdown",
            ".dropdown-toggle",
            ".accordion-toggle",
            "button.collapsed",
            "[data-toggle='collapse']",
        ]

        # Try clicking each potential toggle element
        for selector in expand_selectors:
            try:
                elements = self.page.query_selector_all(selector)
                for element in elements:
                    try:
                        # Check if element is visible
                        if element.is_visible():
                            element.click()
                            # Brief pause to let animation complete
                            time.sleep(0.5)
                    except Exception as e:
                        continue
            except Exception:
                continue

        # Additional heuristic: click on elements that likely expand menus
        try:
            # Find elements with "+" or "▶" or "▼" text that might be expanders
            expander_elements = self.page.query_selector_all(
                "button, .expander, [class*='expand'], [class*='toggle'], [class*='collapse']"
            )

            for element in expander_elements:
                try:
                    if element.is_visible():
                        text = element.text_content()
                        if text and any(
                            char in text for char in ["+", ">", "▶", "▼", "..."]
                        ):
                            element.click()
                            time.sleep(0.5)
                except Exception:
                    continue
        except Exception:
            pass

    def _extract_navigation_elements(self) -> List[Dict]:
        """Extract navigation elements from the page using HTML structure"""
        nav_elements = []

        # Run JavaScript to extract all potential navigation links
        nav_data = self.page.evaluate(
            """
        () => {
            const navItems = [];
            
            // Look for navigation elements
            const navSelectors = [
                'nav a', '.nav a', '.navigation a', '.menu a', '.sidebar a',
                '[role="navigation"] a', '.docs-nav a', '.doc-nav a', 
                '.toc a', '.table-of-contents a', '.menu-item a',
                '[class*="menu"] a', '[class*="nav"] a', '[class*="sidebar"] a',
                '.documentation-links a'
            ];
            
            // Create a set to avoid duplicates
            const processedHrefs = new Set();
            
            navSelectors.forEach(selector => {
                try {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {
                        const href = el.getAttribute('href');
                        if (href && !processedHrefs.has(href)) {
                            processedHrefs.add(href);
                            
                            // Get computed styles to check visibility
                            const styles = window.getComputedStyle(el);
                            const isHidden = styles.display === 'none' || 
                                            styles.visibility === 'hidden' || 
                                            styles.opacity === '0';
                            
                            if (!isHidden) {
                                const rect = el.getBoundingClientRect();
                                navItems.push({
                                    title: el.textContent.trim(),
                                    url: href,
                                    selector: getFullSelector(el),
                                    position: {
                                        x: rect.left,
                                        y: rect.top,
                                        width: rect.width,
                                        height: rect.height
                                    }
                                });
                            }
                        }
                    });
                } catch (e) {
                    // Ignore errors for individual selectors
                }
            });
            
            // Helper function to get a unique selector for an element
            function getFullSelector(element) {
                if (!(element instanceof Element)) return '';
                let path = [];
                while (element.nodeType === Node.ELEMENT_NODE) {
                    let selector = element.nodeName.toLowerCase();
                    if (element.id) {
                        selector += '#' + element.id;
                        path.unshift(selector);
                        break;
                    } else {
                        let sibling = element;
                        let nth = 1;
                        while (sibling.previousElementSibling) {
                            sibling = sibling.previousElementSibling;
                            if (sibling.nodeName.toLowerCase() === selector) nth++;
                        }
                        if (nth !== 1) selector += `:nth-of-type(${nth})`;
                    }
                    path.unshift(selector);
                    element = element.parentNode;
                }
                return path.join(' > ');
            }
            
            return navItems;
        }
        """
        )

        return nav_data

    def _extract_navigation_from_ocr(self, screenshot_path: str) -> List[Dict]:
        """Extract navigation elements using OCR on the screenshot"""
        ocr_nav_elements = []

        try:
            # Run OCR on the screenshot
            ocr_results = self.ocr.ocr(screenshot_path)
            if not ocr_results or not ocr_results[0]:
                return []

            # Get all text blocks
            text_blocks = []
            for line in ocr_results[0]:
                if line:
                    position = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    # Skip low confidence text
                    if confidence < 0.7:
                        continue

                    # Skip very short text
                    if len(text) < 3:
                        continue

                    text_blocks.append(
                        {"text": text, "position": position, "confidence": confidence}
                    )

            # Find potential navigation items by looking at patterns in text
            # and positions (e.g., text lines grouped together could be a menu)

            # Group text blocks by vertical position (potential menu items are aligned)
            y_groups = defaultdict(list)
            for block in text_blocks:
                y_center = (block["position"][0][1] + block["position"][2][1]) / 2
                # Group with 20px tolerance
                y_group = round(y_center / 20) * 20
                y_groups[y_group].append(block)

            # Identify text blocks that likely represent navigation items
            for y, blocks in y_groups.items():
                if (
                    len(blocks) >= 2
                ):  # At least 2 items aligned horizontally might be a menu
                    for block in blocks:
                        text = block["text"]
                        # Skip common non-navigation text
                        if any(
                            word in text.lower()
                            for word in ["copyright", "terms", "policy", "©"]
                        ):
                            continue

                        # Heuristic: navigation items are often short, capitalized words or phrases
                        if len(text) <= 30 and (
                            text[0].isupper() or text.startswith(".")
                        ):
                            ocr_nav_elements.append(
                                {
                                    "title": text,
                                    "url": "#"
                                    + text.lower().replace(" ", "-"),  # placeholder URL
                                    "position": {
                                        "x": (
                                            block["position"][0][0]
                                            + block["position"][1][0]
                                        )
                                        / 2,
                                        "y": (
                                            block["position"][0][1]
                                            + block["position"][2][1]
                                        )
                                        / 2,
                                        "width": block["position"][1][0]
                                        - block["position"][0][0],
                                        "height": block["position"][2][1]
                                        - block["position"][0][1],
                                    },
                                    "ocr_detected": True,
                                }
                            )

        except Exception as e:
            logger.error(f"Error in OCR navigation extraction: {e}")

        return ocr_nav_elements

    def _merge_navigation_sources(
        self, html_nav: List[Dict], ocr_nav: List[Dict]
    ) -> List[Dict]:
        """Merge navigation elements found from HTML and OCR"""
        # Create a set of existing titles to avoid duplicates
        existing_titles = {item["title"].lower() for item in html_nav}

        # Add OCR items that don't overlap with HTML items
        merged_nav = html_nav.copy()

        for ocr_item in ocr_nav:
            # Check if this OCR item title is already in the HTML items
            if ocr_item["title"].lower() not in existing_titles:
                # Look for the closest matching link on the page
                closest_link = self._find_closest_link_for_ocr_text(
                    ocr_item["title"], ocr_item["position"]
                )
                if closest_link:
                    ocr_item["url"] = closest_link
                    merged_nav.append(ocr_item)
                    existing_titles.add(ocr_item["title"].lower())

        return merged_nav

    def _find_closest_link_for_ocr_text(
        self, text: str, position: Dict
    ) -> Optional[str]:
        """Find the closest link on the page that might match the OCR-detected text"""
        try:
            # Execute JavaScript to find links near the given position
            nearby_link = self.page.evaluate(
                """
            (text, position) => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                let closestLink = null;
                let minDistance = Infinity;
                
                // Calculate center point of the OCR text
                const ocrX = position.x;
                const ocrY = position.y;
                
                links.forEach(link => {
                    const rect = link.getBoundingClientRect();
                    const linkX = rect.left + rect.width / 2;
                    const linkY = rect.top + rect.height / 2;
                    
                    // Calculate Euclidean distance
                    const distance = Math.sqrt(
                        Math.pow(ocrX - linkX, 2) + 
                        Math.pow(ocrY - linkY, 2)
                    );
                    
                    // Check text similarity (either the link text or its title matches the OCR text)
                    const linkText = link.textContent.trim();
                    const linkTitle = link.getAttribute('title') || '';
                    const similarity = (linkText.includes(text) || 
                                        text.includes(linkText) || 
                                        linkTitle.includes(text) ||
                                        text.includes(linkTitle));
                    
                    // If text is similar and distance is within 150px, consider it
                    if (similarity && distance < 150 && distance < minDistance) {
                        minDistance = distance;
                        closestLink = link.getAttribute('href');
                    }
                });
                
                return closestLink;
            }
            """,
                text,
                position,
            )

            return nearby_link

        except Exception as e:
            logger.error(f"Error finding closest link: {e}")
            return None

    def _is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is likely a documentation page and not a resource or external link"""
        parsed = urlparse(url)

        # Skip URLs with fragments only (same page links)
        if (not parsed.path or parsed.path == "/") and parsed.fragment:
            return False

        # Skip common resource file extensions
        ignored_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",  # Images
            ".css",
            ".js",
            ".json",
            ".xml",  # Code/data files
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",  # Documents
            ".zip",
            ".tar",
            ".gz",
            ".rar",  # Archives
            ".mp4",
            ".webm",
            ".mp3",
            ".wav",
            ".ogg",  # Media
        ]

        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in ignored_extensions):
            return False

        # Allow HTML pages and paths without extensions (likely HTML)
        return True


class SelectorService:
    """Service for determining optimal content selectors using AI"""

    def __init__(self, llm_endpoint: str, debug: bool = False):
        self.llm_endpoint = llm_endpoint
        self.debug = debug
        self.selector_cache = {}  # Cache for previously determined selectors

    def get_content_selectors(
        self, soup: BeautifulSoup, url: str, visual_data: Dict = None
    ) -> Dict[str, str]:
        """Get the best CSS selectors for different content parts using AI"""
        # Check cache first
        cache_key = self._get_cache_key(url)
        if cache_key in self.selector_cache:
            return self.selector_cache[cache_key]

        # Extract simplified HTML structure
        html_structure = self._get_simplified_html(soup)

        # Prepare prompt for LLM
        prompt = self._create_selector_prompt(html_structure, url, visual_data)

        # Call LLM API
        selectors = self._call_llm_api(prompt)

        # Cache the result
        self.selector_cache[cache_key] = selectors
        return selectors

    def _get_cache_key(self, url: str) -> str:
        """Create a cache key for the URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_simplified_html(self, soup: BeautifulSoup) -> str:
        """Create a simplified representation of the HTML for the LLM"""
        structure = []

        def process_element(element, depth=0, max_elements=300):
            if len(structure) >= max_elements:
                return

            if not hasattr(element, "name") or not element.name:
                return

            # Skip script, style tags
            if element.name in ["script", "style", "svg", "path", "meta", "link"]:
                return

            # Get element attributes
            attrs = {}
            for attr in ["id", "class", "role", "aria-label"]:
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
                    if len(direct_text) < 100:
                        el_desc["text"] = direct_text[:100]
                    else:
                        el_desc["text_length"] = len(direct_text)

            structure.append(el_desc)

            # Process children
            for child in element.children:
                if hasattr(child, "name") and child.name:
                    process_element(child, depth + 1)

        if soup.body:
            process_element(soup.body)

        return json.dumps(structure, indent=2)

    def _create_selector_prompt(
        self, html_structure: str, url: str, visual_data: Dict = None
    ) -> str:
        """Create a prompt for the LLM to determine content selectors"""
        prompt = f"""
        You are an AI assistant specialized in web scraping and HTML analysis. 
        Analyze this HTML structure from a documentation page and identify the best CSS selectors for extracting:
        
        1. Main content: The primary documentation content (articles, tutorials, reference material)
        2. Title: The main title of the page
        3. Navigation: The navigation menu or sidebar
        4. Code blocks: Code examples and snippets
        5. Tables: Data tables if present
        6. Headings: Section headings in the main content
        
        URL: {url}
        
        HTML Structure (simplified):
        {html_structure[:5000]}  # Limit size to avoid token limits
        """

        if visual_data:
            prompt += f"""
            
            Visual elements detected:
            {json.dumps(visual_data)[:1000]}
            """

        prompt += """
        
        Return a JSON object with selectors for each element type. For each type, provide:
        1. A primary selector (most specific)
        2. Fallback selectors (more general)
        
        Example response format:
        {
          "main_content": {
            "primary": "article.documentation",
            "fallbacks": ["main", ".content", "#docs-content"]
          },
          "title": {
            "primary": "h1.doc-title",
            "fallbacks": ["article h1:first-child", ".content-header h1"]
          },
          "navigation": {
            "primary": "nav.sidebar",
            "fallbacks": [".docs-nav", "#toc"]
          },
          "code_blocks": {
            "primary": "pre code",
            "fallbacks": [".highlight", ".code-example"]
          },
          "tables": {
            "primary": "table.data-table",
            "fallbacks": ["article table", ".content table"]
          },
          "headings": {
            "primary": "article h2, article h3, article h4",
            "fallbacks": [".content h2, .content h3, .content h4"]
          }
        }
        
        Only return valid CSS selectors in the JSON format shown above.
        """

        return prompt

    def _call_llm_api(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """Call the DeepSeek LLM API to get selectors"""
        try:
            response = requests.post(
                f"{self.llm_endpoint}/generate",
                json={"prompt": prompt, "max_new_tokens": 1024, "temperature": 0.2},
                timeout=60,
            )

            if response.status_code == 200:
                llm_response = response.json().get("text", "")

                # Extract JSON from the response
                json_match = re.search(r"({[\s\S]*})", llm_response)
                if json_match:
                    try:
                        selectors = json.loads(json_match.group(1))
                        return selectors
                    except json.JSONDecodeError:
                        pass

            # Fallback selectors if API call fails or returns invalid JSON
            return {
                "main_content": {
                    "primary": "article, .content, main, #content",
                    "fallbacks": [
                        "div.documentation",
                        ".markdown-body",
                        "main",
                        ".main",
                    ],
                },
                "title": {
                    "primary": "h1",
                    "fallbacks": ["article h1", ".content h1", ".title"],
                },
                "navigation": {
                    "primary": "nav, .nav, .navigation, .menu, .sidebar",
                    "fallbacks": ["[role='navigation']", ".toc", "#toc"],
                },
                "code_blocks": {
                    "primary": "pre code",
                    "fallbacks": [".highlight", "pre", "code", ".code"],
                },
                "tables": {
                    "primary": "table",
                    "fallbacks": [".table", "article table"],
                },
                "headings": {
                    "primary": "h2, h3, h4, h5, h6",
                    "fallbacks": ["article h2, article h3", ".content h2, .content h3"],
                },
            }

        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            # Return fallback selectors
            return {
                "main_content": {
                    "primary": "article, .content, main",
                    "fallbacks": ["#content", "div.documentation", ".markdown-body"],
                },
                "title": {"primary": "h1", "fallbacks": ["article h1", ".title"]},
                "navigation": {
                    "primary": "nav, .navigation, .sidebar",
                    "fallbacks": [".menu", "#menu", ".toc"],
                },
                "code_blocks": {
                    "primary": "pre code",
                    "fallbacks": [".highlight", "code"],
                },
                "tables": {"primary": "table", "fallbacks": [".table"]},
                "headings": {
                    "primary": "h2, h3, h4",
                    "fallbacks": ["article h2, article h3"],
                },
            }


class AdvancedDocumentationScraper:
    """Advanced documentation scraper with structure discovery and AI extraction"""

    def __init__(
        self,
        base_url: str,
        output_dir: str = "docs",
        max_depth: int = 3,
        delay: float = 1.0,
        headless: bool = True,
        llm_endpoint: str = "http://deepseek:8000",
        debug: bool = False,
        max_workers: int = 4,
        session_file: str = None,
    ):
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.delay = delay
        self.headless = headless
        self.llm_endpoint = llm_endpoint
        self.debug = debug
        self.max_workers = max_workers
        self.session_file = session_file
        self.site_structure = None
        self.visited_urls = set()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

        # Initialize browserself.playwright = sync_playwright().start()
        if self.session_file and os.path.exists(self.session_file):
            logger.info(f"Using saved session from: {self.session_file}")
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            
            # Load the stored session state
            with open(self.session_file, 'r') as f:
                storage_state = json.load(f)
            
            # Create a persistent context with the session
            self.persistent_context = self.browser.new_context(storage_state=storage_state)
        else:
            logger.info("Starting with a fresh browser session")
            self.browser = self.playwright.chromium.launch(headless=self.headless)
        

        # Initialize OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        # Initialize services
        self.selector_service = SelectorService(llm_endpoint, debug)

        # HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.body_width = 0
        self.html2text.unicode_snob = True
        self.html2text.mark_code = True

    def __del__(self):
        try:
            self.browser.close()
            self.playwright.stop()
        except:
            pass

    def scrape(self):
        """Start the scraping process with structure discovery first"""
        logger.info(f"Starting the advanced scraping process for {self.base_url}")
        
        # First phase: Discover site structure
        try:
            # Use the persistent context if available, otherwise create a new one
            if hasattr(self, 'persistent_context'):
                context = self.persistent_context
            else:
                context = self.browser.new_context(
                    viewport={"width": 1280, "height": 1024}
                )
            
            discovery_page = context.new_page()

            # Initialize structure discovery service
            structure_service = StructureDiscoveryService(
                discovery_page, self.base_url, self.ocr, self.debug
            )

            # Discover the site structure
            self.site_structure = structure_service.discover_structure()

            # Save the structure as JSON for reference
            self._save_site_structure()

            # Create index markdown file
            self._create_index_file()

            # Second phase: Extract content for each page in the structure
            self._extract_content_from_structure()

        finally:
            try:
                context.close()
            except:
                pass

        logger.info(f"Scraping complete. Results saved to {self.output_dir}/")

    def _save_site_structure(self):
        """Save the discovered site structure to a JSON file"""
        structure_path = os.path.join(self.output_dir, "site_structure.json")
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(self.site_structure.to_dict(), f, indent=2)

        logger.info(f"Site structure saved to {structure_path}")

    def _create_index_file(self):
        """Create an index markdown file with the site structure"""
        index_path = os.path.join(self.output_dir, "index.md")

        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"# Documentation Index\n\n")
            f.write(f"Source: {self.base_url}\n\n")

            f.write("## Table of Contents\n\n")

            def write_toc_item(item, level=0):
                if level > 0:  # Skip the root node
                    # Create file-safe name
                    path = self._url_to_path(item.url)
                    file_path = f"{path}.md"

                    # Write the item with proper indentation
                    f.write(f"{'  ' * (level-1)}* [{item.title}]({file_path})\n")

                # Write children
                for child in item.children:
                    write_toc_item(child, level + 1)

            write_toc_item(self.site_structure)

        logger.info(f"Index file created at {index_path}")

    def _extract_content_from_structure(self):
        """Extract content for each page in the site structure"""
        # Create a list of all pages to extract
        pages_to_extract = []

        def collect_pages(item):
            if item.level > 0:  # Skip root
                pages_to_extract.append(item)
            for child in item.children:
                collect_pages(child)

        collect_pages(self.site_structure)

        # Extract content with a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for page_item in pages_to_extract:
                futures.append(executor.submit(self._extract_page_content, page_item))

            # Wait for all extractions to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in page extraction: {e}")

    def _extract_page_content(self, nav_item: NavItem):
        """Extract content from a specific page"""
        if nav_item.url in self.visited_urls:
            return

        self.visited_urls.add(nav_item.url)
        logger.info(f"Extracting content from: {nav_item.url}")

        # Be polite - add delay between requests
        time.sleep(self.delay)

        try:
            # Create a new context and page for each extraction
            context = self.browser.new_context(viewport={"width": 1280, "height": 1024})
            page = context.new_page()

            # Navigate to the page
            try:
                page.goto(nav_item.url, wait_until="networkidle", timeout=30000)
            except PlaywrightTimeoutError:
                logger.warning(
                    f"Timeout while loading {nav_item.url}, continuing with partial page"
                )
                page.wait_for_load_state("domcontentloaded")

            # Take a screenshot for OCR if in debug mode
            screenshot_path = None
            if self.debug:
                url_hash = hashlib.md5(nav_item.url.encode()).hexdigest()
                screenshot_path = os.path.join(
                    self.output_dir, "debug", f"{url_hash}.png"
                )
                page.screenshot(path=screenshot_path, full_page=True)

            # Get the HTML content
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")

            # Get visual data from OCR if available
            visual_data = None
            if screenshot_path and os.path.exists(screenshot_path):
                visual_data = self._extract_visual_data(screenshot_path)

            # Get content selectors from the selector service
            selectors = self.selector_service.get_content_selectors(
                soup, nav_item.url, visual_data
            )

            # Extract the main content
            main_content = self._extract_content_with_selectors(soup, selectors)

            # Convert to markdown
            markdown_content = self._convert_to_markdown(main_content, nav_item)

            # Save to file
            self._save_page_content(nav_item, markdown_content)

            # Mark as extracted
            nav_item.extracted = True

        except Exception as e:
            logger.error(f"Error extracting content from {nav_item.url}: {e}")

        finally:
            context.close()

    def _extract_visual_data(self, screenshot_path: str) -> Dict:
        """Extract visual data using OCR"""
        try:
            ocr_results = self.ocr.ocr(screenshot_path)
            if not ocr_results or not ocr_results[0]:
                return {"text_blocks": []}

            text_blocks = []
            for idx, line in enumerate(ocr_results[0]):
                if line:
                    position = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    text_blocks.append(
                        {
                            "id": idx,
                            "position": position,
                            "text": text,
                            "confidence": confidence,
                        }
                    )

            return {"text_blocks": text_blocks}

        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return {"text_blocks": []}

    def _extract_content_with_selectors(
        self, soup: BeautifulSoup, selectors: Dict
    ) -> str:
        """Extract content using the provided selectors"""
        # Extract main content
        main_content = None

        # Try primary selector first
        main_selector = selectors.get("main_content", {}).get("primary")
        if main_selector:
            main_elements = soup.select(main_selector)
            if main_elements:
                main_content = "\n".join(str(el) for el in main_elements)

        # If primary fails, try fallbacks
        if not main_content or len(main_content) < 100:  # Too short, probably failed
            fallback_selectors = selectors.get("main_content", {}).get("fallbacks", [])
            for selector in fallback_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = "\n".join(str(el) for el in elements)
                    if len(main_content) > 100:  # Found substantial content
                        break

        # If all selectors fail, use the body
        if not main_content or len(main_content) < 100:
            # Remove obvious non-content elements before using body
            for selector in [
                "nav",
                "header",
                "footer",
                ".sidebar",
                ".menu",
                ".navigation",
                "script",
                "style",
                "meta",
                "svg",
                "path",
            ]:
                for el in soup.select(selector):
                    el.decompose()

            if soup.body:
                main_content = str(soup.body)
            else:
                main_content = str(soup)

        return main_content

    def _convert_to_markdown(self, html_content: str, nav_item: NavItem) -> str:
        """Convert HTML content to markdown"""
        # Convert to markdown
        markdown = self.html2text.handle(html_content)

        # Remove excess newlines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        # Add title and source
        title = nav_item.title
        full_path = " > ".join(nav_item.get_full_path()[1:])  # Skip root

        header = f"# {title}\n\n"

        if full_path:
            header += f"**Path:** {full_path}\n\n"

        header += f"**Source:** [{nav_item.url}]({nav_item.url})\n\n"

        return header + markdown

    def _save_page_content(self, nav_item: NavItem, content: str):
        """Save page content to a markdown file"""
        # Create a file path from the URL
        file_path = self._url_to_path(nav_item.url)
        full_path = os.path.join(self.output_dir, f"{file_path}.md")

        # Create directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write the file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved {nav_item.url} to {full_path}")

    def _url_to_path(self, url: str) -> str:
        """Convert a URL to a valid file path"""
        parsed = urlparse(url)

        # Remove the base URL part to get a relative path
        base_parsed = urlparse(self.base_url)

        # Clean the path
        path = parsed.path.strip("/")

        # Handle empty path
        if not path:
            return "index"

        # Handle URL parameters
        if parsed.query:
            path += f"_{hashlib.md5(parsed.query.encode()).hexdigest()[:6]}"

        # Clean path and create appropriate filename
        clean_path = re.sub(r"[^a-zA-Z0-9/]", "_", path)

        # Handle trailing slash
        if path.endswith("/"):
            clean_path += "index"

        return clean_path


def main():
    parser = argparse.ArgumentParser(description='Advanced Structure-Aware Documentation Scraper')
    parser.add_argument('url', help='URL to scrape')
    parser.add_argument('--output', '-o', default='docs', help='Output directory')
    parser.add_argument('--depth', '-d', type=int, default=3, help='Maximum crawl depth')
    parser.add_argument('--delay', '-w', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--llm', default=os.getenv('LLM_ENDPOINT', 'http://deepseek:8000'), help='DeepSeek LLM endpoint')
    parser.add_argument('--visible', action='store_true', help='Run browser in visible mode (not headless)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (save screenshots, etc.)')
    parser.add_argument('--workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--session', help='Path to a saved browser session file')
    
    args = parser.parse_args()
    
    scraper = AdvancedDocumentationScraper(
        base_url=args.url,
        output_dir=args.output,
        max_depth=args.depth,
        delay=args.delay,
        headless=not args.visible,
        llm_endpoint=args.llm,
        debug=args.debug,
        max_workers=args.workers,
        session_file=args.session
    )

    try:
        scraper.scrape()
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
    except Exception as e:
        print(f"Error during scraping: {e}")


if __name__ == "__main__":
    main()
