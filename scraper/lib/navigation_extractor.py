#!/usr/bin/env python3
"""
Navigation Extractor

Extracts the navigation structure from a webpage using visual analysis
and HTML structure evaluation.
"""

import os
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
from playwright.sync_api import Page
from bs4 import BeautifulSoup
import json


class NavigationExtractor:
    def __init__(self, debug: bool = False, output_dir: str = "docs"):
        self.debug = debug
        self.output_dir = output_dir

        # Create debug directory if needed
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def extract_navigation(self, page: Page, url: str) -> Dict[str, Any]:
        """
        Extract the navigation structure from a webpage

        Args:
            page: Playwright page object
            url: URL of the page

        Returns:
            Dictionary with navigation structure
        """
        print(f"Extracting navigation structure from {url}")

        # Take screenshot for visual analysis if debug mode is enabled
        screenshot_path = None
        if self.debug:
            filename = self._url_to_filename(url) + "_nav.png"
            screenshot_path = os.path.join(self.output_dir, "debug", filename)
            page.screenshot(path=screenshot_path, full_page=True)

        # Extract HTML content
        html_content = page.content()
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract navigation using common patterns
        nav_elements = self._extract_nav_elements(soup)

        # Extract sitemap if available
        sitemap_url = self._find_sitemap_url(soup, url)
        sitemap_links = self._process_sitemap(sitemap_url) if sitemap_url else []

        # Use JavaScript to extract links from interactive menus
        js_nav_links = self._extract_js_nav_links(page)

        # Combine all navigation sources
        navigation = {
            "nav_elements": nav_elements,
            "sitemap_links": sitemap_links,
            "js_nav_links": js_nav_links,
            "base_url": url,
        }

        # Save navigation structure for debugging
        if self.debug:
            with open(
                os.path.join(
                    self.output_dir, "debug", f"{self._url_to_filename(url)}_nav.json"
                ),
                "w",
            ) as f:
                json.dump(navigation, f, indent=2)

        return navigation

    def _extract_nav_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract navigation elements from the HTML"""
        nav_elements = []

        # Find primary navigation elements
        nav_selectors = [
            "nav",
            ".nav",
            ".navigation",
            ".menu",
            ".sidebar",
            "[role=navigation]",
            "header .links",
            ".docs-navigation",
            ".table-of-contents",
            ".toc",
        ]

        for selector in nav_selectors:
            for element in soup.select(selector):
                links = element.find_all("a")
                if len(links) < 2:  # Skip elements with too few links
                    continue

                nav_item = {
                    "element_type": element.name,
                    "element_classes": element.get("class", []),
                    "element_id": element.get("id", ""),
                    "links": [],
                }

                # Process links
                for link in links:
                    link_text = link.get_text(strip=True)
                    link_href = link.get("href", "")

                    if link_text and link_href and not link_href.startswith("#"):
                        nav_item["links"].append(
                            {
                                "text": link_text,
                                "href": link_href,
                                "depth": self._estimate_depth(link),
                                "is_active": "active" in link.get("class", []) or False,
                            }
                        )

                if nav_item["links"]:
                    nav_elements.append(nav_item)

        return nav_elements

    def _find_sitemap_url(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Find sitemap URL if available"""
        # Check for sitemap in HTML
        sitemap_link = soup.find(
            "a", href=lambda href: href and "sitemap" in href.lower()
        )
        if sitemap_link and sitemap_link.get("href"):
            return urljoin(base_url, sitemap_link["href"])

        # Check for sitemap.xml in robots.txt
        try:
            robots_url = urljoin(base_url, "/robots.txt")
            robots_text = requests.get(robots_url, timeout=10).text

            for line in robots_text.split("\n"):
                if "sitemap:" in line.lower():
                    return line.split(":", 1)[1].strip()
        except:
            pass

        # Try common sitemap locations
        common_paths = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap.html"]
        for path in common_paths:
            sitemap_url = urljoin(base_url, path)
            try:
                response = requests.head(sitemap_url, timeout=5)
                if response.status_code == 200:
                    return sitemap_url
            except:
                continue

        return None

    def _process_sitemap(self, sitemap_url: str) -> List[Dict[str, str]]:
        """Process sitemap.xml or sitemap.html to extract links"""
        try:
            response = requests.get(sitemap_url, timeout=10)
            content_type = response.headers.get("Content-Type", "")

            if "xml" in content_type:
                # Process XML sitemap
                soup = BeautifulSoup(response.text, "xml")
                urls = soup.find_all("url")

                sitemap_links = []
                for url in urls:
                    loc = url.find("loc")
                    if loc and loc.text:
                        sitemap_links.append(
                            {
                                "url": loc.text,
                                "priority": (
                                    url.find("priority").text
                                    if url.find("priority")
                                    else None
                                ),
                                "changefreq": (
                                    url.find("changefreq").text
                                    if url.find("changefreq")
                                    else None
                                ),
                            }
                        )
                return sitemap_links
            else:
                # Process HTML sitemap
                soup = BeautifulSoup(response.text, "html.parser")
                links = soup.find_all("a")

                sitemap_links = []
                for link in links:
                    href = link.get("href")
                    text = link.get_text(strip=True)

                    if href and text and not href.startswith("#"):
                        sitemap_links.append(
                            {"url": urljoin(sitemap_url, href), "text": text}
                        )
                return sitemap_links
        except Exception as e:
            print(f"Error processing sitemap {sitemap_url}: {e}")
            return []

    def _extract_js_nav_links(self, page: Page) -> List[Dict[str, str]]:
        """Extract navigation links from JavaScript-powered menus"""
        # Execute JavaScript to find and click navigation toggles
        nav_links = page.evaluate(
            """
        () => {
            const results = [];
            
            // Find navigation toggles
            const toggles = Array.from(document.querySelectorAll('.nav-toggle, .menu-toggle, [aria-expanded="false"]'));
            
            // Click each toggle to reveal hidden navigation items
            toggles.forEach(toggle => {
                try {
                    toggle.click();
                } catch (e) {
                    // Ignore click errors
                }
            });
            
            // Wait a moment for animations/transitions
            setTimeout(() => {
                // Now collect all visible navigation links
                const navLinks = Array.from(document.querySelectorAll('nav a, .navigation a, .menu a, .sidebar a, .toc a'));
                
                navLinks.forEach(link => {
                    if (link.href && !link.href.startsWith('#') && link.innerText.trim()) {
                        results.push({
                            url: link.href,
                            text: link.innerText.trim(),
                            parent_menu: link.closest('ul, nav').className || null
                        });
                    }
                });
            }, 500);
            
            return results;
        }
        """
        )

        return nav_links

    def _estimate_depth(self, element) -> int:
        """Estimate the navigation depth of an element based on CSS classes and position"""
        depth = 0

        # Check for common depth indicators in class names
        classes = element.get("class", [])
        for cls in classes:
            if "level-" in cls:
                try:
                    depth = int(cls.split("level-")[1])
                    return depth
                except:
                    pass

            if cls in ["submenu", "dropdown", "child", "sub"]:
                depth += 1

        # Check for nesting level
        parent = element.parent
        while parent:
            if parent.name == "ul" or parent.name == "ol":
                depth += 1
            parent = parent.parent

        return depth

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
