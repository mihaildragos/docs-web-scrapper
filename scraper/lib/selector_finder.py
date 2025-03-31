#!/usr/bin/env python3
"""
Selector Finder

Uses DeepSeek LLM to analyze page structure and find optimal CSS selectors
for extracting main content and other key components.
"""

import os
import json
import re
from typing import Dict, List, Any
from bs4 import BeautifulSoup
from urllib.parse import urlparse


class SelectorFinder:
    def __init__(
        self,
        llm_endpoint: str = "http://deepseek:8000",
        debug: bool = False,
        output_dir: str = "docs",
    ):
        self.llm_endpoint = llm_endpoint
        self.debug = debug
        self.output_dir = output_dir

        # Create debug directory if needed
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def find_content_selectors(
        self, url: str, html_content: str, visual_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find optimal selectors for content extraction using DeepSeek LLM

        Args:
            url: URL of the page
            html_content: HTML content of the page
            visual_analysis: Visual analysis from VisualAnalyzer

        Returns:
            Dictionary with selectors for different page components
        """
        print(f"Finding content selectors for {url}")

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Get simplified HTML structure
        simplified_html = self._get_simplified_html(soup)

        # Create a prompt for the LLM
        prompt = self._create_selector_prompt(url, simplified_html, visual_analysis)

        # Get selectors from LLM
        selectors = self._get_selectors_from_llm(prompt)

        # Validate selectors against the HTML
        validated_selectors = self._validate_selectors(soup, selectors)

        # Save selectors for debugging
        if self.debug:
            with open(
                os.path.join(
                    self.output_dir,
                    "debug",
                    f"{self._url_to_filename(url)}_selectors.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "proposed_selectors": selectors,
                        "validated_selectors": validated_selectors,
                    },
                    f,
                    indent=2,
                )

        return validated_selectors

    def _get_simplified_html(self, soup: BeautifulSoup) -> str:
        """Create a simplified representation of the HTML structure"""
        structure = []

        def process_element(element, depth=0):
            if not hasattr(element, "name") or not element.name:
                return

            # Skip script and style elements
            if element.name in ["script", "style", "noscript"]:
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

        # Start processing from body
        body = soup.body
        if body:
            process_element(body)

        return json.dumps(structure[:300], indent=2)  # Limit to avoid token limits

    def _create_selector_prompt(
        self, url: str, simplified_html: str, visual_analysis: Dict[str, Any]
    ) -> str:
        """Create a prompt for the LLM to identify content selectors"""
        # Extract relevant information from visual analysis
        visual_sections = {}
        if "visual_sections" in visual_analysis:
            visual_sections = visual_analysis["visual_sections"]

        # Create the prompt
        prompt = f"""
        You are an AI assistant helping with web scraping. Analyze this HTML structure and identify the CSS selectors
        that would extract different components of the documentation page. The goal is to scrape the documentation
        content effectively.
        
        URL: {url}
        
        HTML Structure:
        {simplified_html}
        
        Visual Analysis Information:
        {json.dumps(visual_sections, indent=2) if visual_sections else "No visual analysis available"}
        
        Please provide CSS selectors for the following components:
        1. Main Content: The primary documentation text (avoid navigation, headers, footers)
        2. Page Title: The title of the current documentation page
        3. Navigation: The documentation navigation/table of contents
        4. Code Blocks: Any code examples or snippets
        5. Tables: Data tables in the documentation
        6. Figures/Images: Important visual elements
        7. Notes/Admonitions: Warning boxes, info boxes, etc.
        
        For each component:
        - Provide at least 2-3 alternative selectors in case the primary one doesn't work
        - Order the selectors by specificity and likelihood of success
        - Explain your reasoning for each selector
        
        Return your answer as a JSON object with the following format:
        {{
            "main_content": [
                {{ "selector": "article.content", "explanation": "Primary content container" }},
                {{ "selector": "main .documentation", "explanation": "Backup selector for main content" }}
            ],
            "page_title": [...],
            "navigation": [...],
            "code_blocks": [...],
            "tables": [...],
            "figures": [...],
            "notes": [...]
        }}
        """

        return prompt

    def _get_selectors_from_llm(self, prompt: str) -> Dict[str, List[Dict[str, str]]]:
        """Get selectors from DeepSeek LLM"""
        # Use the new retry function to get the response text
        response_text = self.call_llm_with_retry(prompt)
        
        # If we got a response, try to extract JSON
        if response_text:
            try:
                # Extract JSON from the response
                json_match = re.search(
                    r"```json\s*({.*?})\s*```", response_text, re.DOTALL
                )

                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without markdown code blocks
                    json_match = re.search(r"({.*})", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response_text

                # Try to parse the JSON
                return json.loads(json_str)
                
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response")
        
        # Return default selectors if the response was empty or JSON parsing failed
        return self._get_default_selectors()
    

    def _get_default_selectors(self) -> Dict[str, List[Dict[str, str]]]:
        """Get default selectors if LLM API fails"""
        return {
            "main_content": [
                {"selector": "main", "explanation": "Main content element"},
                {"selector": "article", "explanation": "Article element"},
                {
                    "selector": ".content, .documentation",
                    "explanation": "Common content class",
                },
                {"selector": "#main-content", "explanation": "Common main content ID"},
            ],
            "page_title": [
                {"selector": "h1", "explanation": "Primary heading"},
                {"selector": ".page-title", "explanation": "Page title class"},
                {
                    "selector": "article h1, main h1",
                    "explanation": "Heading in content area",
                },
            ],
            "navigation": [
                {"selector": "nav", "explanation": "Navigation element"},
                {
                    "selector": ".sidebar, .toc",
                    "explanation": "Sidebar or table of contents",
                },
                {"selector": ".navigation, .menu", "explanation": "Navigation menu"},
            ],
            "code_blocks": [
                {"selector": "pre, code", "explanation": "Code elements"},
                {
                    "selector": ".code, .highlight",
                    "explanation": "Code highlighting classes",
                },
                {
                    "selector": ".syntax, .language-*",
                    "explanation": "Syntax highlighting classes",
                },
            ],
            "tables": [
                {"selector": "table", "explanation": "Table element"},
                {"selector": ".table", "explanation": "Table class"},
            ],
            "figures": [
                {"selector": "figure, img", "explanation": "Figure or image elements"},
                {
                    "selector": ".figure, .image",
                    "explanation": "Figure or image classes",
                },
            ],
            "notes": [
                {"selector": ".note, .warning, .info", "explanation": "Note classes"},
                {"selector": ".admonition", "explanation": "Admonition class"},
                {
                    "selector": ".alert, .callout",
                    "explanation": "Alert or callout classes",
                },
            ],
        }

    def _validate_selectors(
        self, soup: BeautifulSoup, selectors: Dict[str, List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Validate selectors against the HTML and select the best ones"""
        validated = {}

        for component, selector_list in selectors.items():
            for selector_info in selector_list:
                selector = selector_info["selector"]

                try:
                    elements = soup.select(selector)

                    if elements:
                        # Store the first working selector and matched elements count
                        validated[component] = {
                            "selector": selector,
                            "explanation": selector_info["explanation"],
                            "elements_count": len(elements),
                            "text_length": sum(
                                len(el.get_text(strip=True)) for el in elements
                            ),
                        }

                        # For main content, check if it has substantial text
                        if (
                            component == "main_content"
                            and validated[component]["text_length"] < 200
                        ):
                            # Too little text, try next selector
                            continue
                        else:
                            # Found a good selector, move to next component
                            break

                except Exception as e:
                    print(f"Error validating selector '{selector}': {e}")

            # If no selector worked for this component, use a default
            if component not in validated:
                validated[component] = {
                    "selector": self._get_default_selector_for_component(component),
                    "explanation": "Default fallback selector",
                    "elements_count": 0,
                    "text_length": 0,
                }

        return validated

    def _get_default_selector_for_component(self, component: str) -> str:
        """Get a default selector for a component if all others fail"""
        defaults = {
            "main_content": "body",
            "page_title": "h1",
            "navigation": "nav",
            "code_blocks": "pre, code",
            "tables": "table",
            "figures": "img",
            "notes": ".note, .warning, .info",
        }

        return defaults.get(component, "body")

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
