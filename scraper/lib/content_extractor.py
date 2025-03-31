#!/usr/bin/env python3
"""
Content Extractor

Uses BeautifulSoup with selectors from SelectorFinder to extract
structured content from a webpage.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse, urljoin
import html2text


class ContentExtractor:
    def __init__(self, base_url: str, debug: bool = False, output_dir: str = "docs"):
        self.base_url = base_url
        self.debug = debug
        self.output_dir = output_dir

        # HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.body_width = 0
        self.html2text.protect_links = True

        # Create debug directory if needed
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def extract_content(
        self, url: str, html_content: str, selectors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract structured content from HTML using the provided selectors

        Args:
            url: URL of the page
            html_content: HTML content of the page
            selectors: Validated selectors from SelectorFinder

        Returns:
            Dictionary with extracted content
        """
        print(f"Extracting content from {url}")

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract content using selectors
        extracted_content = {
            "url": url,
            "title": self._extract_title(soup, selectors),
            "main_content": self._extract_main_content(soup, selectors),
            "navigation": self._extract_navigation(soup, selectors),
            "code_blocks": self._extract_code_blocks(soup, selectors),
            "tables": self._extract_tables(soup, selectors),
            "figures": self._extract_figures(soup, selectors, url),
            "notes": self._extract_notes(soup, selectors),
            "metadata": self._extract_metadata(soup, url),
        }

        # Convert content to markdown for easier reading
        markdown_content = {
            "url": url,
            "title": extracted_content["title"],
            "main_content_markdown": self._convert_to_markdown(
                extracted_content["main_content"]["html"]
            ),
            "navigation_markdown": self._convert_to_markdown(
                extracted_content["navigation"]["html"]
            ),
            "metadata": extracted_content["metadata"],
        }

        # Save extracted content for debugging
        if self.debug:
            filename = self._url_to_filename(url)
            with open(
                os.path.join(self.output_dir, "debug", f"{filename}_extracted.json"),
                "w",
            ) as f:
                json.dump(extracted_content, f, indent=2)

            with open(
                os.path.join(self.output_dir, "debug", f"{filename}_markdown.json"), "w"
            ) as f:
                json.dump(markdown_content, f, indent=2)

        return extracted_content

    def _extract_title(self, soup: BeautifulSoup, selectors: Dict[str, Any]) -> str:
        """Extract the page title"""
        # Try page_title selector if available
        if "page_title" in selectors:
            selector = selectors["page_title"]["selector"]
            elements = soup.select(selector)

            if elements:
                return elements[0].get_text(strip=True)

        # Fallback to document title
        title_tag = soup.title
        if title_tag:
            return title_tag.get_text(strip=True)

        # Fallback to first h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        # Default to URL if no title found
        parsed_url = urlparse(soup.get("url", ""))
        return (
            parsed_url.path.strip("/").replace("-", " ").replace("_", " ").title()
            or "Untitled Page"
        )

    def _extract_main_content(
        self, soup: BeautifulSoup, selectors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract the main content"""
        result = {"html": "", "text": "", "word_count": 0, "headings": []}

        if "main_content" in selectors:
            selector = selectors["main_content"]["selector"]
            elements = soup.select(selector)

            if elements:
                # Combine all matching elements
                content_html = "".join(str(el) for el in elements)
                content_text = " ".join(el.get_text(strip=True) for el in elements)

                result["html"] = content_html
                result["text"] = content_text
                result["word_count"] = len(content_text.split())

                # Extract headings for structure
                for element in elements:
                    headings = element.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                    for heading in headings:
                        result["headings"].append(
                            {
                                "level": int(heading.name[1]),
                                "text": heading.get_text(strip=True),
                                "id": heading.get("id", ""),
                            }
                        )

        # If no content found, try body as fallback
        if not result["html"]:
            body = soup.body
            if body:
                result["html"] = str(body)
                result["text"] = body.get_text(strip=True)
                result["word_count"] = len(result["text"].split())

        return result

    def _extract_navigation(
        self, soup: BeautifulSoup, selectors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract navigation links"""
        result = {"html": "", "links": []}

        if "navigation" in selectors:
            selector = selectors["navigation"]["selector"]
            elements = soup.select(selector)

            if elements:
                # Combine all matching elements
                result["html"] = "".join(str(el) for el in elements)

                # Extract links
                for element in elements:
                    links = element.find_all("a")
                    for link in links:
                        href = link.get("href", "")
                        if href and not href.startswith("#"):
                            # Make relative URLs absolute
                            abs_url = urljoin(self.base_url, href)

                            result["links"].append(
                                {
                                    "text": link.get_text(strip=True),
                                    "url": abs_url,
                                    "is_active": "active" in link.get("class", [])
                                    or False,
                                    "is_external": not abs_url.startswith(
                                        self.base_url
                                    ),
                                }
                            )

        return result

    def _extract_code_blocks(
        self, soup: BeautifulSoup, selectors: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract code blocks"""
        code_blocks = []

        if "code_blocks" in selectors:
            selector = selectors["code_blocks"]["selector"]
            elements = soup.select(selector)

            for element in elements:
                # Skip inline code that's part of a larger pre element
                if element.name == "code" and element.parent.name == "pre":
                    continue

                # Get language from class
                language = None
                for cls in element.get("class", []):
                    if cls.startswith("language-"):
                        language = cls.split("-")[1]
                        break

                code_blocks.append(
                    {
                        "code": element.get_text(strip=True),
                        "language": language,
                        "html": str(element),
                    }
                )

        return code_blocks

    def _extract_tables(
        self, soup: BeautifulSoup, selectors: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract tables"""
        tables = []

        if "tables" in selectors:
            selector = selectors["tables"]["selector"]
            elements = soup.select(selector)

            for element in elements:
                table_data = {"headers": [], "rows": [], "html": str(element)}

                # Extract headers
                thead = element.find("thead")
                if thead:
                    header_cells = thead.find_all("th")
                    if header_cells:
                        table_data["headers"] = [
                            cell.get_text(strip=True) for cell in header_cells
                        ]

                # If no headers in thead, check the first row
                if not table_data["headers"]:
                    first_row = element.find("tr")
                    if first_row:
                        header_cells = first_row.find_all("th")
                        if header_cells:
                            table_data["headers"] = [
                                cell.get_text(strip=True) for cell in header_cells
                            ]

                # Extract rows
                rows = element.find_all("tr")
                for row in rows:
                    # Skip if this row was used for headers
                    if row == element.find("tr") and table_data["headers"]:
                        continue

                    cells = row.find_all(["td", "th"])
                    if cells:
                        table_data["rows"].append(
                            [cell.get_text(strip=True) for cell in cells]
                        )

                tables.append(table_data)

        return tables

    def _extract_figures(
        self, soup: BeautifulSoup, selectors: Dict[str, Any], url: str
    ) -> List[Dict[str, Any]]:
        """Extract figures and images"""
        figures = []

        if "figures" in selectors:
            selector = selectors["figures"]["selector"]
            elements = soup.select(selector)

            for element in elements:
                figure_data = {"html": str(element), "caption": "", "src": ""}

                # Extract caption
                figcaption = element.find("figcaption")
                if figcaption:
                    figure_data["caption"] = figcaption.get_text(strip=True)

                # Extract image source
                img = element.find("img") if element.name != "img" else element
                if img:
                    src = img.get("src", "")
                    if src:
                        # Make relative URLs absolute
                        figure_data["src"] = urljoin(url, src)
                        figure_data["alt"] = img.get("alt", "")

                figures.append(figure_data)

        return figures

    def _extract_notes(
        self, soup: BeautifulSoup, selectors: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract notes, warnings, and admonitions"""
        notes = []

        if "notes" in selectors:
            selector = selectors["notes"]["selector"]
            elements = soup.select(selector)

            for element in elements:
                note_type = "note"  # Default type

                # Try to determine note type from class
                for cls in element.get("class", []):
                    if cls in ["warning", "caution", "danger", "error"]:
                        note_type = "warning"
                    elif cls in ["tip", "hint", "information", "info"]:
                        note_type = "info"
                    elif cls in ["success", "check", "done"]:
                        note_type = "success"

                notes.append(
                    {
                        "type": note_type,
                        "text": element.get_text(strip=True),
                        "html": str(element),
                    }
                )

        return notes

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from the page"""
        metadata = {
            "url": url,
            "canonical_url": url,
            "description": "",
            "keywords": [],
            "author": "",
            "published_date": "",
            "modified_date": "",
            "og_title": "",
            "og_description": "",
            "og_image": "",
        }

        # Extract canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            metadata["canonical_url"] = canonical["href"]

        # Extract description
        description = soup.find("meta", attrs={"name": "description"})
        if description and description.get("content"):
            metadata["description"] = description["content"]

        # Extract keywords
        keywords = soup.find("meta", attrs={"name": "keywords"})
        if keywords and keywords.get("content"):
            metadata["keywords"] = [k.strip() for k in keywords["content"].split(",")]

        # Extract author
        author = soup.find("meta", attrs={"name": "author"})
        if author and author.get("content"):
            metadata["author"] = author["content"]

        # Extract dates
        published = soup.find("meta", attrs={"property": "article:published_time"})
        if published and published.get("content"):
            metadata["published_date"] = published["content"]

        modified = soup.find("meta", attrs={"property": "article:modified_time"})
        if modified and modified.get("content"):
            metadata["modified_date"] = modified["content"]

        # Extract Open Graph metadata
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            metadata["og_title"] = og_title["content"]

        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            metadata["og_description"] = og_desc["content"]

        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_image and og_image.get("content"):
            metadata["og_image"] = og_image["content"]

        return metadata

    def _convert_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown"""
        if not html:
            return ""

        try:
            markdown = self.html2text.handle(html)

            # Clean up the markdown
            import re

            # Remove excess newlines
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)

            # Fix code blocks
            markdown = re.sub(r"```\s+```", "```\n```", markdown)

            return markdown
        except Exception as e:
            print(f"Error converting HTML to Markdown: {e}")
            return html

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
