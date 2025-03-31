#!/usr/bin/env python3
"""
Visual Analyzer

Analyzes webpage content visually using PaddleOCR to recognize text
and structural elements on the page.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from paddleocr import PaddleOCR
from playwright.sync_api import Page
from urllib.parse import urlparse


class VisualAnalyzer:
    def __init__(self, debug: bool = False, output_dir: str = "docs"):
        self.debug = debug
        self.output_dir = output_dir

        # Initialize OCR engine
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # Create debug directory if needed
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def analyze_page_structure(self, page: Page, url: str) -> Dict[str, Any]:
        """
        Analyze the visual structure of a page

        Args:
            page: Playwright page object
            url: URL of the page

        Returns:
            Dictionary with visual structure analysis
        """
        print(f"Analyzing visual structure of {url}")

        # Take screenshot for visual analysis
        filename = self._url_to_filename(url)
        screenshot_path = os.path.join(
            self.output_dir, "debug", f"{filename}_visual.png"
        )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)

        # Take screenshot
        page.screenshot(path=screenshot_path, full_page=True)

        # Extract text using OCR
        visual_content = self._extract_text_from_image(screenshot_path)

        # Identify visual sections (header, main content, sidebar, footer)
        visual_sections = self._identify_visual_sections(page, visual_content)

        # Enhance with element coordinates
        visual_sections = self._enhance_with_element_coords(page, visual_sections)

        # Combine results
        analysis = {
            "url": url,
            "visual_content": visual_content,
            "visual_sections": visual_sections,
        }

        # Save analysis for debugging
        if self.debug:
            with open(
                os.path.join(
                    self.output_dir, "debug", f"{filename}_visual_analysis.json"
                ),
                "w",
            ) as f:
                json.dump(analysis, f, indent=2)

        return analysis

    def _extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using PaddleOCR"""
        results = self.ocr.ocr(image_path)

        if results is None or len(results) == 0:
            return {"text_blocks": []}

        text_blocks = []
        tables = []

        # Process OCR results
        for idx, line in enumerate(results[0]):
            if line is None:
                continue

            position = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text = line[1][0]  # Text content
            confidence = line[1][1]  # Confidence score

            # Calculate bounding box properties
            top_left = position[0]
            bottom_right = position[2]

            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]

            text_blocks.append(
                {
                    "id": idx,
                    "text": text,
                    "confidence": confidence,
                    "position": position,
                    "bounding_box": {
                        "x": top_left[0],
                        "y": top_left[1],
                        "width": width,
                        "height": height,
                    },
                }
            )

        # Detect tables in the image (simple heuristic)
        tables = self._detect_tables(text_blocks)

        return {"text_blocks": text_blocks, "tables": tables}

    def _detect_tables(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential tables in the text blocks"""
        tables = []

        # Group text blocks by Y coordinate proximity to detect rows
        y_sorted_blocks = sorted(text_blocks, key=lambda b: b["bounding_box"]["y"])

        # Simple row detection
        rows = []
        current_row = []
        last_y = 0

        for block in y_sorted_blocks:
            y = block["bounding_box"]["y"]

            # If this block is significantly below the last one, it's a new row
            if current_row and (y - last_y > block["bounding_box"]["height"] * 1.5):
                rows.append(current_row)
                current_row = []

            current_row.append(block)
            last_y = y

        # Add the last row
        if current_row:
            rows.append(current_row)

        # Detect tables by looking for multiple rows with similar structure
        if len(rows) >= 3:
            # Check if blocks in each row have similar x-coordinates
            columns = self._detect_columns(rows)

            if len(columns) >= 2:
                # This looks like a table!
                table = {
                    "rows": len(rows),
                    "columns": len(columns),
                    "position": {
                        "x": min(
                            block["bounding_box"]["x"] for row in rows for block in row
                        ),
                        "y": min(
                            block["bounding_box"]["y"] for row in rows for block in row
                        ),
                        "width": max(
                            block["bounding_box"]["x"] + block["bounding_box"]["width"]
                            for row in rows
                            for block in row
                        )
                        - min(
                            block["bounding_box"]["x"] for row in rows for block in row
                        ),
                        "height": max(
                            block["bounding_box"]["y"] + block["bounding_box"]["height"]
                            for row in rows
                            for block in row
                        )
                        - min(
                            block["bounding_box"]["y"] for row in rows for block in row
                        ),
                    },
                    "cells": [],
                }

                # Extract cell data
                for row_idx, row in enumerate(rows):
                    for block in row:
                        # Determine which column this block belongs to
                        col_idx = self._get_column_index(block, columns)

                        if col_idx is not None:
                            table["cells"].append(
                                {
                                    "row": row_idx,
                                    "column": col_idx,
                                    "text": block["text"],
                                    "position": block["bounding_box"],
                                }
                            )

                tables.append(table)

        return tables

    def _detect_columns(
        self, rows: List[List[Dict[str, Any]]]
    ) -> List[Tuple[float, float]]:
        """Detect columns in rows of text blocks"""
        # Collect all x-coordinates
        x_coords = []

        for row in rows:
            for block in row:
                x_coords.append(block["bounding_box"]["x"])

        # Cluster x-coordinates to find column positions
        if not x_coords:
            return []

        # Simple clustering
        x_coords.sort()
        columns = []
        current_column = [x_coords[0]]

        for x in x_coords[1:]:
            # If this x-coordinate is close to the last one, it's in the same column
            if x - current_column[-1] < 50:  # Adjust this threshold as needed
                current_column.append(x)
            else:
                # New column
                if current_column:
                    columns.append((min(current_column), max(current_column)))
                current_column = [x]

        # Add the last column
        if current_column:
            columns.append((min(current_column), max(current_column)))

        return columns

    def _get_column_index(
        self, block: Dict[str, Any], columns: List[Tuple[float, float]]
    ) -> Optional[int]:
        """Determine which column a block belongs to"""
        x = block["bounding_box"]["x"]

        for i, (min_x, max_x) in enumerate(columns):
            if min_x <= x <= max_x:
                return i

        return None

    def _identify_visual_sections(
        self, page: Page, visual_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify visual sections of the page using visual cues"""
        # Use JavaScript to get information about key visual sections
        sections_info = page.evaluate(
            """
        () => {
            const viewportHeight = window.innerHeight;
            const documentHeight = document.documentElement.scrollHeight;
            
            // Helper function to get element info
            const getElementInfo = (el) => {
                if (!el) return null;
                
                const rect = el.getBoundingClientRect();
                return {
                    tag: el.tagName.toLowerCase(),
                    id: el.id || null,
                    classes: Array.from(el.classList),
                    text: el.innerText.slice(0, 100),
                    position: {
                        x: rect.left,
                        y: rect.top,
                        width: rect.width,
                        height: rect.height
                    }
                };
            };
            
            // Find header
            let headerElement = document.querySelector('header, .header, #header');
            if (!headerElement) {
                // Try to find header by position
                const possibleHeaders = Array.from(document.querySelectorAll('*'))
                    .filter(el => {
                        const rect = el.getBoundingClientRect();
                        return rect.top < 150 && rect.height > 30 && rect.height < 200;
                    });
                
                if (possibleHeaders.length > 0) {
                    headerElement = possibleHeaders[0];
                }
            }
            
            // Find main content
            let mainElement = document.querySelector('main, [role="main"], #main, .content, .documentation, article');
            
            // Find sidebar
            let sidebarElement = document.querySelector('.sidebar, #sidebar, aside, .toc, .table-of-contents');
            
            // Find footer
            let footerElement = document.querySelector('footer, .footer, #footer');
            if (!footerElement) {
                // Try to find footer by position
                const possibleFooters = Array.from(document.querySelectorAll('*'))
                    .filter(el => {
                        const rect = el.getBoundingClientRect();
                        return rect.bottom > documentHeight - 200 && rect.height > 30 && rect.height < 300;
                    });
                
                if (possibleFooters.length > 0) {
                    footerElement = possibleFooters[possibleFooters.length - 1];
                }
            }
            
            return {
                header: getElementInfo(headerElement),
                main: getElementInfo(mainElement),
                sidebar: getElementInfo(sidebarElement),
                footer: getElementInfo(footerElement)
            };
        }
        """
        )

        # Combine with OCR results
        # For each section, find text blocks that are within the section's boundaries
        if visual_content and "text_blocks" in visual_content:
            for section_key, section_info in sections_info.items():
                if section_info and "position" in section_info:
                    section_info["text_blocks"] = []

                    for block in visual_content["text_blocks"]:
                        block_x = block["bounding_box"]["x"]
                        block_y = block["bounding_box"]["y"]

                        section_x = section_info["position"]["x"]
                        section_y = section_info["position"]["y"]
                        section_width = section_info["position"]["width"]
                        section_height = section_info["position"]["height"]

                        # Check if the block is within the section
                        if (
                            section_x <= block_x <= section_x + section_width
                            and section_y <= block_y <= section_y + section_height
                        ):
                            section_info["text_blocks"].append(block)

        return sections_info

    def _enhance_with_element_coords(
        self, page: Page, visual_sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance visual sections with coordinates of important elements"""
        # Use JavaScript to get coordinates of important elements
        important_elements = page.evaluate(
            """
        () => {
            // Elements we want to analyze
            const selectors = [
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'nav', 'div.navigation', '.menu',
                'table', 'ul', 'ol', '.code', 'pre',
                'img', 'figure', '.card', '.panel'
            ];
            
            const results = {};
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    results[selector] = Array.from(elements).map(el => {
                        const rect = el.getBoundingClientRect();
                        return {
                            text: el.innerText.slice(0, 100),
                            html: el.outerHTML.slice(0, 500),
                            position: {
                                x: rect.left,
                                y: window.scrollY + rect.top, // Adjust for scrolling
                                width: rect.width,
                                height: rect.height
                            }
                        };
                    });
                }
            });
            
            return results;
        }
        """
        )

        visual_sections["important_elements"] = important_elements
        return visual_sections

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
