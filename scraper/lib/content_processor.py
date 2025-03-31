#!/usr/bin/env python3
"""
Content Processor

Uses DeepSeek LLM to process and restructure the extracted content
into a coherent, comprehensive document.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import time


class ContentProcessor:
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

    def process_content(
        self, url: str, extracted_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and restructure extracted content using DeepSeek LLM

        Args:
            url: URL of the page
            extracted_content: Content extracted by ContentExtractor

        Returns:
            Dictionary with processed content
        """
        print(f"Processing content for {url}")

        # Prepare content for processing
        title = extracted_content.get("title", "")
        main_content = extracted_content.get("main_content", {})
        navigation = extracted_content.get("navigation", {})
        code_blocks = extracted_content.get("code_blocks", [])
        tables = extracted_content.get("tables", [])
        figures = extracted_content.get("figures", [])
        notes = extracted_content.get("notes", [])
        metadata = extracted_content.get("metadata", {})

        # Create a prompt for the LLM
        prompt = self._create_processing_prompt(
            url,
            title,
            main_content,
            navigation,
            code_blocks,
            tables,
            figures,
            notes,
            metadata,
        )

        # Get processed content from LLM
        processed_markdown = self._get_processed_content_from_llm(prompt)

        # Extract document structure
        document_structure = self._extract_document_structure(processed_markdown)

        # Create processed content object
        processed_content = {
            "url": url,
            "title": title,
            "processed_markdown": processed_markdown,
            "document_structure": document_structure,
            "metadata": metadata,
        }

        # Save processed content for debugging
        if self.debug:
            with open(
                os.path.join(
                    self.output_dir,
                    "debug",
                    f"{self._url_to_filename(url)}_processed.json",
                ),
                "w",
            ) as f:
                json.dump(processed_content, f, indent=2)

            with open(
                os.path.join(
                    self.output_dir,
                    "debug",
                    f"{self._url_to_filename(url)}_processed.md",
                ),
                "w",
            ) as f:
                f.write(processed_markdown)

        return processed_content

    def _create_processing_prompt(
        self,
        url: str,
        title: str,
        main_content: Dict[str, Any],
        navigation: Dict[str, Any],
        code_blocks: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        figures: List[Dict[str, Any]],
        notes: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        """Create a prompt for the LLM to process and restructure content"""
        # Prepare content sections
        main_content_text = main_content.get("text", "")[
            :5000
        ]  # Limit length to avoid token limits

        # Create list of headings
        headings = []
        for heading in main_content.get("headings", []):
            headings.append(f"{'#' * heading['level']} {heading['text']}")

        headings_text = "\n".join(headings)

        # Create list of navigation links
        nav_links = []
        for link in navigation.get("links", []):
            nav_links.append(f"- {link['text']} -> {link['url']}")

        nav_links_text = "\n".join(nav_links)

        # Create list of code blocks
        code_blocks_text = ""
        for i, block in enumerate(code_blocks[:3]):  # Limit to avoid token limits
            language = block.get("language", "")
            code = block.get("code", "")[:200]  # Limit length
            code_blocks_text += (
                f"Code Block {i+1} ({language}):\n```{language}\n{code}\n```\n\n"
            )

        # Create list of tables
        tables_text = ""
        for i, table in enumerate(tables[:2]):  # Limit to avoid token limits
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            if headers:
                tables_text += f"Table {i+1}:\nHeaders: {headers}\n"

            tables_text += f"Rows ({len(rows)}):\n"
            for j, row in enumerate(rows[:3]):  # Limit rows
                tables_text += f"  Row {j+1}: {row}\n"

            tables_text += "\n"

        # Create the prompt
        prompt = f"""
        You are an AI assistant helping to process and restructure documentation content.
        Your task is to organize the extracted content into a coherent, well-structured document.
        
        URL: {url}
        Title: {title}
        
        Content Overview:
        -----------------
        
        Headings Structure:
        {headings_text}
        
        Navigation Links:
        {nav_links_text}
        
        Code Blocks:
        {code_blocks_text}
        
        Tables:
        {tables_text}
        
        Main Content:
        {main_content_text}
        
        Instructions:
        1. Organize the content into a logical structure with proper headings
        2. Ensure code blocks are properly formatted with language indicators
        3. Tables should be represented in markdown format
        4. Include important information from notes/admonitions
        5. Preserve links but make them relative when they point to other documentation pages
        6. Remove any duplicate content
        7. Fix any formatting issues
        
        Output the restructured content in clean Markdown format, preserving the original information
        but improving organization and readability.
        """

        return prompt

    def _get_processed_content_from_llm(self, prompt: str) -> str:
        """Get processed content from DeepSeek LLM"""
        # Use the retry function but customize for content processing (larger token limit)
        response_text = self.call_llm_with_retry(
            prompt=prompt,
            max_retries=3,  # Same as your original function
            initial_backoff=2  # Same as your original sleep time
        )
        
        # If we got a response, return it
        if response_text:
            return response_text
        
        # Return error message if all retries fail
        return "Failed to process content with LLM"

    def _extract_document_structure(self, markdown: str) -> Dict[str, Any]:
        """Extract document structure from processed markdown"""
        structure = {
            "headings": [],
            "sections": [],
            "code_blocks": [],
            "tables": [],
            "links": [],
        }

        # Extract headings
        import re

        heading_pattern = r"^(#{1,6})\s+(.+?)$"
        for match in re.finditer(heading_pattern, markdown, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()

            structure["headings"].append(
                {"level": level, "text": text, "position": match.start()}
            )

        # Extract code blocks
        code_block_pattern = r"```(\w*)\n(.*?)```"
        for match in re.finditer(code_block_pattern, markdown, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)

            structure["code_blocks"].append(
                {"language": language, "code": code, "position": match.start()}
            )

        # Extract tables (simplified)
        table_start_pattern = r"\n(\|[^\n]+\|)\n(\|[\s\-:]+\|)"
        for match in re.finditer(table_start_pattern, markdown):
            structure["tables"].append(
                {"position": match.start(), "header": match.group(1)}
            )

        # Extract links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        for match in re.finditer(link_pattern, markdown):
            text = match.group(1)
            url = match.group(2)

            structure["links"].append(
                {"text": text, "url": url, "position": match.start()}
            )

        # Create sections based on headings
        if structure["headings"]:
            for i, heading in enumerate(structure["headings"]):
                # Find the end of this section (next heading or end of document)
                section_end = len(markdown)
                if i < len(structure["headings"]) - 1:
                    section_end = structure["headings"][i + 1]["position"]

                # Extract section content
                section_start = heading["position"] + len(
                    f"{'#' * heading['level']} {heading['text']}"
                )
                section_content = markdown[section_start:section_end].strip()

                structure["sections"].append(
                    {
                        "heading": heading,
                        "content": section_content,
                        "length": len(section_content),
                    }
                )

        return structure

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
