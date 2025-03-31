#!/usr/bin/env python3
"""
Traversal Planner

Determines the logical order for scraping pages using DeepSeek LLM
to analyze content relationships and navigation structure.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlparse, urljoin
import time


class TraversalPlanner:
    def __init__(
        self,
        base_url: str,
        llm_endpoint: str = "http://deepseek:8000",
        debug: bool = False,
        output_dir: str = "docs",
    ):
        self.base_url = base_url
        self.llm_endpoint = llm_endpoint
        self.debug = debug
        self.output_dir = output_dir

        # Store visited URLs to avoid duplicates
        self.visited_urls = set()

        # Store URL relationships
        self.url_relationships = {}

        # Create debug directory if needed
        if self.debug:
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def plan_traversal(
        self,
        current_url: str,
        extracted_content: Dict[str, Any],
        processed_content: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Plan the next pages to scrape based on content analysis

        Args:
            current_url: Current page URL
            extracted_content: Content extracted by ContentExtractor
            processed_content: Content processed by ContentProcessor

        Returns:
            List of next URLs to scrape with priorities
        """
        print(f"Planning traversal from {current_url}")

        # Add current URL to visited set
        self.visited_urls.add(current_url)

        # Extract navigation links
        navigation_links = []
        if (
            "navigation" in extracted_content
            and "links" in extracted_content["navigation"]
        ):
            navigation_links = extracted_content["navigation"]["links"]

        # Extract links from main content
        content_links = self._extract_links_from_content(processed_content)

        # Combine all links
        all_links = self._combine_links(navigation_links, content_links)

        # Filter links to same domain and not already visited
        filtered_links = self._filter_links(all_links)

        # Use LLM to prioritize links
        prioritized_links = self._prioritize_links_with_llm(
            current_url, filtered_links, processed_content
        )

        # Store URL relationships for debugging
        self.url_relationships[current_url] = {
            "next_urls": [link["url"] for link in prioritized_links],
            "title": extracted_content.get("title", ""),
        }

        # Save traversal plan for debugging
        if self.debug:
            with open(
                os.path.join(self.output_dir, "debug", "traversal_plan.json"), "w"
            ) as f:
                json.dump(
                    {
                        "current_url": current_url,
                        "visited_urls": list(self.visited_urls),
                        "relationships": self.url_relationships,
                    },
                    f,
                    indent=2,
                )

        return prioritized_links

    def _extract_links_from_content(
        self, processed_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract links from processed content"""
        content_links = []

        # Extract links from document structure
        if (
            "document_structure" in processed_content
            and "links" in processed_content["document_structure"]
        ):
            for link in processed_content["document_structure"]["links"]:
                if "url" in link and "text" in link:
                    # Make relative URLs absolute
                    url = urljoin(
                        processed_content.get("url", self.base_url), link["url"]
                    )

                    content_links.append(
                        {"url": url, "text": link["text"], "source": "content"}
                    )

        return content_links

    def _combine_links(
        self,
        navigation_links: List[Dict[str, Any]],
        content_links: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Combine links from navigation and content"""
        # Create a set of URLs to avoid duplicates
        seen_urls = set()
        combined_links = []

        # Add navigation links first (they're usually more important)
        for link in navigation_links:
            if link["url"] not in seen_urls:
                combined_links.append(
                    {
                        "url": link["url"],
                        "text": link["text"],
                        "source": "navigation",
                        "is_active": link.get("is_active", False),
                        "is_external": link.get("is_external", False),
                    }
                )
                seen_urls.add(link["url"])

        # Add content links
        for link in content_links:
            if link["url"] not in seen_urls:
                combined_links.append(link)
                seen_urls.add(link["url"])

        return combined_links

    def _filter_links(self, links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter links to same domain and not already visited"""
        filtered_links = []

        for link in links:
            url = link["url"]

            # Skip if already visited
            if url in self.visited_urls:
                continue

            # Parse URL
            parsed_url = urlparse(url)
            parsed_base = urlparse(self.base_url)

            # Skip if different domain
            if parsed_url.netloc != parsed_base.netloc:
                continue

            # Skip common non-content URLs (login, search, etc.)
            skip_patterns = [
                "login",
                "sign-in",
                "search",
                "contact",
                "feed",
                "rss",
                "download",
                "register",
                "signup",
                "account",
            ]

            skip = False
            for pattern in skip_patterns:
                if pattern in parsed_url.path.lower():
                    skip = True
                    break

            if skip:
                continue

            # Skip URLs with unusual fragments or query params (often not content)
            if parsed_url.fragment or len(parsed_url.query) > 20:
                continue

            # Add to filtered links
            filtered_links.append(link)

        return filtered_links

    def _prioritize_links_with_llm(
        self,
        current_url: str,
        links: List[Dict[str, Any]],
        processed_content: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Use LLM to prioritize links based on content and structure"""
        # If no links, return empty list
        if not links:
            return []

        # If there are too many links, use heuristics first to reduce the number
        if len(links) > 20:
            links = self._heuristic_prioritization(links)[:20]

        # Create a prompt for the LLM
        prompt = self._create_prioritization_prompt(
            current_url, links, processed_content
        )

        # Get prioritized links from LLM
        prioritized_links = self._get_prioritized_links_from_llm(prompt, links)

        return prioritized_links

    def _heuristic_prioritization(
        self, links: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use heuristics to prioritize links"""
        # Score each link
        scored_links = []

        for link in links:
            score = 0

            # Navigation links are usually more important
            if link.get("source") == "navigation":
                score += 50

            # Active links in navigation are likely the current section
            if link.get("is_active", False):
                score += 30

            # Links with longer text are often more informative
            text_length = len(link.get("text", ""))
            score += min(text_length, 50)

            # Prefer shorter URLs (often higher in hierarchy)
            url_length = len(link.get("url", ""))
            score -= min(url_length // 10, 20)

            # Prefer URLs with simpler paths (less segments)
            path_segments = urlparse(link.get("url", "")).path.strip("/").count("/") + 1
            score -= path_segments * 5

            # Common documentation sections get higher scores
            doc_patterns = [
                "introduction",
                "getting-started",
                "guide",
                "tutorial",
                "overview",
                "reference",
                "api",
                "examples",
                "usage",
            ]

            for pattern in doc_patterns:
                if (
                    pattern in link.get("url", "").lower()
                    or pattern in link.get("text", "").lower()
                ):
                    score += 20
                    break

            scored_links.append({**link, "score": score})

        # Sort by score (descending)
        return sorted(scored_links, key=lambda x: x.get("score", 0), reverse=True)

    def _create_prioritization_prompt(
        self,
        current_url: str,
        links: List[Dict[str, Any]],
        processed_content: Dict[str, Any],
    ) -> str:
        """Create a prompt for the LLM to prioritize links"""
        # Extract relevant information from processed content
        title = processed_content.get("title", "")

        # Get headings from document structure
        headings = []
        if (
            "document_structure" in processed_content
            and "headings" in processed_content["document_structure"]
        ):
            for heading in processed_content["document_structure"]["headings"]:
                headings.append(
                    f"{'#' * heading.get('level', 1)} {heading.get('text', '')}"
                )

        headings_text = "\n".join(headings)

        # Create list of links
        links_text = ""
        for i, link in enumerate(links):
            links_text += f"{i+1}. {link.get('text', '')} -> {link.get('url', '')}\n"
            links_text += f"   Source: {link.get('source', 'unknown')}"
            if link.get("is_active", False):
                links_text += ", Active in navigation"
            links_text += "\n"

        # Create the prompt
        prompt = f"""
        You are an AI assistant helping to determine the logical order for scraping documentation pages.
        Your task is to prioritize the next pages to visit based on the current page content and available links.
        
        Current Page: {current_url}
        Title: {title}
        
        Page Structure:
        {headings_text}
        
        Available Links:
        {links_text}
        
        Instructions:
        1. Analyze the content structure and available links
        2. Prioritize links that likely contain logical next steps in the documentation
        3. Consider the following factors:
           - Links that continue the current topic
           - Links to related concepts mentioned in the current page
           - Links that are part of a sequence (next chapter, next step, etc.)
           - Links to prerequisite information needed to understand the current page
        4. Return a JSON array of objects with:
           - url: The URL to visit
           - priority: A number from 1-10 (10 being highest priority)
           - reason: Short explanation for the priority
        
        Return only a valid JSON array with no other text.
        """

        return prompt

    def _get_prioritized_links_from_llm(
        self, prompt: str, original_links: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get prioritized links from DeepSeek LLM"""
        # Use the retry function to get the response text
        response_text = self.call_llm_with_retry(
            prompt=prompt,
            max_retries=3,
            initial_backoff=2
        )
        
        # If we got a response, try to extract and process the JSON
        if response_text:
            try:
                # Extract JSON from the response
                import re

                json_match = re.search(
                    r"```json\s*(\[.*?\])\s*```", response_text, re.DOTALL
                )

                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without markdown code blocks
                    json_match = re.search(r"(\[.*?\])", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response_text

                # Try to parse the JSON
                prioritized_links = json.loads(json_str)

                # Ensure each prioritized link exists in original links
                result = []

                # Create URL lookup for original links
                url_to_link = {link["url"]: link for link in original_links}

                for pl in prioritized_links:
                    if "url" in pl and pl["url"] in url_to_link:
                        # Combine original link info with priority info
                        combined = {**url_to_link[pl["url"]]}
                        combined["priority"] = pl.get("priority", 5)
                        combined["reason"] = pl.get("reason", "")
                        result.append(combined)

                return result
                
            except json.JSONDecodeError:
                print("Failed to parse JSON from LLM response")
        
        # Return original links with default priorities if response was empty or JSON parsing failed
        return self._heuristic_prioritization(original_links)