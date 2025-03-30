import re
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger('doc_scraper')

class VersionExtractor:
    """Version extraction module with multiple strategies for different documentation platforms."""
    
    @staticmethod
    def extract_version(html_content, selector, strategy="standard"):
        """
        Extract version information from HTML content using the specified selector and strategy.
        
        Args:
            html_content (str): The HTML content to parse
            selector (str): CSS selector to find the version element
            strategy (str): Version extraction strategy name
            
        Returns:
            str: Extracted version or "unknown" if not found
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        version_element = soup.select_one(selector)
        
        if not version_element:
            logger.warning(f"Version element not found with selector: {selector}")
            return "unknown"
            
        version_text = version_element.get_text().strip()
        
        if strategy == "standard":
            return VersionExtractor._standard_extraction(version_text)
        elif strategy == "semver":
            return VersionExtractor._semver_extraction(version_text)
        elif strategy == "date_based":
            return VersionExtractor._date_based_extraction(version_text)
        else:
            logger.warning(f"Unknown version extraction strategy: {strategy}, using standard")
            return VersionExtractor._standard_extraction(version_text)
    
    @staticmethod
    def _standard_extraction(text):
        """Standard version extraction for X.Y.Z patterns."""
        version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', text)
        if version_match:
            return version_match.group(1)
        return text
    
    @staticmethod
    def _semver_extraction(text):
        """SemVer compliant version extraction (X.Y.Z-suffix)."""
        version_match = re.search(r'(\d+\.\d+\.\d+(?:-[a-zA-Z0-9\.]+)?)', text)
        if version_match:
            return version_match.group(1)
        return text
    
    @staticmethod
    def _date_based_extraction(text):
        """Date-based version extraction (YYYY-MM-DD or YYYY.MM.DD)."""
        version_match = re.search(r'(\d{4}[-\.]\d{2}[-\.]\d{2})', text)
        if version_match:
            return version_match.group(1)
        return text
        
    @staticmethod
    def detect_version_from_url(url):
        """Attempt to extract version information from a URL path."""
        # Common patterns for versions in URLs
        patterns = [
            r'/v(\d+\.\d+(?:\.\d+)?)/',  # /v1.2.3/
            r'/(\d+\.\d+(?:\.\d+)?)/docs',  # /1.2.3/docs
            r'version-(\d+\.\d+(?:\.\d+)?)',  # version-1.2.3
            r'release-(\d+\.\d+(?:\.\d+)?)'   # release-1.2.3
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None 