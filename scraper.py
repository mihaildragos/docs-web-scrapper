import os
import logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from utils.config import ConfigManager
from utils.http import HTTPClient
from version_extractor import VersionExtractor
from pdf_generator import PDFGenerator

# Configure basic logging, will be enhanced by utils.logging if used from __main__
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('doc_scraper')

class DocumentationScraper:
    def __init__(self, config_path="config.yaml"):
        """Initialize the scraper with configuration."""
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Set up output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      self.config.get('settings', {}).get('output_dir', "output"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize PDF generator
        renderer = self.config.get('settings', {}).get('renderer', 'pdfkit')
        self.pdf_generator = PDFGenerator(self.output_dir, renderer=renderer)
        
        # Set up authentication if configured
        self.auth_config = self.config.get('auth')

    def process_target(self, target):
        """Process a single documentation target."""
        logger.info(f"Processing target: {target['name']}")
        
        try:
            # Get rate limit from target or default to 1
            rate_limit = float(target.get('rate_limit', 1.0))
            
            # Create HTTP client with rate limiting
            with HTTPClient(rate_limit=rate_limit, auth_config=self.auth_config) as client:
                # Fetch the documentation page
                content, status_code = client.get(target['base_url'], timeout=30)
                
                if status_code != 200:
                    logger.error(f"Failed to fetch {target['base_url']}: HTTP {status_code}")
                    return False
                
                if isinstance(content, dict) and 'error' in content:
                    logger.error(f"Error fetching {target['base_url']}: {content['error']}")
                    return False
                
                # Extract version using the version extractor
                version_strategy = target.get('version_strategy', 'standard')
                version = VersionExtractor.extract_version(
                    content, 
                    target['version_selector'],
                    strategy=version_strategy
                )
                logger.info(f"Detected version: {version}")
                
                # If version not found in content, try to extract from URL
                if version == "unknown" and 'version_from_url' not in target:
                    url_version = VersionExtractor.detect_version_from_url(target['base_url'])
                    if url_version:
                        version = url_version
                        logger.info(f"Detected version from URL: {version}")
                
                # Extract content
                soup = BeautifulSoup(content, 'html.parser')
                content_element = soup.select_one(target['content_selector'])
                
                if not content_element:
                    logger.error(f"Content element not found with selector: {target['content_selector']}")
                    return False
                
                # Ensure all relative URLs are converted to absolute
                for tag in content_element.find_all(['a', 'img', 'link']):
                    for attr in ['href', 'src']:
                        if tag.has_attr(attr) and not tag[attr].startswith(('http://', 'https://')):
                            tag[attr] = urljoin(target['base_url'], tag[attr])
                
                # Generate PDF filename with version
                output_filename = target['output_filename'].format(version=version)
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Generate PDF using the PDF generator
                pdf_options = target.get('pdf_options', None)
                return self.pdf_generator.generate_pdf(str(content_element), output_path, options=pdf_options)
                
        except Exception as e:
            logger.error(f"Error processing target {target['name']}: {e}")
            return False

    def run(self):
        """Run the scraper for all configured targets."""
        results = []
        
        for target in self.config_manager.get_targets():
            success = self.process_target(target)
            results.append({
                'name': target['name'],
                'success': success
            })
        
        # Summary
        logger.info("--- Scraping Summary ---")
        for result in results:
            status = "✓ Success" if result['success'] else "✗ Failed"
            logger.info(f"{status}: {result['name']}")

if __name__ == "__main__":
    scraper = DocumentationScraper()
    scraper.run() 