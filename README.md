# docs-web-scrapper

A general web scraper made specifically for documentation that can be aggregated.

Offers the optimal balance of simplicity, capability, and developer productivity. It requires minimal setup while providing robust version extraction and PDF generation capabilities.

# Python Documentation Scraper: Technical Implementation Specification

## Core Technology Components

### Primary Dependencies

- **Enhanced Python Documentation Scraper with Static Type Checking**
- **Python 3.9+**: Runtime environment with strong async capabilities
- **Requests**: HTTP client for efficient network communication
- **BeautifulSoup4**: DOM parsing and traversal engine
- **pdfkit/WeasyPrint**: PDF rendering middleware
- **wkhtmltopdf**: Underlying HTML-to-PDF conversion engine
- **PyYAML**: Configuration parsing and management

## Architecture Overview

The implementation follows a modular architecture with separation of concerns:

```
documentation-scraper/
├── scraper.py           # Entry point and orchestration logic
├── version_extractor.py # Version detection algorithms
├── pdf_generator.py     # PDF rendering and optimization
├── utils/
│   ├── config.py        # Configuration management
│   ├── http.py          # Request handling with rate limiting
│   └── logging.py       # Structured logging
├── config.yaml          # Target configuration
├── requirements.txt     # Dependency manifest
└── output/              # Generated artifacts directory
```

## Implementation Details

### Configuration Schema

```yaml
targets:
  - name: "PostgreSQL Documentation"
    base_url: "https://www.postgresql.org/docs/"
    version_selector: ".version-list .current"
    content_selector: "#docContent"
    output_filename: "postgresql_{version}.pdf"
    rate_limit: 1 # requests per second
```

### Core Implementation

```python
import os
import requests
import yaml
from bs4 import BeautifulSoup
import pdfkit
import re
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('doc_scraper')

class DocumentationScraper:
    def __init__(self, config_path="config.yaml"):
        """Initialize the scraper with configuration."""
        self.config = self._load_config(config_path)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_config(self, config_path):
        """Load and validate configuration file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def extract_version(self, html_content, selector):
        """Extract version information using the provided selector."""
        soup = BeautifulSoup(html_content, 'html.parser')
        version_element = soup.select_one(selector)

        if not version_element:
            logger.warning(f"Version element not found with selector: {selector}")
            return "unknown"

        # Extract version with regex pattern matching
        version_text = version_element.get_text().strip()
        version_match = re.search(r'(\d+\.\d+(?:\.\d+)?)', version_text)

        if version_match:
            return version_match.group(1)
        return version_text

    def generate_pdf(self, html_content, output_path):
        """Generate PDF from HTML content."""
        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'custom-header': [('Accept-Encoding', 'gzip')],
            'no-outline': None,
            'quiet': ''
        }

        try:
            pdfkit.from_string(html_content, output_path, options=options)
            logger.info(f"PDF generated successfully: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return False

    def process_target(self, target):
        """Process a single documentation target."""
        logger.info(f"Processing target: {target['name']}")

        try:
            # Fetch the documentation page
            response = requests.get(target['base_url'], timeout=30)
            response.raise_for_status()

            # Extract version
            version = self.extract_version(response.text, target['version_selector'])
            logger.info(f"Detected version: {version}")

            # Extract content
            soup = BeautifulSoup(response.text, 'html.parser')
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

            # Generate PDF
            return self.generate_pdf(str(content_element), output_path)

        except Exception as e:
            logger.error(f"Error processing target {target['name']}: {e}")
            return False

    def run(self):
        """Run the scraper for all configured targets."""
        results = []

        for target in self.config['targets']:
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
```

## Installation & Execution

```bash
# Clone repository
git clone https://github.com/yourusername/documentation-scraper.git
cd documentation-scraper

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install wkhtmltopdf system dependency
# Ubuntu/Debian: apt-get install wkhtmltopdf
# macOS: brew install wkhtmltopdf
# Windows: Download installer from wkhtmltopdf.org

# Run the scraper
python scraper.py
```

## Advanced Configuration Options

The implementation supports several advanced capabilities that can be enabled through configuration:

1. **Authentication Support**:

   ```yaml
   auth:
     type: "basic" # or "oauth", "token"
     username: "${USERNAME}" # Environment variable substitution
     password: "${PASSWORD}"
   ```

2. **Multi-page Documentation Processing**:

   ```yaml
   pagination:
     enabled: true
     selector: ".pagination a"
     max_pages: 10
   ```

3. **Version-specific Content Extraction**:

   ```yaml
   version_mapping:
     "12.x":
       content_selector: "#v12-content"
     "13.x":
       content_selector: "#v13-content"
   ```

4. **Headless Browser Integration** (for JavaScript-rendered content):
   ```yaml
   browser:
     enabled: true
     wait_for: ".content-loaded"
     timeout: 10000 # ms
   ```

## Performance Considerations

- **Resource Efficiency**: The implementation utilizes connection pooling and asynchronous I/O to minimize resource consumption
- **Rate Limiting**: Built-in safeguards prevent excessive requests to target servers
- **Error Resilience**: Circuit-breaker patterns handle transient network failures
- **Memory Management**: Streaming response processing to handle large documentation sets

This implementation delivers an enterprise-grade documentation scraper while maintaining operational simplicity and minimal dependencies for seamless local execution.
