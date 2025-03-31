# AI-Enhanced Workflow Documentation Scraper

An advanced documentation scraping tool that combines browser automation, OCR technology, and AI language models to intelligently extract and organize documentation content from websites.

## New Workflow-Based Architecture

The scraper now uses a modular workflow-based architecture that divides the scraping process into distinct steps:

1. **Navigation Structure Extraction**: Scans the page and extracts the navigation structure or sitemap
2. **Visual Analysis**: Analyzes the non-navigational structure of the page visually with PaddleOCR
3. **Selector Finding**: Uses DeepSeek LLM to determine optimal HTML selectors for content extraction
4. **Content Extraction**: Uses the selectors with BeautifulSoup to extract structured content
5. **Content Processing**: Feeds the extracted content to DeepSeek for coherent document organization
6. **Traversal Planning**: DeepSeek determines the next logical pages to scrape based on content relationships

## Features

- **Intelligent Content Extraction**: Uses DeepSeek LLM to identify and extract relevant content
- **Visual Recognition**: Leverages PaddleOCR to recognize text from visual elements
- **SPA Support**: Handles JavaScript-heavy and single-page applications
- **CloudFlare Bypass**: Built-in solution for websites protected by CloudFlare
- **Hierarchical Output**: Preserves the original documentation structure
- **Markdown Conversion**: Converts HTML content to clean, readable markdown
- **Docker-based**: Runs in containerized environments for portability and isolation
- **Modular Architecture**: Each step is implemented in a separate module for easy customization

## Project Structure

```
docs-web-scraper/
├── LICENSE                   # MIT License file
├── README.md                 # This documentation
├── docker-compose.yml        # Container orchestration
├── scrape.sh                 # Main execution script
│
├── deepseek-server/          # AI language model service
│   ├── Dockerfile            # Container definition for DeepSeek
│   ├── app.py                # FastAPI server for the LLM
│   ├── download_model.py     # Script to download the model
│   └── requirements.txt      # Python dependencies
│
└── scraper/                  # Documentation scraper service
    ├── Dockerfile            # Container definition for scraper
    ├── workflow_scraper.py   # Main workflow-based scraper implementation
    ├── download_paddle_models.py # Pre-downloads OCR models
    ├── requirements.txt      # Python dependencies
    ├── config_example.json   # Example configuration file
    │
    └── lib/                  # Modular workflow components
        ├── __init__.py       # Package initialization
        ├── navigation_extractor.py # Navigation structure extraction
        ├── visual_analyzer.py      # Visual page analysis with OCR
        ├── selector_finder.py      # AI-powered selector discovery
        ├── content_extractor.py    # Content extraction with BeautifulSoup
        ├── content_processor.py    # AI-powered content processing
        └── traversal_planner.py    # Intelligent traversal planning
```

## Prerequisites

- Docker and Docker Compose
- Git
- 8GB+ RAM allocated to Docker
- Stable internet connection (for downloading models)
- (Optional) NVIDIA GPU with CUDA support for faster AI inference

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/docs-web-scraper.git
   cd docs-web-scraper
   ```

2. **Create required directories**:

   ```bash
   mkdir -p output
   mkdir -p deepseek-server/logs
   mkdir -p session_data
   ```

3. **Build the Docker containers**:

   ```bash
   docker-compose build
   ```

   This process may take some time, as it:

   - Builds the DeepSeek LLM server container
   - Builds the scraper container
   - Pre-downloads PaddleOCR models

4. **Make the script executable**:
   ```bash
   chmod +x scrape.sh
   ```

## Usage Instructions

### Basic Usage

```bash
./scrape.sh https://docs.example.com
```

This will scrape the specified documentation site with default parameters and save the results to the `./output` directory.

### Advanced Usage

```bash
./scrape.sh https://shopify.dev/docs/apps/build --depth 3 --delay 2 --debug --max-pages 50
```

### Available Options

- `--output`, `-o`: Output directory (default: "docs")
- `--depth`, `-d`: Maximum crawl depth (default: 3)
- `--delay`, `-w`: Delay between requests in seconds (default: 1.0)
- `--llm`: DeepSeek LLM endpoint (default: "http://deepseek:8000")
- `--visible`: Run browser in visible mode (not headless)
- `--debug`: Enable debug mode (save intermediate files)
- `--max-pages`: Maximum number of pages to scrape (default: 100)
- `--solve-captcha`: Open browser on your Mac to solve CloudFlare CAPTCHA first

### Pre-warm the DeepSeek server

Start the server first and wait for model loading to complete:

```bash
bashCopydocker-compose up -d deepseek
# Wait for ~5 minutes before running the scrape
./scrape.sh https://shopify.dev/docs/apps/build
```


## CloudFlare Bypass Instructions

Many documentation sites use CloudFlare protection that requires solving a CAPTCHA. The scraper includes functionality to handle this case:

1. **Run with the solve-captcha option**:

   ```bash
   ./scrape.sh https://docs.example.com --solve-captcha
   ```

2. **Solve the CAPTCHA in the browser window** that opens on your machine

3. **Wait for confirmation** that the session has been saved

4. **Continue with scraping** using the authenticated session:
   ```bash
   ./scrape.sh https://docs.example.com --depth 3 --debug
   ```

The authenticated session will be stored in `./session_data/cloudflare_session.json` and automatically mounted into the container for subsequent scraping runs.

## Workflow Architecture Details

### 1. Navigation Extraction

The `NavigationExtractor` identifies the site's navigation structure by:

- Analyzing common navigation elements (menus, sidebars, etc.)
- Detecting and processing sitemaps
- Extracting links from JavaScript-powered menus

### 2. Visual Analysis

The `VisualAnalyzer` performs visual analysis using PaddleOCR to:

- Extract text from screenshots
- Identify visual sections (header, main content, sidebar, footer)
- Detect tables and structured content visually

### 3. Selector Finding

The `SelectorFinder` uses DeepSeek LLM to:

- Analyze HTML structure and visual analysis data
- Determine optimal CSS selectors for different content types
- Validate selectors against the actual HTML

### 4. Content Extraction

The `ContentExtractor` uses BeautifulSoup with the optimal selectors to:

- Extract main content, navigation, code blocks, tables, figures, and notes
- Convert content to structured format
- Extract metadata from the page

### 5. Content Processing

The `ContentProcessor` uses DeepSeek LLM to:

- Organize extracted content into a coherent document
- Improve formatting and structure
- Create a comprehensive document structure

### 6. Traversal Planning

The `TraversalPlanner` uses DeepSeek LLM to:

- Analyze content relationships
- Determine the logical next pages to scrape
- Prioritize pages based on relevance and structure

## Advanced Configuration

For advanced customization, you can edit the configuration example in `scraper/config_example.json`.

### Using a Different LLM Model

To use a different DeepSeek model, edit the `docker-compose.yml` file:

```yaml
services:
  deepseek:
    environment:
      - MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct # Change this line
```

Available models:

- `deepseek-ai/deepseek-llm-7b-base` (default)
- `deepseek-ai/deepseek-llm-7b-chat`
- `deepseek-ai/deepseek-coder-6.7b-instruct` (good for code documentation)
- `deepseek-ai/deepseek-llm-1.3b-base` (for low-memory systems)

## Output Format

The scraper generates:

- Markdown files for each documentation page
- A hierarchical directory structure mirroring the website
- Debug files for each step in the process when debug mode is enabled

Each markdown file includes:

- The original title
- Source URL
- Processed content with preserved formatting

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:

   - Increase Docker's available memory (8GB+ recommended)
   - Use a smaller model (`deepseek-ai/deepseek-llm-1.3b-base`)
   - Reduce `MAX_LENGTH` and `MAX_NEW_TOKENS` in `docker-compose.yml`

2. **CloudFlare Detection**:

   - Use the `--solve-captcha` option to manually solve the CAPTCHA
   - Increase delay between requests with `--delay 2` or higher

3. **Scraper Not Finding Content**:

   - Enable debug mode with `--debug` to see what's being captured
   - Use `--visible` to watch the browser's activity
   - Check the debug files to see the selectors being used

4. **PaddleOCR Model Download Fails**:

   - The scraper will retry downloads automatically
   - If it persistently fails, check network connectivity
   - Consider downloading the models manually according to PaddleOCR documentation

5. **DeepSeek Model Not Loading**:
   - Check Docker logs with `docker-compose logs deepseek`
   - Verify you have enough disk space for model files
   - Try a smaller model if memory is constrained



### Logging

You can view the logs from your Docker containers in a few different ways:

#### Basic Logs

```bash
# View logs for the deepseek container
docker-compose logs deepseek
```

#### Watch Logs in Real-time

```bash
# Follow logs as they come in (stream)
docker-compose logs -f deepseek
```

This is especially useful to see the model loading progress.

#### Advanced Options

```bash
# View last 100 lines with timestamps
docker-compose logs --timestamps --tail=100 deepseek

# View only errors
docker-compose logs deepseek 2>&1 | grep -i error

# Check model loading status
docker-compose logs deepseek | grep -i "loading model"
```

#### Alternative Using Direct Docker Commands

```bash
# Using container name
docker logs deepseek-api

# Follow logs
docker logs -f deepseek-api
```

#### Check Container Status

```bash
# List all running containers with their status
docker ps

# Show detailed info about the container
docker inspect deepseek-api | grep -A 10 State
```

#### Check API Status Directly

```bash
# Check if the API is up and if the model is loaded
curl http://localhost:8000/status
```

This should return JSON that includes `"is_loaded": true` when the model is fully loaded.

The logs will show you the model loading progress and any errors that might be occurring during initialization. This is helpful for debugging the "Model is still loading" errors you were experiencing.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created by Mihail Dragos - Feel free to contribute by submitting pull requests or opening issues on GitHub.
