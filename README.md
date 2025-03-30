# Doc-Scraper-AI: An AI-Enhanced Documentation Scraper

This project combines browser automation, optical character recognition (OCR), and a local large language model (LLM) to create a powerful documentation scraper that can handle modern web apps and extract high-quality content.

## Features

- 🖥️ **Handles SPAs and JavaScript-heavy sites** using Playwright
- 👁️ **Visual content recognition** with PaddleOCR
- 🧠 **Intelligent content extraction** using DeepSeek LLM
- 🔄 **Follows links** to create complete documentation archives
- 📝 **Markdown output** for easy integration with knowledge management tools

## Project Structure

```
doc-scraper-ai/
├── scraper/                   # Documentation scraper
│   ├── advanced_doc_scraper.py  # Main scraper code
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Scraper container
├── deepseek-server/           # LLM server
│   ├── app.py                   # FastAPI server
│   ├── download_model.py        # Model downloader
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # LLM container
├── docker-compose.yml         # Orchestrates both services
└── README.md                  # This file
```

## Prerequisites

- Docker and Docker Compose
- For GPU acceleration (highly recommended):
  - NVIDIA GPU with at least 16GB VRAM
  - NVIDIA Container Toolkit installed

## Setup Instructions

1. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/doc-scraper-ai
   cd doc-scraper-ai
   ```

2. **Create directories for model cache and output**:

   ```bash
   mkdir -p deepseek-server/model_cache deepseek-server/logs output
   ```

3. **Build the containers**:

   ```bash
   docker-compose build
   ```

4. **Start the DeepSeek server** (this may take a while on first run as it downloads the model):

   ```bash
   docker-compose up deepseek
   ```

   Wait until you see "Model loaded successfully" in the logs.

## Usage

### Scraping a Documentation Site

Run the scraper with Docker Compose:

```bash
docker-compose run --rm scraper python advanced_doc_scraper.py https://docs.example.com --depth 2
```

### Command-Line Options

- `url`: The URL to scrape (required)
- `--output`, `-o`: Output directory (default: "docs")
- `--depth`, `-d`: Maximum crawl depth (default: 1)
- `--delay`, `-w`: Delay between requests in seconds (default: 1.0)
- `--llm`: DeepSeek LLM endpoint (default: uses Docker service)
- `--visible`: Run browser in visible mode (not headless)
- `--debug`: Enable debug mode (save screenshots, etc.)

Example with options:

```bash
docker-compose run --rm scraper python advanced_doc_scraper.py https://docs.example.com --depth 3 --delay 2 --debug
```

### Output

The scraped documentation will be saved to the `output` directory in your project folder, with the following structure:

```
output/
├── index.md                  # Homepage
├── getting-started.md        # Example page
├── api/
│   ├── index.md              # API section homepage
│   ├── authentication.md     # API authentication page
│   └── ...                   # Other API pages
└── ...                       # Other sections
```

## Customization

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

### CPU-Only Mode

For systems without a GPU, modify the `docker-compose.yml` file:

1. Remove the NVIDIA deployment section from the deepseek service
2. Set the device to CPU:
   ```yaml
   environment:
     - DEVICE=cpu
   ```
3. Consider using the smaller 1.3B model for better performance

## Troubleshooting

### DeepSeek Server Issues

- **Out of Memory**: Try a smaller model or increase your GPU memory
- **Slow Responses**: If using CPU mode, expect much slower performance
- **Model Loading Errors**: Check disk space and network connectivity

### Scraper Issues

- **JavaScript Not Working**: Try increasing the wait time with `--delay`
- **OCR Not Accurate**: Use `--debug` to save screenshots and inspect results
- **Not Finding Links**: Check if the site uses custom navigation or requires authentication

## License

This project is licensed under the MIT License.
