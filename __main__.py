#!/usr/bin/env python3
"""Documentation Scraper CLI entrypoint."""

import os
import sys
import argparse
import logging

from scraper import DocumentationScraper
from utils.logging import LoggerFactory

def main():
    """Main entry point for the documentation scraper."""
    parser = argparse.ArgumentParser(description="Documentation Scraper")
    parser.add_argument(
        "-c", "--config", 
        default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        help="Output directory for generated PDFs"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-t", "--target", 
        help="Process only the specified target by name"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = LoggerFactory.create_logger(
        name="doc_scraper", 
        level=log_level, 
        output_file="scraper.log", 
        log_dir="logs"
    )
    
    # Create and run the scraper
    scraper = DocumentationScraper(config_path=args.config)
    
    # Override output directory if specified
    if args.output_dir:
        scraper.output_dir = os.path.abspath(args.output_dir)
        os.makedirs(scraper.output_dir, exist_ok=True)
        
    # Run the scraper
    if args.target:
        # Process only the specified target
        for target in scraper.config.get('targets', []):
            if target['name'] == args.target:
                success = scraper.process_target(target)
                status = "✓ Success" if success else "✗ Failed"
                print(f"{status}: {target['name']}")
                return 0 if success else 1
                
        print(f"Target not found: {args.target}")
        return 1
    else:
        # Process all targets
        scraper.run()
        return 0

if __name__ == "__main__":
    sys.exit(main()) 