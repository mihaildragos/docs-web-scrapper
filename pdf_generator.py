import os
import pdfkit
import logging
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

logger = logging.getLogger('doc_scraper')

class PDFGenerator:
    """PDF generation module with multiple rendering backends and optimization capabilities."""
    
    def __init__(self, output_dir, renderer="pdfkit"):
        """
        Initialize the PDF generator.
        
        Args:
            output_dir (str): Directory to store generated PDFs
            renderer (str): PDF rendering backend ('pdfkit' or 'weasyprint')
        """
        self.output_dir = output_dir
        self.renderer = renderer
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_pdf(self, html_content, output_path, options=None):
        """
        Generate a PDF from HTML content.
        
        Args:
            html_content (str): HTML content to convert
            output_path (str): Path to save the PDF
            options (dict): Renderer-specific options
            
        Returns:
            bool: Success status
        """
        if self.renderer == "pdfkit":
            return self._generate_with_pdfkit(html_content, output_path, options)
        elif self.renderer == "weasyprint":
            return self._generate_with_weasyprint(html_content, output_path, options)
        else:
            logger.error(f"Unknown PDF renderer: {self.renderer}")
            return False
            
    def _generate_with_pdfkit(self, html_content, output_path, options=None):
        """Generate PDF using pdfkit/wkhtmltopdf."""
        default_options = {
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
        
        # Merge with custom options if provided
        if options:
            default_options.update(options)
        
        try:
            pdfkit.from_string(html_content, output_path, options=default_options)
            logger.info(f"PDF generated successfully: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate PDF with pdfkit: {e}")
            return False
            
    def _generate_with_weasyprint(self, html_content, output_path, options=None):
        """Generate PDF using WeasyPrint."""
        try:
            # Configure fonts
            font_config = FontConfiguration()
            
            # Apply default styling
            css = CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm;
                }
                body {
                    font-family: sans-serif;
                }
                pre, code {
                    font-family: monospace;
                    background-color: #f5f5f5;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                }
                a {
                    color: #0366d6;
                    text-decoration: none;
                }
            ''', font_config=font_config)
            
            # Generate PDF
            HTML(string=html_content).write_pdf(output_path, stylesheets=[css], font_config=font_config)
            logger.info(f"PDF generated successfully: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate PDF with WeasyPrint: {e}")
            return False
            
    def optimize_pdf(self, input_path, output_path=None):
        """
        Optimize the PDF file for size (if supported).
        
        Args:
            input_path (str): Path to the input PDF
            output_path (str): Path to save the optimized PDF, defaults to overwriting input
            
        Returns:
            bool: Success status
        """
        if not output_path:
            output_path = input_path
            
        # Placeholder for PDF optimization functionality
        # Could use tools like ghostscript, qpdf, etc.
        logger.warning("PDF optimization not implemented yet")
        return False 