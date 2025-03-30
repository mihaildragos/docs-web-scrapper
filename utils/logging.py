import os
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        """
        Initialize the structured log formatter.
        
        Args:
            include_timestamp (bool): Whether to include timestamp in logs
            include_level (bool): Whether to include log level in logs
        """
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON object.
        
        Args:
            record (logging.LogRecord): Log record to format
            
        Returns:
            str: JSON formatted log entry
        """
        log_data = {}
        
        # Include timestamp if requested
        if self.include_timestamp:
            log_data['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
            
        # Include log level if requested
        if self.include_level:
            log_data['level'] = record.levelname
            
        # Always include logger name and message
        log_data['logger'] = record.name
        log_data['message'] = record.getMessage()
        
        # Include exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        # Include extra fields if present
        if hasattr(record, 'data') and isinstance(record.data, dict):
            log_data.update(record.data)
            
        return json.dumps(log_data)

class LoggerFactory:
    """Factory class for creating configured loggers."""
    
    @staticmethod
    def create_logger(name: str, 
                     level: int = logging.INFO,
                     output_file: Optional[str] = None,
                     console_output: bool = True,
                     structured: bool = False,
                     log_dir: str = "logs") -> logging.Logger:
        """
        Create and configure a logger.
        
        Args:
            name (str): Logger name
            level (int): Logging level
            output_file (str, optional): File to write logs to
            console_output (bool): Whether to output logs to console
            structured (bool): Whether to use structured JSON logging
            log_dir (str): Directory for log files
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatters
        if structured:
            formatter = StructuredLogFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        # Add file handler if requested
        if output_file:
            # Create log directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_path = os.path.join(log_dir, output_file)
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
        
    @staticmethod
    def setup_root_logger(level: int = logging.INFO, 
                         output_file: Optional[str] = None, 
                         structured: bool = False) -> logging.Logger:
        """
        Set up the root logger.
        
        Args:
            level (int): Logging level
            output_file (str, optional): File to write logs to
            structured (bool): Whether to use structured JSON logging
            
        Returns:
            logging.Logger: Configured root logger
        """
        return LoggerFactory.create_logger(
            name="root", 
            level=level, 
            output_file=output_file, 
            structured=structured
        )

def log_with_context(logger: logging.Logger, level: int, msg: str, context: Dict[str, Any]) -> None:
    """
    Log a message with additional context data.
    
    Args:
        logger (logging.Logger): Logger to use
        level (int): Logging level (e.g. logging.INFO)
        msg (str): Log message
        context (dict): Additional context data
    """
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None
    )
    record.data = context
    
    for handler in logger.handlers:
        if record.levelno >= handler.level:
            handler.handle(record) 