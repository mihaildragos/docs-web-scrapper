import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger('doc_scraper')

class RateLimitedSession:
    """Session class with built-in rate limiting and retry logic."""
    
    def __init__(self, rate_limit: float = 1.0, retry_attempts: int = 3):
        """
        Initialize rate-limited session.
        
        Args:
            rate_limit (float): Maximum requests per second
            retry_attempts (int): Number of retry attempts for failed requests
        """
        self.rate_limit = rate_limit
        self.session = self._create_session(retry_attempts)
        self.last_request_time = 0
        
    def _create_session(self, retry_attempts: int) -> requests.Session:
        """
        Create a requests session with retry configuration.
        
        Args:
            retry_attempts (int): Number of retry attempts
            
        Returns:
            requests.Session: Configured session
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def _wait_for_rate_limit(self) -> None:
        """Wait to respect the rate limit before making a request."""
        if self.rate_limit <= 0:
            return
            
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a rate-limited HTTP request.
        
        Args:
            method (str): HTTP method
            url (str): URL to request
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: Response object
        """
        self._wait_for_rate_limit()
        logger.debug(f"Making {method} request to {url}")
        return self.session.request(method, url, **kwargs)
        
    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Make a rate-limited GET request.
        
        Args:
            url (str): URL to request
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: Response object
        """
        return self.request("GET", url, **kwargs)
        
    def close(self) -> None:
        """Close the session."""
        self.session.close()
        
    def __enter__(self) -> 'RateLimitedSession':
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()

class HTTPClient:
    """HTTP client for documentation scraping with rate limiting and authentication."""
    
    def __init__(self, rate_limit: float = 1.0, auth_config: Optional[Dict[str, Any]] = None):
        """
        Initialize HTTP client.
        
        Args:
            rate_limit (float): Maximum requests per second
            auth_config (dict, optional): Authentication configuration
        """
        self.session = RateLimitedSession(rate_limit=rate_limit)
        self.auth_config = auth_config
        
        if auth_config:
            self._configure_auth()
            
    def _configure_auth(self) -> None:
        """Configure authentication based on auth_config."""
        if not self.auth_config:
            return
            
        auth_type = self.auth_config.get('type', '').lower()
        
        if auth_type == 'basic':
            username = self.auth_config.get('username')
            password = self.auth_config.get('password')
            
            if username and password:
                self.session.session.auth = (username, password)
                logger.info("Basic authentication configured")
            else:
                logger.warning("Basic authentication missing username/password")
                
        elif auth_type == 'token':
            token = self.auth_config.get('token')
            header_name = self.auth_config.get('header_name', 'Authorization')
            prefix = self.auth_config.get('prefix', 'Bearer')
            
            if token:
                self.session.session.headers.update({
                    header_name: f"{prefix} {token}" if prefix else token
                })
                logger.info(f"Token authentication configured with {header_name} header")
            else:
                logger.warning("Token authentication missing token value")
                
        elif auth_type == 'oauth':
            logger.warning("OAuth authentication not yet implemented")
        else:
            logger.warning(f"Unknown authentication type: {auth_type}")
            
    def get(self, url: str, **kwargs) -> Tuple[Union[Dict[str, Any], str], int]:
        """
        Make a GET request with error handling.
        
        Args:
            url (str): URL to request
            **kwargs: Additional arguments for requests
            
        Returns:
            tuple: (content, status_code) - content is JSON dict or string
        """
        try:
            response = self.session.get(url, **kwargs)
            status_code = response.status_code
            
            # Try to parse as JSON
            try:
                content = response.json()
            except ValueError:
                content = response.text
                
            return content, status_code
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}, 0
            
    def close(self) -> None:
        """Close the session."""
        self.session.close()
        
    def __enter__(self) -> 'HTTPClient':
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close() 