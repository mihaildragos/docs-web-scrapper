import os
import re
import yaml
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger('doc_scraper')

class ConfigManager:
    """Configuration management with environment variable substitution and validation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and process the configuration file.
        
        Returns:
            dict: Processed configuration
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            return {"targets": []}
            
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Process environment variables
            config = self._substitute_env_vars(config)
            
            # Validate the configuration
            self._validate_config(config)
                
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {"targets": []}
            
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in the configuration.
        
        Args:
            config: Configuration object (dict, list, or scalar)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR} or $VAR with environment variable
            pattern = r'\${([^}]+)}|\$([a-zA-Z0-9_]+)'
            
            def replace_env_var(match):
                var_name = match.group(1) or match.group(2)
                return os.environ.get(var_name, f"${var_name}")
                
            return re.sub(pattern, replace_env_var, config)
        else:
            return config
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration structure.
        
        Args:
            config (dict): Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        if 'targets' not in config:
            raise ValueError("Configuration must contain 'targets' key")
            
        if not isinstance(config['targets'], list):
            raise ValueError("'targets' must be a list")
            
        for i, target in enumerate(config['targets']):
            if not isinstance(target, dict):
                raise ValueError(f"Target at index {i} must be a dictionary")
                
            required_keys = ['name', 'base_url', 'content_selector', 'output_filename']
            for key in required_keys:
                if key not in target:
                    raise ValueError(f"Target '{target.get('name', f'at index {i}')}' is missing required key '{key}'")
                    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the processed configuration.
        
        Returns:
            dict: The configuration
        """
        return self.config
        
    def get_target(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific target by name.
        
        Args:
            name (str): Target name
            
        Returns:
            dict or None: The target configuration or None if not found
        """
        for target in self.config.get('targets', []):
            if target.get('name') == name:
                return target
        return None
        
    def get_targets(self) -> List[Dict[str, Any]]:
        """
        Get all targets.
        
        Returns:
            list: All target configurations
        """
        return self.config.get('targets', []) 