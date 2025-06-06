"""
Configuration management for Josephson Junction Analysis
"""
import os
import json
import multiprocessing
from pathlib import Path
from typing import Dict, Any
import logging

class Config:
    """Configuration manager with environment variable support and validation"""
    
    DEFAULT_CONFIG = {
        # File paths
        'INPUT_FOLDER': 'data/Ic',
        'OUTPUT_FOLDER': 'output',
        'SUMMARY_FILE': 'analysis_summary.csv',
        'LOG_FILE': 'processing.log',
        
        # Performance settings
        'MAX_WORKERS': min(8, multiprocessing.cpu_count()),
        'MEMORY_THRESHOLD': 85,  # Percentage
        'TIMEOUT_PER_FILE': 300,  # seconds
        'CHUNK_SIZE': 100,  # files per chunk for large batches
        
        # Processing settings
        'MIN_DATA_POINTS': 20,
        'PREPROCESSING_SKIP': 10,
        'FAST_MODE': False,
        
        # Visualization settings
        'DPI_HIGH': 100,  # Changed for 1920x1080 output
        'DPI_FAST': 100,  # Consistent DPI
        'FIGURE_SIZE': (19.2, 10.8),  # 1920x1080 at 100 DPI
        'SAVE_PLOTS': True,
        'PLOT_FORMATS': ['png'],  # ['png', 'pdf', 'svg']
        
        # Analysis parameters
        'FREQUENCY_POINTS': 10000,
        'PEAK_HEIGHT_THRESHOLD': 0.1,
        'PEAK_DISTANCE': 100,
        'PHASE_BINS': 20,
        'MAX_ITERATIONS': 50000,
        
        # Logging settings
        'LOG_LEVEL': 'INFO',
        'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'CONSOLE_LOG': True,
        'FILE_LOG': True,
    }

    def __init__(self, config_file: str = 'config.json'):
        self.config_file = Path(config_file)
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config()
        self._override_from_env()
        self._validate_config()
        
    def _load_config(self):
        """Load configuration from JSON file if exists"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                logging.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config file: {e}")
                
    def _override_from_env(self):
        """Override config with environment variables"""
        env_mappings = {
            'JJ_INPUT_FOLDER': 'INPUT_FOLDER',
            'JJ_OUTPUT_FOLDER': 'OUTPUT_FOLDER',
            'JJ_MAX_WORKERS': 'MAX_WORKERS',
            'JJ_FAST_MODE': 'FAST_MODE',
            'JJ_LOG_LEVEL': 'LOG_LEVEL',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Type conversion
                if config_key in ['MAX_WORKERS', 'MEMORY_THRESHOLD', 'TIMEOUT_PER_FILE']:
                    self.config[config_key] = int(env_value)
                elif config_key in ['FAST_MODE']:
                    self.config[config_key] = env_value.lower() == 'true'
                else:
                    self.config[config_key] = env_value
                    
    def _validate_config(self):
        """Validate configuration values"""
        # Ensure directories exist
        Path(self.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)
        
        # Validate numeric values
        if self.config['MAX_WORKERS'] < 1:
            self.config['MAX_WORKERS'] = 1
        if self.config['MAX_WORKERS'] > multiprocessing.cpu_count():
            self.config['MAX_WORKERS'] = multiprocessing.cpu_count()
            
        # Validate memory threshold
        if not 50 <= self.config['MEMORY_THRESHOLD'] <= 95:
            self.config['MEMORY_THRESHOLD'] = 85
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
        
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            
    @property
    def dpi(self) -> int:
        """Get DPI based on fast mode setting"""
        return self.config['DPI_FAST'] if self.config['FAST_MODE'] else self.config['DPI_HIGH']

# Global config instance
config = Config()