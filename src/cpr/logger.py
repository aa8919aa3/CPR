"""
Enhanced logging system with performance monitoring
"""
import logging
import logging.handlers
import time
import psutil
import threading
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Any
import sys

class PerformanceLogger:
    """Performance monitoring and logging"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def setup_logging(self):
        """Setup logging with both file and console handlers"""
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Create logger
        self.logger = logging.getLogger('JJ_Analysis')
        self.logger.setLevel(getattr(logging, self.config.get('LOG_LEVEL', 'INFO')))
        
        # Create formatters
        detailed_formatter = logging.Formatter(self.config.get('LOG_FORMAT'))
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        handlers = []
        
        # File handler with rotation
        if self.config.get('FILE_LOG', True):
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.get('LOG_FILE', 'processing.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(detailed_formatter)
            handlers.append(file_handler)
        
        # Console handler
        if self.config.get('CONSOLE_LOG', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            handlers.append(console_handler)
        
        # Add handlers
        for handler in handlers:
            self.logger.addHandler(handler)
            
    def log_system_info(self):
        """Log system information"""
        self.logger.info("="*60)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("="*60)
        self.logger.info(f"CPU cores: {psutil.cpu_count()}")
        self.logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        self.logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        self.logger.info(f"Python version: {sys.version}")
        
        # Log optimization stack
        try:
            import numba
            self.logger.info(f"Numba version: {numba.__version__}")
        except ImportError:
            self.logger.warning("Numba not available - performance may be reduced")
            
        try:
            import fireducks.pandas as pd
            self.logger.info(f"FireDucks pandas: Available")
        except ImportError:
            self.logger.warning("FireDucks not available - using standard pandas")
            
    def monitor_memory(self) -> dict:
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3)
        }
        
    def log_memory_warning(self):
        """Log memory warning if usage is high"""
        memory_info = self.monitor_memory()
        if memory_info['percent'] > self.config.get('MEMORY_THRESHOLD', 85):
            self.logger.warning(f"High memory usage: {memory_info['percent']:.1f}%")
            self.logger.warning(f"Available: {memory_info['available_gb']:.1f} GB")
            
    def performance_timer(self, func_name: str = None):
        """Decorator for timing functions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                name = func_name or func.__name__
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    self.logger.debug(f"{name} completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start
                    self.logger.error(f"{name} failed after {duration:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator
        
    def log_progress(self, current: int, total: int, item_name: str = "item"):
        """Log progress with memory monitoring"""
        with self.lock:
            percentage = (current / total) * 100
            elapsed = time.time() - self.start_time
            
            if current > 0:
                estimated_total = elapsed * total / current
                remaining = estimated_total - elapsed
                
                self.logger.info(
                    f"Progress: {current}/{total} ({percentage:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {remaining:.1f}s | "
                    f"Memory: {self.monitor_memory()['percent']:.1f}%"
                )
            
            # Log memory warning if needed
            if current % 10 == 0:  # Check every 10 items
                self.log_memory_warning()

# Global logger instance will be created after config is loaded
logger: Optional[PerformanceLogger] = None

def init_logger(config):
    """Initialize global logger"""
    global logger
    logger = PerformanceLogger(config)
    return logger