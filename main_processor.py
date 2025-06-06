"""
Enhanced main processor with all optimizations
"""
import os
import sys
import glob
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import numpy as np
import traceback

# Try to import optimized pandas, fallback to standard pandas
try:
    import fireducks.pandas as pd
    USING_FIREDUCKS = True
except ImportError:
    import pandas as pd
    USING_FIREDUCKS = False

from config import config
from logger import init_logger
from josephson_model import JosephsonFitter, preprocess_data_numba
from analysis_utils import FrequencyAnalyzer, PhaseAnalyzer, validate_data
from visualization import PublicationPlotter
from memory_manager import AdaptiveProcessor

class EnhancedJosephsonProcessor:
    """Enhanced Josephson junction data processor with all optimizations"""
    
    def __init__(self):
        self.config = config
        self.logger = init_logger(config)
        self.memory_manager = AdaptiveProcessor(config)
        self.plotter = PublicationPlotter(config) if config.get('SAVE_PLOTS', True) else None
        
        # Initialize components
        self.josephson_fitter = JosephsonFitter(config)
        self.frequency_analyzer = FrequencyAnalyzer(config)
        self.phase_analyzer = PhaseAnalyzer(config)
        
        # Thread-safe counters
        self.lock = threading.Lock()
        self.progress = {'current': 0, 'total': 0}
        self.results = []
        
        # Pre-compile numba functions
        self._precompile_numba()
        
    def _precompile_numba(self):
        """Pre-compile numba functions for better performance"""
        self.logger.logger.info("Pre-compiling Numba functions...")
        try:
            # Dummy compilation
            dummy_x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            dummy_y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            
            from josephson_model import josephson_model_numba, calculate_statistics_numba
            from analysis_utils import calculate_phase_data_numba, calculate_binned_average_numba
            
            _ = josephson_model_numba(dummy_x, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0)
            _ = calculate_statistics_numba(dummy_y, dummy_y, 6)
            _ = preprocess_data_numba(dummy_x, dummy_y)
            _ = calculate_phase_data_numba(dummy_x, 1.0)
            _ = calculate_binned_average_numba(dummy_x/2, dummy_y)
            
            self.logger.logger.info("✓ Numba functions compiled successfully")
        except Exception as e:
            self.logger.logger.warning(f"Numba compilation warning: {e}")
    
    def process_single_file(self, csv_file_path: str) -> Dict[str, Any]:
        """Process a single CSV file with comprehensive error handling"""
        dataid = Path(csv_file_path).stem
        start_time = time.time()
        
        try:
            # Load data
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_cols = ['y_field', 'Ic']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract data
            x_data = df['y_field'].values.astype(np.float64)
            y_data = df['Ic'].values.astype(np.float64)
            
            # Validate data
            validation = validate_data(x_data, y_data, self.config.get('MIN_DATA_POINTS', 20))
            if not validation['valid']: