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

from .config import config
from .logger import init_logger
from .josephson_model import JosephsonFitter, preprocess_data_numba
from .analysis_utils import FrequencyAnalyzer, PhaseAnalyzer, validate_data
from .visualization import PublicationPlotter
from .memory_manager import AdaptiveProcessor

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
            
            from .josephson_model import josephson_model_numba, calculate_statistics_numba
            from .analysis_utils import calculate_phase_data_numba, calculate_binned_average_numba
            
            _ = josephson_model_numba(dummy_x, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0)
            _ = calculate_statistics_numba(dummy_y, dummy_y, 6)
            _ = preprocess_data_numba(dummy_x, dummy_y)
            _ = calculate_phase_data_numba(dummy_x, 1.0)
            _ = calculate_binned_average_numba(dummy_x/2, dummy_y)
            
            self.logger.logger.info("✓ Numba functions compiled successfully")
        except Exception as e:
            self.logger.logger.warning(f"Numba compilation warning: {e}")
    
    def load_data(self, filepath: str) -> tuple:
        """Load and validate data from CSV file"""
        try:
            # Load data
            df = pd.read_csv(filepath)
            
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
                raise ValueError(f"Data validation failed: {validation['message']}")
            
            return x_data, y_data
            
        except Exception as e:
            self.logger.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise

    def process_single_file(self, csv_file_path: str) -> Dict[str, Any]:
        """Process a single CSV file with comprehensive error handling"""
        dataid = Path(csv_file_path).stem
        start_time = time.time()
        
        try:
            # Load data
            x_data, y_data = self.load_data(csv_file_path)
            
            # Basic processing result
            result = {
                'dataid': dataid,
                'filepath': csv_file_path,
                'n_points': len(x_data),
                'x_range': [float(np.min(x_data)), float(np.max(x_data))],
                'y_range': [float(np.min(y_data)), float(np.max(y_data))],
                'processing_time': time.time() - start_time,
                'status': 'success'
            }
            
            self.logger.logger.info(f"Processed {dataid}: {len(x_data)} data points")
            return result
            
        except Exception as e:
            error_msg = f"Error processing {dataid}: {str(e)}"
            self.logger.logger.error(error_msg)
            return {
                'dataid': dataid,
                'filepath': csv_file_path,
                'status': 'error',
                'error': error_msg,
                'processing_time': time.time() - start_time
            }

    def batch_process_files(self):
        """Batch process all CSV files in the input folder"""
        try:
            input_folder = Path(self.config.get('INPUT_FOLDER', 'data/Ic'))
            if not input_folder.exists():
                self.logger.logger.error(f"Input folder not found: {input_folder}")
                return
            
            csv_files = list(input_folder.glob('*.csv'))
            if not csv_files:
                self.logger.logger.warning(f"No CSV files found in {input_folder}")
                return
            
            self.logger.logger.info(f"Found {len(csv_files)} CSV files to process")
            
            results = []
            for csv_file in csv_files:
                try:
                    self.logger.logger.info(f"Processing {csv_file.name}...")
                    result = self.process_single_file(str(csv_file))
                    results.append(result)
                    if result.get('status') == 'success':
                        self.logger.logger.info(f"Successfully processed {csv_file.name}")
                    else:
                        self.logger.logger.warning(f"Failed to process {csv_file.name}")
                except Exception as e:
                    self.logger.logger.error(f"Error processing {csv_file.name}: {str(e)}")
                    continue
            
            # Save summary
            if results:
                output_folder = Path(self.config.get('OUTPUT_FOLDER', 'output'))
                output_folder.mkdir(exist_ok=True)
                
                summary_file = output_folder / 'processing_summary.csv'
                summary_df = pd.DataFrame(results)
                summary_df.to_csv(summary_file, index=False)
                self.logger.logger.info(f"Processing summary saved to {summary_file}")
                    
        except Exception as e:
            self.logger.logger.error(f"Error in batch processing: {str(e)}")
            raise

def main():
    """Main entry point for command line usage"""
    processor = EnhancedJosephsonProcessor()
    processor.batch_process_files()

if __name__ == "__main__":
    main()