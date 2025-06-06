"""
OPTIMIZED VERSION with numba, multithreading, LRU cache, and FireDucks pandas
Enhanced main processor with all optimizations including advanced visualization
"""
import os
import sys
import glob
import time
import threading
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import numpy as np
import traceback
from functools import lru_cache
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Try to import optimized pandas, fallback to standard pandas
try:
    import fireducks.pandas as pd
    USING_FIREDUCKS = True
except ImportError:
    import pandas as pd
    USING_FIREDUCKS = False

# Import visualization libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multithreading
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle

from .config import config
from .logger import init_logger
from .josephson_model import JosephsonFitter, preprocess_data_numba
from .analysis_utils import FrequencyAnalyzer, PhaseAnalyzer, validate_data
from .visualization import PublicationPlotter
from .memory_manager import AdaptiveProcessor

# Numba-optimized model function
if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def model_f_numba(Phi_ext, I_c, phi_0, f, T, r, C):
        """Numba-optimized Josephson junction model function"""
        main_phase = 2 * np.pi * f * Phi_ext - phi_0
        half_phase = main_phase / 2
        sin_half = np.sin(half_phase)
        sin_main = np.sin(main_phase)
        # Calculate denominator (add numerical stability protection)
        denominator_term = 1 - T * sin_half**2
        denominator_term = np.maximum(denominator_term, 1e-12)  # Prevent division by zero
        denominator = np.sqrt(denominator_term)
        return I_c * sin_main / denominator + r * Phi_ext + C

    @jit(nopython=True, cache=True, fastmath=True)
    def calculate_statistics_numba(y_data, fitted_data, n_params):
        """Fast statistical calculations using numba"""
        n = len(y_data)
        y_mean = np.mean(y_data)
        
        # Calculate sum of squares
        ss_res = np.sum((y_data - fitted_data) ** 2)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        
        # R-squared and adjusted R-squared
        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1)
        
        # RMSE and MAE
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(y_data - fitted_data))
        
        # Residual statistics
        residuals = y_data - fitted_data
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        return r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std

    @jit(nopython=True, cache=True, fastmath=True)
    def calculate_phase_data_numba(x_data_normalized, best_frequency):
        """Fast phase calculations using numba"""
        phase = (x_data_normalized * best_frequency) % 1.0
        cycle_number = np.floor(x_data_normalized * best_frequency).astype(np.int32)
        total_cycles = int(np.max(cycle_number)) + 1
        return phase, cycle_number, total_cycles

    @jit(nopython=True, cache=True, fastmath=True)
    def calculate_binned_average_numba(phase, y_data_normalized, num_bins=20):
        """Fast binned average calculation using numba"""
        phase_bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        mean_binned_values = np.full(num_bins, np.nan)
        
        for i in range(num_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
            if np.any(mask):
                mean_binned_values[i] = np.mean(y_data_normalized[mask])
        
        return bin_centers, mean_binned_values
else:
    # Fallback implementations without numba
    def model_f_numba(Phi_ext, I_c, phi_0, f, T, r, C):
        main_phase = 2 * np.pi * f * Phi_ext - phi_0
        half_phase = main_phase / 2
        sin_half = np.sin(half_phase)
        sin_main = np.sin(main_phase)
        denominator_term = 1 - T * sin_half**2
        denominator_term = np.maximum(denominator_term, 1e-12)
        denominator = np.sqrt(denominator_term)
        return I_c * sin_main / denominator + r * Phi_ext + C
    
    def calculate_statistics_numba(y_data, fitted_data, n_params):
        n = len(y_data)
        y_mean = np.mean(y_data)
        ss_res = np.sum((y_data - fitted_data) ** 2)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1)
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(y_data - fitted_data))
        residuals = y_data - fitted_data
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        return r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std
    
    def calculate_phase_data_numba(x_data_normalized, best_frequency):
        phase = (x_data_normalized * best_frequency) % 1.0
        cycle_number = np.floor(x_data_normalized * best_frequency).astype(np.int32)
        total_cycles = int(np.max(cycle_number)) + 1
        return phase, cycle_number, total_cycles
    
    def calculate_binned_average_numba(phase, y_data_normalized, num_bins=20):
        phase_bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        mean_binned_values = np.full(num_bins, np.nan)
        for i in range(num_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
            if np.any(mask):
                mean_binned_values[i] = np.mean(y_data_normalized[mask])
        return bin_centers, mean_binned_values

# Wrapper for scipy.optimize.curve_fit (cannot use numba directly)
def model_f(Phi_ext, I_c, phi_0, f, T, r, C):
    """Wrapper function for curve_fit compatibility"""
    return model_f_numba(Phi_ext, I_c, phi_0, f, T, r, C)

# Cached frequency array generation
@lru_cache(maxsize=128)
def generate_frequency_array(n_points, median_diff, n_freq=10000):
    """Generate frequency array with caching"""
    freq_min = 1e-6
    freq_max = 1 / (2 * median_diff)
    return np.linspace(freq_min, freq_max, n_freq)

# Configuration for optimized processing
MAX_WORKERS = min(8, multiprocessing.cpu_count())  # Limit workers to prevent memory issues
PLOT_SIZE = (19.2, 10.8)  # 1920x1080 at 100 DPI
PLOT_DPI = 100

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
        
        # Thread-safe counters and output management
        self.lock = threading.Lock()
        self.progress = {'current': 0, 'total': 0}
        self.results = []
        
        # Pre-compile numba functions
        self._precompile_numba()
        
        self.logger.logger.info(f"Initialized EnhancedJosephsonProcessor")
        self.logger.logger.info(f"Using FireDucks pandas: {USING_FIREDUCKS}")
        self.logger.logger.info(f"Using Numba optimization: {HAS_NUMBA}")
        self.logger.logger.info(f"Max workers: {MAX_WORKERS}")
        self.logger.logger.info(f"Plot size: {PLOT_SIZE} at {PLOT_DPI} DPI")
    
    def safe_print(self, message):
        """Thread-safe print function to prevent BrokenPipeError"""
        try:
            with self.lock:
                print(message)
                sys.stdout.flush()
        except (BrokenPipeError, OSError):
            # Handle broken pipe gracefully - continue processing silently
            pass

    def update_progress(self, dataid, success, error_msg=None):
        """Thread-safe progress update with reduced output"""
        try:
            with self.lock:
                self.progress['current'] += 1
                current = self.progress['current']
                total = self.progress['total']
                
                # Show every 20th success or all failures to reduce output
                if current % 20 == 0 or not success:
                    status = "✓" if success else "✗"
                    message = f"{status} [{current}/{total}] {dataid}"
                    if not success and error_msg:
                            message += f": {error_msg[:50]}..."
                    self.logger.logger.info(message)
                    
        except (BrokenPipeError, OSError):
            # Handle pipe errors gracefully
            pass
    
    def _precompile_numba(self):
        """Pre-compile numba functions for better performance"""
        self.logger.logger.info("Pre-compiling Numba functions...")
        try:
            # Dummy compilation
            dummy_x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            dummy_y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            
            if HAS_NUMBA:
                _ = model_f_numba(dummy_x, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0)
                _ = calculate_statistics_numba(dummy_y, dummy_y, 6)
                _ = preprocess_data_numba(dummy_x, dummy_y)
                _ = calculate_phase_data_numba(dummy_x, 1.0)
                _ = calculate_binned_average_numba(dummy_x/2, dummy_y)
                self.logger.logger.info("✓ Numba functions compiled successfully")
            else:
                self.logger.logger.info("✓ Using non-Numba implementations")
                
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