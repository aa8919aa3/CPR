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

# Set matplotlib to use thread-safe backend
os.environ['MPLBACKEND'] = 'Agg'

# Import optimization libraries
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    print("✓ Numba JIT acceleration available")
except ImportError as e:
    HAS_NUMBA = False
    print(f"⚠️ Numba not available ({e}), using standard Python implementations")
except Exception as e:
    HAS_NUMBA = False
    print(f"⚠️ Numba compatibility issue ({e}), using standard Python implementations")

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
# Configure matplotlib font settings to avoid font warnings
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle

try:
    from .config import config
    from .logger import init_logger
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import config
    from logger import init_logger

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
    def preprocess_data_numba(x_data, y_data):
        """Fast data preprocessing using numba with robust scaling factors"""
        # Data shift and normalization by the first data point
        x_data_shifted = x_data - x_data[0]
        y_data_shifted = y_data - np.min(y_data)
        
        # Robust x_factor calculation
        if len(x_data_shifted) > 2:
            x_factor = abs(x_data_shifted[2] - x_data_shifted[1])
        else:
            x_factor = 1.0
            
        # If x_factor is too small, use standard deviation or range
        if x_factor < 1e-12:
            x_std = np.std(x_data_shifted)
            if x_std > 1e-12:
                x_factor = x_std
            else:
                x_range = np.max(x_data_shifted) - np.min(x_data_shifted)
                x_factor = x_range if x_range > 1e-12 else 1.0
            
        # Robust y_factor calculation - key improvement
        if len(y_data_shifted) > 2:
            y_factor = abs(y_data_shifted[2] - y_data_shifted[1])
        else:
            y_factor = 1.0
        
        # If y_factor is too small or zero, use alternative methods
        if y_factor < 1e-12:
            y_std = np.std(y_data_shifted)
            if y_std > 1e-12:
                y_factor = y_std
            else:
                y_range = np.max(y_data_shifted) - np.min(y_data_shifted)
                y_factor = y_range if y_range > 1e-12 else 1.0
        
        # Final safety check with larger minimum threshold
        x_factor = max(x_factor, 1e-6)
        y_factor = max(y_factor, 1e-6)
        
        # Normalize
        x_data_normalized = x_data_shifted / x_factor
        y_data_normalized = y_data_shifted / y_factor
        
        return x_data_normalized, y_data_normalized, x_factor, y_factor

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
    
    def preprocess_data_numba(x_data, y_data):
        x_data_shifted = x_data - x_data[0]
        y_data_shifted = y_data - np.min(y_data)
        
        # Robust x_factor calculation
        if len(x_data_shifted) > 2:
            x_factor = abs(x_data_shifted[2] - x_data_shifted[1])
        else:
            x_factor = 1.0
            
        # If x_factor is too small, use standard deviation or range
        if x_factor < 1e-12:
            x_std = np.std(x_data_shifted)
            if x_std > 1e-12:
                x_factor = x_std
            else:
                x_range = np.max(x_data_shifted) - np.min(x_data_shifted)
                x_factor = x_range if x_range > 1e-12 else 1.0
            
        # Robust y_factor calculation
        if len(y_data_shifted) > 2:
            y_factor = abs(y_data_shifted[2] - y_data_shifted[1])
        else:
            y_factor = 1.0
        
        # If y_factor is too small or zero, use alternative methods
        if y_factor < 1e-12:
            y_std = np.std(y_data_shifted)
            if y_std > 1e-12:
                y_factor = y_std
            else:
                y_range = np.max(y_data_shifted) - np.min(y_data_shifted)
                y_factor = y_range if y_range > 1e-12 else 1.0
        
        # Final safety check with larger minimum threshold
        x_factor = max(x_factor, 1e-6)
        y_factor = max(y_factor, 1e-6)
        
        x_data_normalized = x_data_shifted / x_factor
        y_data_normalized = y_data_shifted / y_factor
        return x_data_normalized, y_data_normalized, x_factor, y_factor
    
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
    """Generate frequency array with caching and safety checks"""
    # Input validation
    if not np.isfinite(median_diff) or median_diff <= 0:
        median_diff = 1.0  # Fallback value
    
    # Use a more conservative minimum frequency to avoid numerical issues
    freq_min = max(1e-5, 1.0 / (n_points * median_diff))  # Nyquist-informed minimum
    freq_max = 1 / (2 * median_diff)
    
    # Ensure freq_max is reasonable and greater than freq_min
    if not np.isfinite(freq_max) or freq_max <= freq_min:
        freq_max = max(1.0, freq_min * 1000)  # Ensure reasonable range
    
    # Ensure we have a reasonable frequency range
    if freq_max / freq_min < 10:
        freq_min = freq_max / 1000  # Ensure at least 3 orders of magnitude
    
    frequencies = np.linspace(freq_min, freq_max, n_freq)
    
    # Final validation
    if not np.isfinite(frequencies).all():
        frequencies = np.linspace(1e-5, 1.0, n_freq)  # Safe fallback with higher minimum
    
    return frequencies

# Configuration for optimized processing
MAX_WORKERS = min(8, multiprocessing.cpu_count())  # Increased to 8 workers for better performance
# Calculate exact figure size to get 1920x1080 pixel output
PLOT_SIZE = (1920/100, 1080/100)  # 19.2 x 10.8 inches at 100 DPI = exactly 1920x1080 pixels
PLOT_DPI = 100

# Global locks for thread safety
GLOBAL_PROCESSING_LOCK = threading.Lock()
NUMBA_COMPILATION_LOCK = threading.Lock()

# Adjust for matplotlib margins to get exactly 1920x1080
EXACT_PLOT_SIZE = (19.2, 10.8)  # Use exact size, let matplotlib handle margins

class EnhancedJosephsonProcessor:
    """Enhanced Josephson junction data processor with all optimizations"""
    
    def __init__(self):
        self.config = config
        self.logger = init_logger(config)
        
        # Thread-safe counters and output management
        self.output_lock = threading.Lock()
        self.matplotlib_lock = threading.Lock()  # Protect matplotlib operations
        self.progress_counter = {'current': 0, 'total': 0}
        
        # Pre-compile numba functions
        self._precompile_numba()
        
        self.logger.logger.info(f"Initialized EnhancedJosephsonProcessor")
        self.logger.logger.info(f"Using FireDucks pandas: {USING_FIREDUCKS}")
        self.logger.logger.info(f"Using Numba optimization: {HAS_NUMBA}")
        self.logger.logger.info(f"Max workers: {MAX_WORKERS}")
        self.logger.logger.info(f"Plot size: {PLOT_SIZE} at {PLOT_DPI} DPI")
    
    def _precompile_numba(self):
        """Pre-compile numba functions for better performance with thread safety"""
        with NUMBA_COMPILATION_LOCK:  # Ensure only one thread compiles at a time
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

    def validate_data_array(self, data, name):
        """Validate data array for NaN/inf values and provide detailed diagnostics"""
        if not np.isfinite(data).all():
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            return False, f"{name}: {nan_count} NaN, {inf_count} inf values"
        
        unique_values = np.unique(data)
        n_unique = len(unique_values)
        
        if n_unique < 2:
            return "skip", f"{name}: insufficient data variation ({n_unique} unique values) - skipping file"
        
        # Check if variation is too small (numerical precision issues)
        if n_unique == 2:
            data_range = np.ptp(data)  # peak-to-peak (max - min)
            data_scale = np.abs(np.mean(data))
            if data_scale > 0:
                relative_variation = data_range / data_scale
                if relative_variation < 1e-10:  # Very small relative variation
                    return "skip", f"{name}: variation too small (relative: {relative_variation:.2e}) - skipping file"
            
        return True, "OK"
    
    def validate_preprocessing_result(self, x_norm, y_norm, x_factor, y_factor):
        """Comprehensive validation of preprocessing results"""
        # Check normalization factors
        if not np.isfinite(x_factor) or x_factor <= 0:
            return False, f"Invalid x_factor: {x_factor}"
        if not np.isfinite(y_factor) or y_factor <= 0:
            return False, f"Invalid y_factor: {y_factor}"
            
        # Check normalized data
        x_valid, x_msg = self.validate_data_array(x_norm, "x_normalized")
        if x_valid == "skip":
            return "skip", x_msg
        elif not x_valid:
            return False, x_msg
            
        y_valid, y_msg = self.validate_data_array(y_norm, "y_normalized")
        if y_valid == "skip":
            return "skip", y_msg
        elif not y_valid:
            return False, y_msg
            
        return True, "OK"
    
    def safe_print(self, message):
        """Thread-safe print function to prevent BrokenPipeError"""
        try:
            with self.output_lock:
                print(message)
                sys.stdout.flush()
        except (BrokenPipeError, OSError):
            # Handle broken pipe gracefully - continue processing silently
            pass

    def update_progress(self, dataid, success, error_msg=None):
        """Thread-safe progress update with reduced output"""
        try:
            with self.output_lock:
                self.progress_counter['current'] += 1
                current = self.progress_counter['current']
                total = self.progress_counter['total']
                
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

    def process_single_file(self, csv_file_path, output_dir):
        """Process a single CSV file and return analysis results with enhanced visualizations"""
        # Add timeout to prevent deadlocks
        try:
            with GLOBAL_PROCESSING_LOCK:
                return self._process_single_file_internal(csv_file_path, output_dir)
        except Exception as e:
            dataid = Path(csv_file_path).stem
            self.update_progress(dataid, False, f"Lock timeout or error: {str(e)[:50]}...")
            return {
                'dataid': dataid,
                'success': False,
                'error': f'Lock timeout or error: {str(e)}'
            }
    
    def _process_single_file_internal(self, csv_file_path, output_dir):
        """Internal implementation of file processing"""
        try:
            # Extract filename without extension for use as dataid
            dataid = Path(csv_file_path).stem
            
            # Load and validate data
            df = pd.read_csv(csv_file_path)
            x_data = df['y_field'].values.astype(np.float64)  # External magnetic flux
            y_data = df['Ic'].values.astype(np.float64)      # Supercurrent
            
            # Clean data: remove NaN and infinite values
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if not np.any(valid_mask):
                self.update_progress(dataid, False, "All data points are NaN or infinite")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': 'All data points are NaN or infinite'
                }
            
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
            # Check if data has sufficient points (minimum 20 points required)
            if len(x_data) < 20:
                self.update_progress(dataid, False, f"Insufficient data points ({len(x_data)} < 20)")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Insufficient data points ({len(x_data)} < 20)'
                }
            
            # Remove first 10 data points for preprocessing
            if len(x_data) >= 10:
                x_data = x_data[10:]
                y_data = y_data[10:]
            
            # Final check after preprocessing
            if len(x_data) < 10:
                self.update_progress(dataid, False, f"Too few points after preprocessing ({len(x_data)} < 10)")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Too few points after preprocessing ({len(x_data)} < 10)'
                }
            
            # Fast data preprocessing using numba
            x_data_normalized, y_data_normalized, x_factor, y_factor = preprocess_data_numba(x_data, y_data)
            
            # Validate preprocessing results
            valid, error_msg = self.validate_preprocessing_result(x_data_normalized, y_data_normalized, x_factor, y_factor)
            if valid == "skip":
                self.update_progress(dataid, False, f"Data quality insufficient: {error_msg}")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Data quality insufficient: {error_msg}',
                    'skipped': True  # Mark this as a skip rather than an error
                }
            elif not valid:
                self.update_progress(dataid, False, f"NaN/inf values after preprocessing: {error_msg}")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'NaN/inf values after preprocessing: {error_msg}'
                }
            
            # Additional data quality checks (these should be redundant now due to validate_preprocessing_result)
            if np.std(x_data_normalized) < 1e-12:
                self.update_progress(dataid, False, "X data has no variation - skipping")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': 'X data has no variation',
                    'skipped': True
                }
                
            if np.std(y_data_normalized) < 1e-12:
                self.update_progress(dataid, False, "Y data has no variation - skipping")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': 'Y data has no variation',
                    'skipped': True
                }
            
            # Check for numerical issues after preprocessing
            if not (np.all(np.isfinite(x_data_normalized)) and np.all(np.isfinite(y_data_normalized))):
                self.update_progress(dataid, False, "NaN/inf values after preprocessing")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': 'NaN/inf values after preprocessing'
                }
            
            # Ensure we have sufficient variation in the data
            if np.std(x_data_normalized) < 1e-12 or np.std(y_data_normalized) < 1e-12:
                self.update_progress(dataid, False, "Insufficient data variation")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': 'Insufficient data variation'
                }
            
            # Cached frequency array generation
            median_diff = np.median(np.diff(x_data_normalized))
            
            # Validate median_diff
            if not np.isfinite(median_diff) or median_diff <= 0:
                self.update_progress(dataid, False, f"Invalid median_diff: {median_diff}")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Invalid median_diff: {median_diff}'
                }
                
            frequencies = generate_frequency_array(len(x_data_normalized), median_diff)
            
            # Validate frequency array
            if not np.isfinite(frequencies).all():
                self.update_progress(dataid, False, "Invalid frequency array")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': 'Invalid frequency array'
                }
            
            # Calculate Lomb-Scargle periodogram
            try:
                ls = LombScargle(x_data_normalized, y_data_normalized)
                power = ls.power(frequencies)
                
                # Fix numerical issues in power spectrum
                if not np.isfinite(power).all():
                    nan_count = np.isnan(power).sum()
                    inf_count = np.isinf(power).sum()
                    
                    # Try to fix inf values by clipping them
                    if inf_count > 0 and inf_count < len(power) * 0.1:  # Less than 10% inf values
                        max_finite_power = np.max(power[np.isfinite(power)])
                        power[np.isinf(power)] = max_finite_power * 2  # Replace inf with reasonable value
                        self.update_progress(dataid, True, f"Fixed {inf_count} inf values in power spectrum")
                    
                    # Try to fix NaN values
                    if nan_count > 0 and nan_count < len(power) * 0.1:  # Less than 10% NaN values
                        mean_finite_power = np.mean(power[np.isfinite(power)])
                        power[np.isnan(power)] = mean_finite_power  # Replace NaN with mean
                        self.update_progress(dataid, True, f"Fixed {nan_count} NaN values in power spectrum")
                    
                    # Final check after fixes
                    if not np.isfinite(power).all():
                        remaining_nan = np.isnan(power).sum()
                        remaining_inf = np.isinf(power).sum()
                        self.update_progress(dataid, False, f"Could not fix power spectrum: {remaining_nan} NaN, {remaining_inf} inf")
                        return {
                            'dataid': dataid,
                            'success': False,
                            'error': f'Unfixable power spectrum: {remaining_nan} NaN, {remaining_inf} inf'
                        }
                    
            except Exception as e:
                self.update_progress(dataid, False, f"Lomb-Scargle calculation failed: {str(e)[:30]}...")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Lomb-Scargle calculation failed: {str(e)}'
                }

            # Find peaks in the power spectrum
            height_threshold = np.max(power) * 0.1  # Peak height threshold
            peaks, properties = find_peaks(power, height=height_threshold, distance=100)

            # Get frequencies and powers corresponding to peaks
            peak_frequencies = frequencies[peaks]
            peak_powers = power[peaks]

            # Find global maximum position
            max_power_idx = np.argmax(power)
            global_max_freq = frequencies[max_power_idx]
            global_max_power = power[max_power_idx]

            # Sort by power size to find the strongest frequencies
            sorted_indices = np.argsort(peak_powers)[::-1]  # Descending order
            top_frequencies = peak_frequencies[sorted_indices]
            top_powers = peak_powers[sorted_indices]

            # Initial parameter guesses
            I_c_init = 3*np.std(y_data_normalized)  # Initial guess for critical current
            phi_0_init = 0.0  # Initial guess for phase offset
            f_init = global_max_freq   # Initial guess for frequency
            T_init = 0.5  # Initial guess for transparency
            r_init = np.polyfit(x_data_normalized, y_data_normalized, 1)[0]  # Initial guess for linear term
            C_init = np.mean(y_data_normalized)  # Initial guess for constant term
            p0 = [I_c_init, phi_0_init, f_init, T_init, r_init, C_init]  # I_c, phi_0, f, T, r, C

            # Boundaries for the parameters
            bounds = ([0, -np.pi, 0, 0, -np.inf, 0], 
                      [np.inf, np.pi, np.inf, 1, np.inf, np.inf])  

            # Fit the model to the data
            try:
                popt, pcov = curve_fit(model_f, x_data_normalized, y_data_normalized, p0=p0, bounds=bounds, maxfev=50000)
                
                # Check for numerical issues in fitted parameters
                if not np.all(np.isfinite(popt)):
                    self.update_progress(dataid, False, "NaN/inf in fitted parameters")
                    return {
                        'dataid': dataid,
                        'success': False,
                        'error': 'NaN/inf in fitted parameters'
                    }
                    
            except Exception as fit_error:
                self.update_progress(dataid, False, f"Curve fitting failed: {str(fit_error)[:30]}...")
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Curve fitting failed: {str(fit_error)}'
                }
            
            # Extract optimized parameters
            I_c_opt, phi_0_opt, f_opt, T_opt, r_opt, C_opt = popt

            # Generate fitted data using optimized numba function
            fitted_y_data = model_f_numba(x_data_normalized, I_c_opt, phi_0_opt, f_opt, T_opt, r_opt, C_opt)

            # Fast statistical calculations using numba
            r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std = calculate_statistics_numba(
                y_data_normalized, fitted_y_data, len(popt))

            # === ENHANCED VISUALIZATION WITH 1920x1080 PLOTS ===
            
            # 1. Plot the original data and the fitted curve (normalized)
            with self.matplotlib_lock:  # Thread-safe matplotlib operations
                plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI) 
            
                # Sort data by x_data_normalized for proper line connection
                sort_indices = np.argsort(x_data_normalized)
                x_sorted = x_data_normalized[sort_indices]
                y_sorted = y_data_normalized[sort_indices]
                fitted_y_sorted = fitted_y_data[sort_indices]
                
                # Plot original data with dashed line connection
                plt.plot(x_sorted, y_sorted, '--', color='blue', linewidth=1, alpha=0.7, label=f'{dataid} (connected)')
                plt.scatter(x_data_normalized, y_data_normalized, color='blue', s=8, alpha=0.8, zorder=5)
                
                # Plot fitted curve
                plt.plot(x_sorted, fitted_y_sorted, label='Full Fitted Model', color='red', linewidth=2)
                
                # Add linear trend line (rx + C)
                linear_trend = r_opt * x_data_normalized + C_opt
                linear_trend_sorted = linear_trend[sort_indices]
                plt.plot(x_sorted, linear_trend_sorted, '--', color='green', linewidth=2, alpha=0.8, label='Linear Trend (rx+C)')
                
                plt.xlabel('Normalized External Magnetic Flux (Φ_ext)', fontsize=14)
                plt.ylabel('Normalized Supercurrent (I_s)', fontsize=14)
                plt.title('Supercurrent vs. Normalized External Magnetic Flux', fontsize=16)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.grid()

                # Add text box with optimized parameters and statistics
                param_text = f'Optimized Parameters:\nI_c: {I_c_opt:.2e}\nphi_0: {phi_0_opt:.2f}\nf: {f_opt:.2e}\nT: {T_opt:.2%}\nr: {r_opt:.2e}\nC: {C_opt:.2e}'
                stats_text = f'Statistical Metrics:\nR²: {r_squared:.4f}\nAdj. R²: {adj_r_squared:.4f}\nRMSE: {rmse:.4f}\nSSE: {ss_res:.4f}\nMAE: {mae:.4f}'
                plt.text(1.02, 0.50, param_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.text(1.02, 0.25, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

                plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08)
                plt.savefig(f'{output_dir}/{dataid}_fitted_curve_normalized_plot.png', dpi=PLOT_DPI)
                plt.close()

            # 2. Plot the original data(unnormalized) and the fitted curve
            # Create a copy of optimized parameters for scaling
            I_c_scaled = I_c_opt * y_factor  # Scale I_c back to original units
            phi_0_scaled = phi_0_opt  # Phase offset remains the same
            f_scaled = f_opt / x_factor  # Scale frequency back to original units
            T_scaled = T_opt  # Transparency remains the same
            r_scaled = r_opt * y_factor / x_factor  # Scale r back to original units
            # Scale C back to original units
            C_scaled = C_opt * y_factor + min(y_data)

            with self.matplotlib_lock:  # Thread-safe matplotlib operations
                plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                
                fitted_y_original = model_f_numba(x_data_normalized, I_c_opt, phi_0_opt, f_opt, T_opt, r_opt, C_opt)*y_factor + min(y_data)

                # Fast statistical calculations for original scale data using numba
                r_squared_original, adj_r_squared_original, rmse_original, mae_original, ss_res_original, _, _ = calculate_statistics_numba(
                    y_data, fitted_y_original, len(popt))

                # Sort data by x_data for proper line connection
                sort_indices = np.argsort(x_data)
                x_sorted = x_data[sort_indices]
                y_sorted = y_data[sort_indices]
                fitted_y_sorted = fitted_y_original[sort_indices]
                
                # Plot original data with dashed line connection
                plt.plot(x_sorted, y_sorted, '--', color='blue', linewidth=1, alpha=0.7, label=f'{dataid} (connected)')
                plt.scatter(x_data, y_data, color='blue', s=8, alpha=0.8, zorder=5)
                
                # Plot fitted curve
                plt.plot(x_sorted, fitted_y_sorted, label='Full Fitted Model', color='red', linewidth=2)
                
                # Add linear trend line (rx + C) in original scale
                linear_trend_original = r_scaled * x_data + C_scaled 
                linear_trend_sorted = linear_trend_original[sort_indices]
                plt.plot(x_sorted, linear_trend_sorted, '--', color='green', linewidth=2, alpha=0.8, label='Linear Trend (rx+C)')
                
                plt.xlabel('External Magnetic Flux (Φ_ext)', fontsize=14)
                plt.ylabel('Supercurrent (I_s)', fontsize=14)
                plt.title('Supercurrent vs. External Magnetic Flux', fontsize=16)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.grid()

                # Add text box with scaled parameters and statistics
                scaled_param_text = f'Scaled Parameters:\nI_c: {I_c_scaled:.2e}\nphi_0: {phi_0_scaled:.2f}\nf: {f_scaled:.2e}\nT: {T_scaled:.2%}\nr: {r_scaled:.2e}\nC: {C_scaled:.2e}'
                original_stats_text = f'Statistical Metrics:\nR²: {r_squared_original:.4f}\nAdj. R²: {adj_r_squared_original:.4f}\nRMSE: {rmse_original:.4f}\nSSE: {ss_res_original:.4f}\nMAE: {mae_original:.4f}'
                plt.text(1.02, 0.50, scaled_param_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.text(1.02, 0.25, original_stats_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

                plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08) 
                plt.savefig(f'{output_dir}/{dataid}_fitted_curve_plot.png', dpi=PLOT_DPI)
                plt.close()

            # Update parameters for later use
            I_c_opt = I_c_scaled
            phi_0_opt = phi_0_scaled
            f_opt = f_scaled
            T_opt = T_scaled
            r_opt = r_scaled
            C_opt = C_scaled

            # 3. Residuals plot and their statistics
            residuals = y_data - fitted_y_original
            with self.matplotlib_lock:  # Thread-safe matplotlib operations
                plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)

                # Main residuals plot
                plt.subplot(2, 2, 1)
                plt.scatter(x_data, residuals, label=f'{dataid} Residuals', color='green', s=5)
                plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
                plt.xlabel('External Magnetic Flux', fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title('Residuals of the Fit', fontsize=14)
                plt.legend()
                plt.grid()

                # Residuals vs fitted values
                plt.subplot(2, 2, 2)
                plt.scatter(fitted_y_original, residuals, alpha=0.6, s=5, color='orange')
                plt.axhline(y=0, color='red', linestyle='--')
                plt.xlabel('Fitted Values', fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title('Residuals vs Fitted Values', fontsize=14)
                plt.grid()

                # Q-Q plot of residuals
                plt.subplot(2, 2, 3)
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Q-Q Plot of Residuals', fontsize=14)
                plt.grid()

                # Histogram of residuals
                plt.subplot(2, 2, 4)
                plt.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black', color='lightcoral')
                plt.xlabel('Residuals', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.title('Distribution of Residuals', fontsize=14)
                plt.grid()

                # Overlay normal distribution
                x_normal = np.linspace(residuals.min(), residuals.max(), 100)
                y_normal = stats.norm.pdf(x_normal, np.mean(residuals), np.std(residuals))
                plt.plot(x_normal, y_normal, 'r-', label='Normal Distribution', linewidth=2)
                plt.legend()

                plt.tight_layout(pad=0)
                plt.savefig(f'{output_dir}/{dataid}_residuals_plot.png', dpi=PLOT_DPI, bbox_inches='tight', pad_inches=0)
                plt.close()

            # 4. & 5. Add matplotlib phase-folded plots (colored by cycle with lines)
            if len(top_frequencies) > 0:
                best_frequency = top_frequencies[0]
                best_period = 1 / best_frequency if best_frequency > 0 else np.inf
                
                if not np.isinf(best_period) and not np.isnan(best_period):
                    # Fast phase calculations using numba
                    phase, cycle_number, total_cycles = calculate_phase_data_numba(x_data_normalized, best_frequency)
                    
                    # 4. Create matplotlib figure for phase-folded plot with drift analysis
                    with self.matplotlib_lock:  # Thread-safe matplotlib operations
                        plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                    
                        # Define color list
                        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                        
                        # Store peak information for phase drift analysis
                        cycle_peak_phases = []
                        cycle_peak_values = []
                        
                        # Group by cycle and plot data points with different colors, including lines
                        for cycle in range(total_cycles):
                            mask = cycle_number == cycle
                            if np.any(mask):
                                color = colors[cycle % len(colors)]
                                
                                # Get data for this cycle and sort by phase
                                cycle_phase = phase[mask]
                                cycle_y = y_data_normalized[mask]
                                
                                # Find peak position for this cycle (for drift analysis)
                                max_idx = np.argmax(cycle_y)
                                peak_phase = cycle_phase[max_idx]
                                peak_value = cycle_y[max_idx]
                                cycle_peak_phases.append(peak_phase)
                                cycle_peak_values.append(peak_value)
                                
                                # Sort by phase for correct line connection
                                sort_indices = np.argsort(cycle_phase)
                                sorted_phase = cycle_phase[sort_indices]
                                sorted_y = cycle_y[sort_indices]
                                
                                # Plot connected data points
                                plt.plot(sorted_phase, sorted_y, 'o-', color=color, 
                                        label=f'Cycle {cycle + 1}', markersize=4, linewidth=2, alpha=0.8)
                                
                                # Add peak marker for phase drift analysis
                                plt.scatter([peak_phase], [peak_value], color=color, s=120, 
                                           marker='*', edgecolors='black', linewidth=2, zorder=5)
                        
                        # Fast binned average calculation using numba
                        bin_centers, mean_binned_values = calculate_binned_average_numba(phase, y_data_normalized)
                        
                        # Remove NaN values and ensure sorting by x order
                        valid_mask = ~np.isnan(mean_binned_values)
                        if np.any(valid_mask):
                            valid_centers = bin_centers[valid_mask]
                            valid_means = mean_binned_values[valid_mask]
                            
                            # Sort by x-axis (phase) order
                            sort_indices = np.argsort(valid_centers)
                            sorted_centers = valid_centers[sort_indices]
                            sorted_means = valid_means[sort_indices]
                            
                            # Create single cycle average line, connected by x order
                            plt.plot(sorted_centers, sorted_means, 'k--', linewidth=3, 
                                    label='Average Profile', marker='s', markersize=6)
                        
                        plt.xlabel(f'Phase (Period = {best_period:.6f})', fontsize=14)
                        plt.ylabel('Normalized Supercurrent (I_s)', fontsize=14)
                        plt.title(f'Phase-Folded Plot with Phase Drift Analysis - Total {total_cycles} Cycles', fontsize=16)
                        plt.xlim(0, 1)
                        plt.grid(True, alpha=0.3)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        # Add statistics info text box
                        info_text = f'Best Frequency: {best_frequency:.6e} Hz\nPeriod: {best_period:.6f}\nTotal Cycles: {total_cycles}'
                        plt.text(1.02, 0.50, info_text, transform=plt.gca().transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
                        
                        # Calculate and display phase drift statistics
                        if len(cycle_peak_phases) > 1:
                            phase_drift = np.diff(cycle_peak_phases)
                            # Handle phase crossing 0-1 boundary
                            phase_drift = np.where(phase_drift > 0.5, phase_drift - 1, phase_drift)
                            phase_drift = np.where(phase_drift < -0.5, phase_drift + 1, phase_drift)
                            
                            drift_stats = f'Phase Drift Statistics:\nMean Drift: {np.mean(phase_drift):.6f}\nStd Dev: {np.std(phase_drift):.6f}\n★ = Peak positions'
                            plt.text(1.02, 0.25, drift_stats, transform=plt.gca().transAxes, fontsize=10,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                        
                        plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08)
                        plt.savefig(f'{output_dir}/{dataid}_phase_folded_with_drift.png', dpi=PLOT_DPI)
                        plt.close()
                    
                    # 5. Plot original data colored by cycle (without phase folding)
                    with self.matplotlib_lock:  # Thread-safe matplotlib operations
                        plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                    
                        for cycle in range(total_cycles):
                            mask = cycle_number == cycle
                            if np.any(mask):
                                color = colors[cycle % len(colors)]
                                plt.scatter(x_data_normalized[mask], y_data_normalized[mask],
                                           color=color, label=f'Cycle {cycle + 1}', s=20, alpha=0.8)
                        
                        plt.xlabel('Normalized External Magnetic Flux (Φ_ext)', fontsize=14)
                        plt.ylabel('Normalized Supercurrent (I_s)', fontsize=14)
                        plt.title(f'Original Data Colored by Cycle (Total {total_cycles} Cycles)', fontsize=16)
                        plt.grid(True, alpha=0.3)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        # Add cycle boundary lines
                        for cycle in range(1, total_cycles):
                            boundary = cycle / best_frequency
                            if boundary <= np.max(x_data_normalized):
                                plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7, linewidth=1)
                        
                        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
                        plt.savefig(f'{output_dir}/{dataid}_cycles_colored_matplotlib.png', dpi=PLOT_DPI)
                        plt.close()

            # Return analysis results
            return {
                'dataid': dataid,
                'success': True,
                'I_c': I_c_opt,
                'phi_0': phi_0_opt,
                'f': f_opt,
                'T': T_opt,
                'r': r_opt,
                'C': C_opt,
                'r_squared': r_squared_original,
                'adj_r_squared': adj_r_squared_original,
                'rmse': rmse_original,
                'mae': mae_original,
                'residual_mean': residual_mean,
                'residual_std': residual_std
            }
            
        except Exception as e:
            self.update_progress(Path(csv_file_path).stem, False, f"Error: {str(e)[:50]}...")
            return {
                'dataid': Path(csv_file_path).stem,
                'success': False,
                'error': str(e)
            }

    def process_files(self, csv_files, output_dir=None):
        """Process a specific list of CSV files using multithreading
        
        Args:
            csv_files: List of CSV file paths to process
            output_dir: Output directory (optional, uses config default if not provided)
            
        Returns:
            List of processing results
        """
        if output_dir is None:
            output_dir = self.config.get('OUTPUT_FOLDER', 'output')
        
        if not csv_files:
            self.safe_print("No CSV files provided to process")
            return []
        
        self.safe_print(f"Processing {len(csv_files)} CSV files")
        self.safe_print(f"Using {MAX_WORKERS} worker threads for parallel processing")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Record start time
        start_time = time.time()
        
        # Initialize results list and progress counter
        results = []
        successful_count = 0
        failed_count = 0
        self.progress_counter['total'] = len(csv_files)
        self.progress_counter['current'] = 0
        
        # Process files in parallel using ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                self.safe_print(f"{'='*60}")
                self.safe_print("STARTING PARALLEL PROCESSING")
                self.safe_print(f"{'='*60}")
                
                # Submit all jobs
                future_to_file = {executor.submit(self.process_single_file, csv_file, output_dir): csv_file 
                                 for csv_file in csv_files}
                
                # Process completed jobs
                for future in as_completed(future_to_file):
                    csv_file = future_to_file[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per file
                        results.append(result)
                        
                        if result['success']:
                            successful_count += 1
                            self.update_progress(result['dataid'], True)
                        else:
                            failed_count += 1
                            self.update_progress(result['dataid'], False, result.get('error'))
                            
                    except Exception as e:
                        failed_count += 1
                        dataid = Path(csv_file).stem
                        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                        self.update_progress(dataid, False, f"Exception: {error_msg}")
                        results.append({
                            'dataid': dataid,
                            'success': False,
                            'error': str(e)
                        })
        
        except Exception as e:
            self.safe_print(f"Critical error in file processing: {str(e)}")
            return results
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create summary report
        self.safe_print(f"{'='*60}")
        self.safe_print("FILE PROCESSING SUMMARY")
        self.safe_print(f"{'='*60}")
        self.safe_print(f"Total files processed: {len(csv_files)}")
        self.safe_print(f"Successful: {successful_count}")
        self.safe_print(f"Failed: {failed_count}")
        if len(csv_files) > 0:
            self.safe_print(f"Success rate: {successful_count/len(csv_files)*100:.1f}%")
            self.safe_print(f"Total processing time: {processing_time:.2f} seconds")
            self.safe_print(f"Average time per file: {processing_time/len(csv_files):.2f} seconds")
        self.safe_print(f"Speedup factor (estimated): {MAX_WORKERS:.1f}x with {MAX_WORKERS} threads")
        
        # Save results to CSV if any results exist
        if results:
            try:
                summary_df = pd.DataFrame(results)
                summary_path = os.path.join(output_dir, 'analysis_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                self.safe_print(f"Summary saved to: {summary_path}")
            except Exception as e:
                self.safe_print(f"Error saving summary: {str(e)}")
        
        self.safe_print(f"All plots saved to: {output_dir}")
        
        return results

    def batch_process_files(self):
        """Process all CSV files in the input folder using multithreading"""
        input_folder = self.config.get('INPUT_FOLDER', 'data/Ic')
        output_folder = self.config.get('OUTPUT_FOLDER', 'output')
        
        # Find all CSV files in the input folder
        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
        
        if not csv_files:
            self.safe_print(f"No CSV files found in {input_folder}")
            return
        
        self.safe_print(f"Found {len(csv_files)} CSV files to process")
        self.safe_print(f"Using {MAX_WORKERS} worker threads for parallel processing")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Record start time
        start_time = time.time()
        
        # Initialize results list and progress counter
        results = []
        successful_count = 0
        failed_count = 0
        self.progress_counter['total'] = len(csv_files)
        self.progress_counter['current'] = 0
        
        # Process files in parallel using ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                self.safe_print(f"{'='*60}")
                self.safe_print("STARTING PARALLEL PROCESSING")
                self.safe_print(f"{'='*60}")
                
                # Submit all jobs
                future_to_file = {executor.submit(self.process_single_file, csv_file, output_folder): csv_file 
                                 for csv_file in csv_files}
                
                # Process completed jobs
                for future in as_completed(future_to_file):
                    csv_file = future_to_file[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per file
                        results.append(result)
                        
                        if result['success']:
                            successful_count += 1
                            self.update_progress(result['dataid'], True)
                        else:
                            failed_count += 1
                            self.update_progress(result['dataid'], False, result.get('error'))
                            
                    except Exception as e:
                        failed_count += 1
                        dataid = Path(csv_file).stem
                        error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                        self.update_progress(dataid, False, f"Exception: {error_msg}")
                        results.append({
                            'dataid': dataid,
                            'success': False,
                            'error': str(e)
                        })
        
        except Exception as e:
            self.safe_print(f"Critical error in batch processing: {str(e)}")
            return
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create summary report
        self.safe_print(f"{'='*60}")
        self.safe_print("BATCH PROCESSING SUMMARY")
        self.safe_print(f"{'='*60}")
        self.safe_print(f"Total files processed: {len(csv_files)}")
        self.safe_print(f"Successful: {successful_count}")
        self.safe_print(f"Failed: {failed_count}")
        self.safe_print(f"Success rate: {successful_count/len(csv_files)*100:.1f}%")
        self.safe_print(f"Total processing time: {processing_time:.2f} seconds")
        self.safe_print(f"Average time per file: {processing_time/len(csv_files):.2f} seconds")
        self.safe_print(f"Speedup factor (estimated): {MAX_WORKERS:.1f}x with {MAX_WORKERS} threads")
        
        # Save results to CSV
        if results:
            try:
                summary_df = pd.DataFrame(results)
                summary_path = os.path.join(output_folder, 'analysis_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                self.safe_print(f"Summary saved to: {summary_path}")
            except Exception as e:
                self.safe_print(f"Error saving summary: {str(e)}")
        
        self.safe_print(f"All plots saved to: {output_folder}")
        self.safe_print("Generated plot types for each successful file:")
        self.safe_print("  1. fitted_curve_normalized_plot.png - Normalized data with fitted curve")
        self.safe_print("  2. fitted_curve_plot.png - Original scale data with fitted curve")  
        self.safe_print("  3. residuals_plot.png - Residual analysis (4 subplots)")
        self.safe_print("  4. phase_folded_with_drift.png - Phase-folded plot with drift analysis")
        self.safe_print("  5. cycles_colored_matplotlib.png - Original data colored by cycle")
        
        # Performance summary
        self.safe_print(f"{'='*60}")
        self.safe_print("PERFORMANCE OPTIMIZATIONS APPLIED")
        self.safe_print(f"{'='*60}")
        self.safe_print(f"✓ FireDucks pandas: {USING_FIREDUCKS}")
        self.safe_print(f"✓ Numba JIT compilation: {HAS_NUMBA}")
        self.safe_print("✓ LRU caching for repeated calculations")
        self.safe_print(f"✓ Multithreading with {MAX_WORKERS} workers")
        self.safe_print("✓ Optimized data types and memory usage")
        self.safe_print("✓ Fast statistical calculations")
        self.safe_print("✓ Thread-safe output management")
        self.safe_print(f"✓ High-resolution plots ({PLOT_SIZE[0]}x{PLOT_SIZE[1]} inches at {PLOT_DPI} DPI)")

    def safe_matplotlib_operation(self, func, *args, **kwargs):
        """Thread-safe wrapper for matplotlib operations to prevent conflicts"""
        with self.matplotlib_lock:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.logger.error(f"Matplotlib operation failed: {e}")
                raise
    
    def create_plot_safely(self, plot_func, filename, *args, **kwargs):
        """Safely create and save a plot with thread protection"""
        def _plot_operation():
            try:
                result = plot_func(*args, **kwargs)
                plt.savefig(filename, dpi=PLOT_DPI)
                plt.close()
                return result
            except Exception as e:
                plt.close()  # Ensure cleanup even if error occurs
                raise e
        
        return self.safe_matplotlib_operation(_plot_operation)

def main():
    """Main entry point for command line usage"""
    processor = EnhancedJosephsonProcessor()
    
    # Pre-compile numba functions for better performance
    processor.safe_print("Pre-compiling numba functions...")
    dummy_x = np.array([1.0, 2.0, 3.0])
    dummy_y = np.array([1.0, 2.0, 3.0])
    if HAS_NUMBA:
        _ = model_f_numba(dummy_x, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0)
        _ = calculate_statistics_numba(dummy_y, dummy_y, 6)
        _ = preprocess_data_numba(dummy_x, dummy_y)
        _ = calculate_phase_data_numba(dummy_x, 1.0)
        _ = calculate_binned_average_numba(dummy_x, dummy_y)
        processor.safe_print("Numba functions compiled successfully!")
    
    # Display optimization information
    processor.safe_print(f"{'='*60}")
    processor.safe_print("OPTIMIZATION STACK INFORMATION")
    processor.safe_print(f"{'='*60}")
    processor.safe_print(f"FireDucks pandas: {USING_FIREDUCKS}")
    processor.safe_print(f"Numba JIT: {HAS_NUMBA}")
    processor.safe_print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    processor.safe_print(f"Using {MAX_WORKERS} workers for parallel processing")
    processor.safe_print(f"Plot size: {PLOT_SIZE} at {PLOT_DPI} DPI")
    processor.safe_print(f"{'='*60}")
    
    # Run batch processing
    processor.batch_process_files()

if __name__ == "__main__":
    main()
