"""
Analysis utilities for frequency analysis and phase calculations
"""
import numpy as np
import numba
from numba import jit
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from functools import lru_cache
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

@lru_cache(maxsize=128)
def generate_frequency_array(n_points: int, median_diff: float, n_freq: int = 10000) -> np.ndarray:
    """Generate frequency array with caching for repeated use"""
    freq_min = 1e-6
    freq_max = 1 / (2 * median_diff) if median_diff > 0 else 1.0
    return np.linspace(freq_min, freq_max, n_freq)

@jit(nopython=True, cache=True, fastmath=True)
def calculate_phase_data_numba(x_data: np.ndarray, frequency: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fast phase calculations using Numba
    
    Returns:
    --------
    phase, cycle_number, total_cycles
    """
    if frequency <= 0:
        # Handle invalid frequency
        phase = np.zeros_like(x_data)
        cycle_number = np.zeros(len(x_data), dtype=np.int32)
        return phase, cycle_number, 1
    
    phase = (x_data * frequency) % 1.0
    cycle_number = np.floor(x_data * frequency).astype(np.int32)
    total_cycles = int(np.max(cycle_number)) + 1
    
    return phase, cycle_number, total_cycles

@jit(nopython=True, cache=True, fastmath=True)
def calculate_binned_average_numba(phase: np.ndarray, y_data: np.ndarray, 
                                  num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast binned average calculation using Numba
    
    Returns:
    --------
    bin_centers, mean_binned_values
    """
    phase_bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    mean_binned_values = np.full(num_bins, np.nan)
    
    for i in range(num_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.any(mask):
            values = y_data[mask]
            mean_binned_values[i] = np.mean(values)
    
    return bin_centers, mean_binned_values

@jit(nopython=True, cache=True, fastmath=True)
def calculate_phase_drift_stats(cycle_peak_phases: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate phase drift statistics
    
    Returns:
    --------
    mean_drift, std_drift, max_drift
    """
    if len(cycle_peak_phases) < 2:
        return 0.0, 0.0, 0.0
    
    phase_diffs = np.diff(cycle_peak_phases)
    
    # Handle phase wrapping (0-1 boundary crossings)
    phase_diffs = np.where(phase_diffs > 0.5, phase_diffs - 1, phase_diffs)
    phase_diffs = np.where(phase_diffs < -0.5, phase_diffs + 1, phase_diffs)
    
    mean_drift = np.mean(phase_diffs)
    std_drift = np.std(phase_diffs)
    max_drift = np.max(np.abs(phase_diffs))
    
    return mean_drift, std_drift, max_drift

class FrequencyAnalyzer:
    """Enhanced frequency analysis with peak detection and validation"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_periodogram(self, x_data: np.ndarray, y_data: np.ndarray) -> dict:
        """
        Perform Lomb-Scargle periodogram analysis
        
        Returns:
        --------
        Dictionary with frequency analysis results
        """
        try:
            # Calculate median difference for frequency range
            x_diffs = np.diff(x_data)
            median_diff = np.median(x_diffs[x_diffs > 0]) if np.any(x_diffs > 0) else 1.0
            
            # Generate frequency array
            n_freq = self.config.get('FREQUENCY_POINTS', 10000)
            frequencies = generate_frequency_array(len(x_data), median_diff, n_freq)
            
            # Calculate Lomb-Scargle periodogram
            ls = LombScargle(x_data, y_data)
            power = ls.power(frequencies)
            
            # Find peaks
            height_threshold = np.max(power) * self.config.get('PEAK_HEIGHT_THRESHOLD', 0.1)
            peak_distance = self.config.get('PEAK_DISTANCE', 100)
            
            peaks, properties = find_peaks(
                power, 
                height=height_threshold, 
                distance=peak_distance
            )
            
            # Extract peak information
            peak_frequencies = frequencies[peaks]
            peak_powers = power[peaks]
            
            # Global maximum
            max_power_idx = np.argmax(power)
            global_max_freq = frequencies[max_power_idx]
            global_max_power = power[max_power_idx]
            
            # Sort peaks by power
            if len(peak_powers) > 0:
                sorted_indices = np.argsort(peak_powers)[::-1]
                top_frequencies = peak_frequencies[sorted_indices]
                top_powers = peak_powers[sorted_indices]
                best_frequency = top_frequencies[0]
            else:
                top_frequencies = np.array([global_max_freq])
                top_powers = np.array([global_max_power])
                best_frequency = global_max_freq
            
            return {
                'success': True,
                'frequencies': frequencies,
                'power': power,
                'best_frequency': best_frequency,
                'best_period': 1 / best_frequency if best_frequency > 0 else np.inf,
                'peak_frequencies': peak_frequencies,
                'peak_powers': peak_powers,
                'top_frequencies': top_frequencies,
                'top_powers': top_powers,
                'global_max_freq': global_max_freq,
                'global_max_power': global_max_power
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'best_frequency': 1.0,  # Fallback value
                'best_period': 1.0
            }

class PhaseAnalyzer:
    """Phase analysis and drift detection"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_phase_folded_data(self, x_data: np.ndarray, y_data: np.ndarray, 
                                frequency: float) -> dict:
        """
        Analyze phase-folded data with drift detection
        
        Returns:
        --------
        Dictionary with phase analysis results
        """
        try:
            if frequency <= 0:
                return {'success': False, 'error': 'Invalid frequency'}
            
            # Calculate phase data
            phase, cycle_number, total_cycles = calculate_phase_data_numba(x_data, frequency)
            
            # Calculate binned average
            num_bins = self.config.get('PHASE_BINS', 20)
            bin_centers, mean_binned_values = calculate_binned_average_numba(
                phase, y_data, num_bins
            )
            
            # Analyze phase drift
            cycle_peak_phases = []
            cycle_peak_values = []
            cycle_data = {}
            
            for cycle in range(total_cycles):
                mask = cycle_number == cycle
                if np.any(mask):
                    cycle_phase = phase[mask]
                    cycle_y = y_data[mask]
                    
                    # Find peak in this cycle
                    if len(cycle_y) > 0:
                        max_idx = np.argmax(cycle_y)
                        peak_phase = cycle_phase[max_idx]
                        peak_value = cycle_y[max_idx]
                        
                        cycle_peak_phases.append(peak_phase)
                        cycle_peak_values.append(peak_value)
                        
                        # Store cycle data
                        sort_indices = np.argsort(cycle_phase)
                        cycle_data[cycle] = {
                            'phase': cycle_phase[sort_indices],
                            'values': cycle_y[sort_indices],
                            'peak_phase': peak_phase,
                            'peak_value': peak_value
                        }
            
            # Calculate drift statistics
            if len(cycle_peak_phases) > 1:
                drift_stats = calculate_phase_drift_stats(np.array(cycle_peak_phases))
                mean_drift, std_drift, max_drift = drift_stats
            else:
                mean_drift = std_drift = max_drift = 0.0
            
            return {
                'success': True,
                'phase': phase,
                'cycle_number': cycle_number,
                'total_cycles': total_cycles,
                'bin_centers': bin_centers,
                'mean_binned_values': mean_binned_values,
                'cycle_data': cycle_data,
                'cycle_peak_phases': np.array(cycle_peak_phases),
                'cycle_peak_values': np.array(cycle_peak_values),
                'drift_stats': {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'max_drift': max_drift
                }
            }
            
        except Exception as e:
            return {
                'success': False, 
                'error': str(e)
            }

def validate_data(x_data: np.ndarray, y_data: np.ndarray, min_points: int = 20) -> dict:
    """
    Validate input data quality
    
    Returns:
    --------
    Dictionary with validation results
    """
    issues = []
    
    # Check data length
    if len(x_data) != len(y_data):
        issues.append("X and Y data lengths don't match")
    
    if len(x_data) < min_points:
        issues.append(f"Insufficient data points ({len(x_data)} < {min_points})")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)):
        issues.append("Data contains NaN values")
    
    if np.any(np.isinf(x_data)) or np.any(np.isinf(y_data)):
        issues.append("Data contains infinite values")
    
    # Check data range
    if np.std(x_data) < 1e-12:
        issues.append("X data has insufficient variation")
    
    if np.std(y_data) < 1e-12:
        issues.append("Y data has insufficient variation")
    
    # Check for monotonicity in x_data (should be mostly increasing)
    x_diffs = np.diff(x_data)
    if np.sum(x_diffs > 0) / len(x_diffs) < 0.5:
        issues.append("X data is not primarily monotonic")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_points': len(x_data),
        'x_range': (np.min(x_data), np.max(x_data)),
        'y_range': (np.min(y_data), np.max(y_data))
    }