#!/usr/bin/env python3
"""
Debug script for power spectrum calculation issues
"""

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

def debug_file(filepath):
    """Debug a specific file for power spectrum issues"""
    print(f"\n=== Debugging {filepath} ===")
    
    # Load data
    try:
        data = pd.read_csv(filepath)
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Extract data
        x_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values
        
        print(f"X range: {x_data.min():.6e} to {x_data.max():.6e}")
        print(f"Y range: {y_data.min():.6e} to {y_data.max():.6e}")
        print(f"X std: {np.std(x_data):.6e}")
        print(f"Y std: {np.std(y_data):.6e}")
        
        # Check for NaN/inf
        print(f"X NaN count: {np.isnan(x_data).sum()}")
        print(f"Y NaN count: {np.isnan(y_data).sum()}")
        print(f"X inf count: {np.isinf(x_data).sum()}")
        print(f"Y inf count: {np.isinf(y_data).sum()}")
        
        # Data preprocessing
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) >= 10:
            x_data = x_data[10:]
            y_data = y_data[10:]
        
        print(f"Data points after preprocessing: {len(x_data)}")
        
        # Normalize data
        x_data_shifted = x_data - x_data[0]
        y_data_shifted = y_data - np.min(y_data)
        
        # Robust factor calculation
        if len(x_data_shifted) > 2:
            x_factor = abs(x_data_shifted[2] - x_data_shifted[1])
        else:
            x_factor = 1.0
            
        if x_factor < 1e-12:
            x_std = np.std(x_data_shifted)
            if x_std > 1e-12:
                x_factor = x_std
            else:
                x_range = np.max(x_data_shifted) - np.min(x_data_shifted)
                x_factor = x_range if x_range > 1e-12 else 1.0
        
        if len(y_data_shifted) > 2:
            y_factor = abs(y_data_shifted[2] - y_data_shifted[1])
        else:
            y_factor = 1.0
            
        if y_factor < 1e-12:
            y_std = np.std(y_data_shifted)
            if y_std > 1e-12:
                y_factor = y_std
            else:
                y_range = np.max(y_data_shifted) - np.min(y_data_shifted)
                y_factor = y_range if y_range > 1e-12 else 1.0
        
        x_data_normalized = x_data_shifted / x_factor
        y_data_normalized = y_data_shifted / y_factor
        
        print(f"X factor: {x_factor:.6e}")
        print(f"Y factor: {y_factor:.6e}")
        print(f"Normalized X range: {x_data_normalized.min():.6e} to {x_data_normalized.max():.6e}")
        print(f"Normalized Y range: {y_data_normalized.min():.6e} to {y_data_normalized.max():.6e}")
        
        # Check for issues in normalized data
        print(f"Normalized X NaN count: {np.isnan(x_data_normalized).sum()}")
        print(f"Normalized Y NaN count: {np.isnan(y_data_normalized).sum()}")
        print(f"Normalized X inf count: {np.isinf(x_data_normalized).sum()}")
        print(f"Normalized Y inf count: {np.isinf(y_data_normalized).sum()}")
        
        # Generate frequency array
        x_diff = np.diff(x_data_normalized)
        median_diff = np.median(x_diff)
        print(f"Median diff: {median_diff:.6e}")
        
        if not np.isfinite(median_diff) or median_diff <= 0:
            median_diff = 1.0
            print(f"Using fallback median_diff: {median_diff}")
        
        freq_min = 1e-6
        freq_max = 1 / (2 * median_diff)
        
        if not np.isfinite(freq_max) or freq_max <= freq_min:
            freq_max = 1.0
            print(f"Using fallback freq_max: {freq_max}")
        
        print(f"Frequency range: {freq_min:.6e} to {freq_max:.6e}")
        
        frequencies = np.linspace(freq_min, freq_max, 10000)  # Use same size as main processor
        
        # Validate frequency array
        print(f"Frequency array NaN count: {np.isnan(frequencies).sum()}")
        print(f"Frequency array inf count: {np.isinf(frequencies).sum()}")
        
        # Test Lomb-Scargle calculation
        try:
            print("Creating LombScargle object...")
            ls = LombScargle(x_data_normalized, y_data_normalized)
            
            print("Calculating power spectrum...")
            power = ls.power(frequencies)
            
            print(f"Power spectrum shape: {power.shape}")
            print(f"Power range: {power.min():.6e} to {power.max():.6e}")
            print(f"Power NaN count: {np.isnan(power).sum()}")
            print(f"Power inf count: {np.isinf(power).sum()}")
            
            if np.any(np.isinf(power)):
                inf_indices = np.where(np.isinf(power))[0]
                print(f"Inf values at indices: {inf_indices[:10]}")  # Show first 10
                print(f"Corresponding frequencies: {frequencies[inf_indices[:10]]}")
            
            if np.any(np.isnan(power)):
                nan_indices = np.where(np.isnan(power))[0]
                print(f"NaN values at indices: {nan_indices[:10]}")  # Show first 10
                print(f"Corresponding frequencies: {frequencies[nan_indices[:10]]}")
                
            # Try to find peaks only if power spectrum is valid
            if np.all(np.isfinite(power)):
                print("Power spectrum is valid, finding peaks...")
                height_threshold = np.max(power) * 0.1
                peaks, properties = find_peaks(power, height=height_threshold, distance=10)
                print(f"Found {len(peaks)} peaks")
            else:
                print("Power spectrum contains NaN/inf values, cannot find peaks")
                
        except Exception as e:
            print(f"Error in Lomb-Scargle calculation: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the problematic file
    problem_file = "/Users/albert-mac/Code/GitHub/CPR/data/Ic/491Ic+.csv"
    debug_file(problem_file)
    
    # Test a few other files for comparison
    test_files = [
        "/Users/albert-mac/Code/GitHub/CPR/data/Ic/100Ic-.csv",
        "/Users/albert-mac/Code/GitHub/CPR/data/Ic/101Ic.csv",
        "/Users/albert-mac/Code/GitHub/CPR/data/Ic/491Ic-.csv"
    ]
    
    for test_file in test_files:
        try:
            debug_file(test_file)
        except Exception as e:
            print(f"Error testing {test_file}: {e}")
