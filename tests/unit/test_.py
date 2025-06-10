import os
import sys
import time
import glob
import threading
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import stats
from astropy.timeseries import LombScargle
csv_file_path = "data/Ic/435Ic.csv"
dataid = Path(csv_file_path).stem
            
# 加載數據
df = pd.read_csv(csv_file_path)
x_data = df['y_field'].values.astype(np.float64)
y_data = df['Ic'].values.astype(np.float64)

# 數據清理
valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
if not np.any(valid_mask):
    return {
        'dataid': dataid,
        'success': False,
        'error': 'All data points are NaN or infinite'
    }

x_data = x_data[valid_mask]
y_data = y_data[valid_mask]

# 移除前10個點
if len(x_data) >= 10:
    x_data = x_data[10:]
    y_data = y_data[10:]

if len(x_data) < 20:
    return {
        'dataid': dataid,
        'success': False,
        'error': f'Insufficient data points ({len(x_data)} < 20)'
    }
# Shift data to start from zero
x_shifted = x_data - x_data[0]
y_shifted = y_data - np.min(y_data)

# Calculate normalization factors
x_factor = np.abs(x_shifted[2] - x_shifted[1]) if len(x_shifted) > 2 else 1.0
y_factor = np.abs(y_shifted[2] - y_shifted[1]) if len(y_shifted) > 2 else 1.0

# Avoid division by zero
x_factor = max(x_factor, 1e-12)
y_factor = max(y_factor, 1e-12)

# Normalize
x_normalized = x_shifted / x_factor
y_normalized = y_shifted / y_factor
# 數據歸一化
# x_data_normalized, y_data_normalized, x_factor, y_factor = preprocess_data_numba(x_data, y_data)

# # 改善的頻率分析
# freq_analysis = self.analyze_frequency_with_validation(x_data_normalized, y_data_normalized, x_factor)
# best_frequency = freq_analysis['best_frequency']