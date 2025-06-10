#!/usr/bin/env python3
"""
改善版本的主處理器 - 修復頻率計算問題
主要改進：
1. 優先使用 Lomb-Scargle 功率譜分析的頻率
2. 添加頻率合理性檢查
3. 改善歸一化處理
4. 提供更清晰的頻率來源說明
"""

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

# Import existing modules
from .config import config
from .logger import init_logger
from .josephson_model import JosephsonFitter, preprocess_data_numba

# Try to import performance libraries
try:
    import numba
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import fireducks.pandas as pd_fast
    USING_FIREDUCKS = True
except ImportError:
    pd_fast = pd
    USING_FIREDUCKS = False

warnings.filterwarnings('ignore')

# Constants
MAX_WORKERS = min(8, multiprocessing.cpu_count())
PLOT_SIZE = (19.2, 10.8)  # 1920x1080 at 100 DPI
PLOT_DPI = 100

# Global locks for thread safety
GLOBAL_PROCESSING_LOCK = threading.Lock()
NUMBA_COMPILATION_LOCK = threading.Lock()

# Numba-optimized functions
if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def model_f_numba(Phi_ext, I_c, phi_0, f, T, r, C):
        """Numba-optimized Josephson junction model function"""
        main_phase = 2 * np.pi * f * Phi_ext - phi_0
        half_phase = main_phase / 2
        sin_half = np.sin(half_phase)
        sin_main = np.sin(main_phase)
        denominator_term = 1 - T * sin_half**2
        denominator_term = np.maximum(denominator_term, 1e-12)
        denominator = np.sqrt(denominator_term)
        return I_c * sin_main / denominator + r * Phi_ext + C

    @jit(nopython=True, cache=True, fastmath=True)
    def calculate_statistics_numba(y_data, fitted_data, n_params):
        """Fast statistical calculations using numba"""
        residuals = y_data - fitted_data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        n = len(y_data)
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params - 1)) if n > n_params + 1 else r_squared
        
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(residuals))
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
        phase_bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        mean_binned_values = np.full(num_bins, np.nan)
        for i in range(num_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
            if np.any(mask):
                mean_binned_values[i] = np.mean(y_data_normalized[mask])
        return bin_centers, mean_binned_values

else:
    # Fallback implementations
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
        residuals = y_data - fitted_data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        n = len(y_data)
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params - 1)) if n > n_params + 1 else r_squared
        
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(residuals))
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

# Model function wrapper for curve_fit
def model_f(x, I_c, phi_0, f, T, r, C):
    return model_f_numba(x, I_c, phi_0, f, T, r, C)

@lru_cache(maxsize=128)
def generate_frequency_array(n_points, median_diff, n_freq=10000):
    """Generate frequency array with caching and safety checks"""
    if not np.isfinite(median_diff) or median_diff <= 0:
        median_diff = 1.0
    
    freq_min = max(1e-5, 1.0 / (n_points * median_diff))
    freq_max = 1 / (2 * median_diff)
    
    if not np.isfinite(freq_max) or freq_max <= freq_min:
        freq_max = max(1.0, freq_min * 1000)
    
    if freq_max / freq_min < 10:
        freq_min = freq_max / 1000
    
    frequencies = np.linspace(freq_min, freq_max, n_freq)
    
    if not np.isfinite(frequencies).all():
        frequencies = np.linspace(1e-5, 1.0, n_freq)
    
    return frequencies

class ImprovedJosephsonProcessor:
    """改善版本的約瑟夫森結處理器 - 修復頻率計算問題"""
    
    def __init__(self):
        self.config = config
        self.logger = init_logger(config)
        
        # Thread-safe management
        self.output_lock = threading.Lock()
        self.matplotlib_lock = threading.Lock()
        self.progress_counter = {'current': 0, 'total': 0}
        
        # Pre-compile numba functions
        self._precompile_numba()
        
        self.logger.logger.info(f"Initialized ImprovedJosephsonProcessor")
        self.logger.logger.info(f"Using FireDucks pandas: {USING_FIREDUCKS}")
        self.logger.logger.info(f"Using Numba optimization: {HAS_NUMBA}")
        self.logger.logger.info(f"Max workers: {MAX_WORKERS}")
        self.logger.logger.info(f"Plot size: {PLOT_SIZE} at {PLOT_DPI} DPI")

    def _precompile_numba(self):
        """Pre-compile numba functions for better performance"""
        with NUMBA_COMPILATION_LOCK:
            self.logger.logger.info("Pre-compiling Numba functions...")
            try:
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

    def safe_print(self, message):
        """Thread-safe printing"""
        try:
            with self.output_lock:
                print(message)
        except (BrokenPipeError, OSError):
            pass

    def update_progress(self, dataid, success, error_msg=None):
        """Thread-safe progress update"""
        try:
            with self.output_lock:
                self.progress_counter['current'] += 1
                current = self.progress_counter['current']
                total = self.progress_counter['total']
                
                if current % 20 == 0 or not success:
                    status = "✓" if success else "✗"
                    message = f"{status} [{current}/{total}] {dataid}"
                    if not success and error_msg:
                        message += f": {error_msg[:50]}..."
                    self.logger.logger.info(message)
                    
        except (BrokenPipeError, OSError):
            pass

    def analyze_frequency_with_validation(self, x_data_normalized, y_data_normalized, x_factor):
        """
        改善的頻率分析 - 包含合理性檢查和多種方法
        
        Returns:
        --------
        dict: {
            'best_frequency': float,
            'frequency_source': str,
            'ls_frequency': float,
            'fit_frequency_normalized': float,
            'fit_frequency_scaled': float,
            'frequency_reliable': bool,
            'analysis_details': dict
        }
        """
        analysis_result = {
            'best_frequency': 1.0,
            'frequency_source': 'fallback',
            'ls_frequency': None,
            'fit_frequency_normalized': None,
            'fit_frequency_scaled': None,
            'frequency_reliable': False,
            'analysis_details': {}
        }
        
        try:
            # 1. Lomb-Scargle 功率譜分析
            median_diff = np.median(np.diff(x_data_normalized))
            if not np.isfinite(median_diff) or median_diff <= 0:
                median_diff = 1.0
            
            frequencies = generate_frequency_array(len(x_data_normalized), median_diff)
            
            ls = LombScargle(x_data_normalized, y_data_normalized)
            power = ls.power(frequencies)
            
            # 修復功率譜中的數值問題
            if not np.isfinite(power).all():
                nan_count = np.isnan(power).sum()
                inf_count = np.isinf(power).sum()
                
                if inf_count > 0 and inf_count < len(power) * 0.1:
                    max_finite_power = np.max(power[np.isfinite(power)])
                    power[np.isinf(power)] = max_finite_power * 2
                
                if nan_count > 0 and nan_count < len(power) * 0.1:
                    mean_finite_power = np.mean(power[np.isfinite(power)])
                    power[np.isnan(power)] = mean_finite_power
            
            # 尋找峰值
            if np.isfinite(power).all():
                height_threshold = np.max(power) * 0.1
                peaks, _ = find_peaks(power, height=height_threshold, distance=100)
                
                if len(peaks) > 0:
                    peak_frequencies = frequencies[peaks]
                    peak_powers = power[peaks]
                    sorted_indices = np.argsort(peak_powers)[::-1]
                    ls_best_frequency = peak_frequencies[sorted_indices[0]]
                else:
                    # 使用全局最大值
                    max_idx = np.argmax(power)
                    ls_best_frequency = frequencies[max_idx]
                
                analysis_result['ls_frequency'] = ls_best_frequency
                analysis_result['analysis_details']['ls_power_max'] = np.max(power)
                analysis_result['analysis_details']['ls_peaks_found'] = len(peaks) if len(peaks) > 0 else 0
                
                # 檢查 Lomb-Scargle 頻率的合理性
                ls_frequency_reasonable = (1e-6 < ls_best_frequency < 1e6)
                
                if ls_frequency_reasonable:
                    analysis_result['best_frequency'] = ls_best_frequency
                    analysis_result['frequency_source'] = 'lomb_scargle'
                    analysis_result['frequency_reliable'] = True
                    
        except Exception as e:
            analysis_result['analysis_details']['ls_error'] = str(e)
        
        # 2. 嘗試非線性擬合頻率（僅作為參考）
        try:
            # 使用 Lomb-Scargle 結果作為初始猜測
            f_init = analysis_result.get('ls_frequency', 1.0)
            
            I_c_init = 3 * np.std(y_data_normalized)
            phi_0_init = 0.0
            T_init = 0.5
            r_init = np.polyfit(x_data_normalized, y_data_normalized, 1)[0] if len(x_data_normalized) > 1 else 0.0
            C_init = np.mean(y_data_normalized)
            
            p0 = [I_c_init, phi_0_init, f_init, T_init, r_init, C_init]
            bounds = ([0, -np.pi, 0, 0, -np.inf, 0], 
                     [np.inf, np.pi, np.inf, 1, np.inf, np.inf])
            
            popt, pcov = curve_fit(model_f, x_data_normalized, y_data_normalized, 
                                 p0=p0, bounds=bounds, maxfev=50000)
            
            fit_frequency_normalized = popt[2]
            fit_frequency_scaled = fit_frequency_normalized / x_factor
            
            analysis_result['fit_frequency_normalized'] = fit_frequency_normalized
            analysis_result['fit_frequency_scaled'] = fit_frequency_scaled
            
            # 檢查擬合頻率的合理性
            fit_frequency_reasonable = (1e-6 < fit_frequency_scaled < 1e6)
            
            # 如果 Lomb-Scargle 不可靠但擬合頻率合理，使用擬合頻率
            if not analysis_result['frequency_reliable'] and fit_frequency_reasonable:
                analysis_result['best_frequency'] = fit_frequency_scaled
                analysis_result['frequency_source'] = 'nonlinear_fit'
                analysis_result['frequency_reliable'] = True
                
            # 記錄比較信息
            if analysis_result['ls_frequency'] is not None and fit_frequency_reasonable:
                frequency_ratio = fit_frequency_scaled / analysis_result['ls_frequency']
                analysis_result['analysis_details']['frequency_ratio'] = frequency_ratio
                analysis_result['analysis_details']['frequencies_consistent'] = (0.1 < frequency_ratio < 10)
                
        except Exception as e:
            analysis_result['analysis_details']['fit_error'] = str(e)
        
        # 3. 最終檢查和回退
        if not analysis_result['frequency_reliable']:
            analysis_result['best_frequency'] = 1.0
            analysis_result['frequency_source'] = 'fallback'
            analysis_result['analysis_details']['warning'] = 'Using fallback frequency due to unreliable analysis'
        
        return analysis_result

    def process_single_file(self, csv_file_path, output_dir):
        """處理單一檔案 - 改善版本"""
        try:
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
            
            # 數據歸一化
            x_data_normalized, y_data_normalized, x_factor, y_factor = preprocess_data_numba(x_data, y_data)
            
            # 改善的頻率分析
            freq_analysis = self.analyze_frequency_with_validation(x_data_normalized, y_data_normalized, x_factor)
            best_frequency = freq_analysis['best_frequency']
            
            # 使用最佳頻率進行最終擬合
            try:
                I_c_init = 3 * np.std(y_data_normalized)
                phi_0_init = 0.0
                f_init = best_frequency * x_factor if freq_analysis['frequency_source'] != 'fallback' else best_frequency
                T_init = 0.5
                r_init = np.polyfit(x_data_normalized, y_data_normalized, 1)[0] if len(x_data_normalized) > 1 else 0.0
                C_init = np.mean(y_data_normalized)
                
                p0 = [I_c_init, phi_0_init, f_init, T_init, r_init, C_init]
                bounds = ([0, -np.pi, 0, 0, -np.inf, 0], 
                         [np.inf, np.pi, np.inf, 1, np.inf, np.inf])
                
                popt, pcov = curve_fit(model_f, x_data_normalized, y_data_normalized, 
                                     p0=p0, bounds=bounds, maxfev=50000)
                
                if not np.all(np.isfinite(popt)):
                    raise ValueError("NaN/inf in fitted parameters")
                
            except Exception as fit_error:
                return {
                    'dataid': dataid,
                    'success': False,
                    'error': f'Curve fitting failed: {str(fit_error)}'
                }
            
            # 提取優化參數
            I_c_opt, phi_0_opt, f_opt, T_opt, r_opt, C_opt = popt
            
            # 生成擬合數據
            fitted_y_data = model_f_numba(x_data_normalized, I_c_opt, phi_0_opt, f_opt, T_opt, r_opt, C_opt)
            
            # 統計計算
            r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std = calculate_statistics_numba(
                y_data_normalized, fitted_y_data, len(popt))
            
            # 參數換算到原始單位
            I_c_scaled = I_c_opt * y_factor
            phi_0_scaled = phi_0_opt
            f_scaled = f_opt / x_factor
            T_scaled = T_opt
            r_scaled = r_opt * y_factor / x_factor
            C_scaled = C_opt * y_factor + min(y_data)
            
            # === 視覺化 ===
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 歸一化數據圖
            with self.matplotlib_lock:
                plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                
                sort_indices = np.argsort(x_data_normalized)
                x_sorted = x_data_normalized[sort_indices]
                y_sorted = y_data_normalized[sort_indices]
                fitted_y_sorted = fitted_y_data[sort_indices]
                
                plt.plot(x_sorted, y_sorted, '--', color='blue', linewidth=1, alpha=0.7, label=f'{dataid} (connected)')
                plt.scatter(x_data_normalized, y_data_normalized, color='blue', s=8, alpha=0.8, zorder=5)
                plt.plot(x_sorted, fitted_y_sorted, label='Full Fitted Model', color='red', linewidth=2)
                
                linear_trend = r_opt * x_data_normalized + C_opt
                linear_trend_sorted = linear_trend[sort_indices]
                plt.plot(x_sorted, linear_trend_sorted, '--', color='green', linewidth=2, alpha=0.8, label='Linear Trend (rx+C)')
                
                plt.xlabel('Normalized External Magnetic Flux (Φ_ext)', fontsize=14)
                plt.ylabel('Normalized Supercurrent (I_s)', fontsize=14)
                plt.title('Supercurrent vs. Normalized External Magnetic Flux', fontsize=16)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.grid()
                
                # 改善的參數文本框 - 包含頻率來源信息
                display_freq_normalized = best_frequency if freq_analysis['frequency_source'] == 'lomb_scargle' else f_opt
                param_text = f'Optimized Parameters:\nI_c: {I_c_opt:.2e}\nphi_0: {phi_0_opt:.2f}\nf: {display_freq_normalized:.2e} (norm.)\nT: {T_opt:.2%}\nr: {r_opt:.2e}\nC: {C_opt:.2e}'
                stats_text = f'Statistical Metrics:\nR²: {r_squared:.4f}\nAdj. R²: {adj_r_squared:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
                
                # 頻率分析信息
                freq_info = f'Frequency Analysis:\nSource: {freq_analysis["frequency_source"]}\nLS Freq: {freq_analysis.get("ls_frequency", "N/A"):.4e}\nReliable: {freq_analysis["frequency_reliable"]}'
                
                plt.text(1.02, 0.70, param_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.text(1.02, 0.45, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                plt.text(1.02, 0.20, freq_info, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                plt.subplots_adjust(left=0.08, right=0.70, top=0.92, bottom=0.08)
                plt.savefig(f'{output_dir}/{dataid}_fitted_curve_normalized_plot.png', dpi=PLOT_DPI)
                plt.close()
            
            # 2. 原始尺度數據圖
            fitted_y_original = model_f_numba(x_data_normalized, I_c_opt, phi_0_opt, f_opt, T_opt, r_opt, C_opt) * y_factor + min(y_data)
            r_squared_original, adj_r_squared_original, rmse_original, mae_original, ss_res_original, _, _ = calculate_statistics_numba(
                y_data, fitted_y_original, len(popt))
            
            with self.matplotlib_lock:
                plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                
                sort_indices = np.argsort(x_data)
                x_sorted = x_data[sort_indices]
                y_sorted = y_data[sort_indices]
                fitted_y_sorted = fitted_y_original[sort_indices]
                
                plt.plot(x_sorted, y_sorted, '--', color='blue', linewidth=1, alpha=0.7, label=f'{dataid} (connected)')
                plt.scatter(x_data, y_data, color='blue', s=8, alpha=0.8, zorder=5)
                plt.plot(x_sorted, fitted_y_sorted, label='Full Fitted Model', color='red', linewidth=2)
                
                linear_trend_original = r_scaled * x_data + C_scaled 
                linear_trend_sorted = linear_trend_original[sort_indices]
                plt.plot(x_sorted, linear_trend_sorted, '--', color='green', linewidth=2, alpha=0.8, label='Linear Trend (rx+C)')
                
                plt.xlabel('External Magnetic Flux (Φ_ext)', fontsize=14)
                plt.ylabel('Supercurrent (I_s)', fontsize=14)
                plt.title('Supercurrent vs. External Magnetic Flux', fontsize=16)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.grid()
                
                # 改善的參數文本框 - 使用正確的頻率值
                # 換算最佳頻率到原始單位
                display_frequency = best_frequency / x_factor if freq_analysis['frequency_source'] == 'lomb_scargle' else f_scaled
                scaled_param_text = f'Scaled Parameters:\nI_c: {I_c_scaled:.2e}\nphi_0: {phi_0_scaled:.2f}\nf: {display_frequency:.2e} Hz\nT: {T_scaled:.2%}\nr: {r_scaled:.2e}\nC: {C_scaled:.2e}'
                original_stats_text = f'Statistical Metrics:\nR²: {r_squared_original:.4f}\nAdj. R²: {adj_r_squared_original:.4f}\nRMSE: {rmse_original:.4f}\nMAE: {mae_original:.4f}'
                
                plt.text(1.02, 0.50, scaled_param_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.text(1.02, 0.25, original_stats_text, transform=plt.gca().transAxes, fontsize=10, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08)
                plt.savefig(f'{output_dir}/{dataid}_fitted_curve_plot.png', dpi=PLOT_DPI)
                plt.close()
            
            # 3. 殘差分析
            residuals = y_data - fitted_y_original
            with self.matplotlib_lock:
                plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                
                plt.subplot(2, 2, 1)
                plt.scatter(x_data, residuals, label=f'{dataid} Residuals', color='green', s=5)
                plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
                plt.xlabel('External Magnetic Flux', fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title('Residuals of the Fit', fontsize=14)
                plt.legend()
                plt.grid()
                
                plt.subplot(2, 2, 2)
                plt.scatter(fitted_y_original, residuals, alpha=0.6, s=5, color='orange')
                plt.axhline(y=0, color='red', linestyle='--')
                plt.xlabel('Fitted Values', fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title('Residuals vs Fitted Values', fontsize=14)
                plt.grid()
                
                plt.subplot(2, 2, 3)
                stats.probplot(residuals, dist="norm", plot=plt)
                plt.title('Q-Q Plot of Residuals', fontsize=14)
                plt.grid()
                
                plt.subplot(2, 2, 4)
                plt.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black', color='lightcoral')
                plt.xlabel('Residuals', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.title('Distribution of Residuals', fontsize=14)
                plt.grid()
                
                x_normal = np.linspace(residuals.min(), residuals.max(), 100)
                y_normal = stats.norm.pdf(x_normal, np.mean(residuals), np.std(residuals))
                plt.plot(x_normal, y_normal, 'r-', label='Normal Distribution', linewidth=2)
                plt.legend()
                
                plt.tight_layout(pad=0)
                plt.savefig(f'{output_dir}/{dataid}_residuals_plot.png', dpi=PLOT_DPI, bbox_inches='tight', pad_inches=0)
                plt.close()
            
            # 4. 相位折疊圖 - 使用正確的頻率
            # 換算顯示頻率到原始單位，用於圖表標籤
            display_frequency = best_frequency / x_factor if freq_analysis['frequency_source'] == 'lomb_scargle' else f_scaled
            
            if best_frequency > 0:
                best_period = 1 / display_frequency
                # 相位計算使用歸一化的最佳頻率
                phase, cycle_number, total_cycles = calculate_phase_data_numba(x_data_normalized, best_frequency)
                
                with self.matplotlib_lock:
                    plt.figure(figsize=PLOT_SIZE, dpi=PLOT_DPI)
                    
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    cycle_peak_phases = []
                    
                    for cycle in range(total_cycles):
                        mask = cycle_number == cycle
                        if np.any(mask):
                            color = colors[cycle % len(colors)]
                            cycle_phase = phase[mask]
                            cycle_y = y_data_normalized[mask]
                            
                            if len(cycle_y) > 0:
                                max_idx = np.argmax(cycle_y)
                                peak_phase = cycle_phase[max_idx]
                                peak_value = cycle_y[max_idx]
                                cycle_peak_phases.append(peak_phase)
                                
                                sort_indices = np.argsort(cycle_phase)
                                sorted_phase = cycle_phase[sort_indices]
                                sorted_y = cycle_y[sort_indices]
                                
                                plt.plot(sorted_phase, sorted_y, 'o-', color=color, 
                                        label=f'Cycle {cycle + 1}', markersize=4, linewidth=2, alpha=0.8)
                                plt.scatter([peak_phase], [peak_value], color=color, s=120, 
                                           marker='*', edgecolors='black', linewidth=2, zorder=5)
                    
                    # 添加平均輪廓
                    bin_centers, mean_binned_values = calculate_binned_average_numba(phase, y_data_normalized)
                    valid_mask = ~np.isnan(mean_binned_values)
                    if np.any(valid_mask):
                        valid_centers = bin_centers[valid_mask]
                        valid_means = mean_binned_values[valid_mask]
                        sort_indices = np.argsort(valid_centers)
                        sorted_centers = valid_centers[sort_indices]
                        sorted_means = valid_means[sort_indices]
                        plt.plot(sorted_centers, sorted_means, 'k--', linewidth=3, 
                                label='Average Profile', marker='s', markersize=6)
                    
                    plt.xlabel(f'Phase (Period = {best_period:.6f})', fontsize=14)
                    plt.ylabel('Normalized Supercurrent (I_s)', fontsize=14)
                    plt.title(f'Phase-Folded Plot - Total {total_cycles} Cycles', fontsize=16)
                    plt.xlim(0, 1)
                    plt.grid(True, alpha=0.3)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # 改善的信息文本框
                    info_text = f'Best Frequency: {display_frequency:.6e} Hz\nSource: {freq_analysis["frequency_source"]}\nPeriod: {best_period:.6f}\nTotal Cycles: {total_cycles}'
                    plt.text(1.02, 0.50, info_text, transform=plt.gca().transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
                    
                    # 相位漂移統計
                    if len(cycle_peak_phases) > 1:
                        phase_drift = np.diff(cycle_peak_phases)
                        phase_drift = np.where(phase_drift > 0.5, phase_drift - 1, phase_drift)
                        phase_drift = np.where(phase_drift < -0.5, phase_drift + 1, phase_drift)
                        
                        drift_stats = f'Phase Drift Statistics:\nMean Drift: {np.mean(phase_drift):.6f}\nStd Dev: {np.std(phase_drift):.6f}\n★ = Peak positions'
                        plt.text(1.02, 0.25, drift_stats, transform=plt.gca().transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                    
                    plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08)
                    plt.savefig(f'{output_dir}/{dataid}_phase_folded_with_drift.png', dpi=PLOT_DPI)
                    plt.close()
                
                # 5. 按週期著色的原始數據
                with self.matplotlib_lock:
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
                    
                    for cycle in range(1, total_cycles):
                        # 使用歸一化的最佳頻率計算週期邊界
                        boundary = cycle / best_frequency
                        if boundary <= np.max(x_data_normalized):
                            plt.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7, linewidth=1)
                    
                    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
                    plt.savefig(f'{output_dir}/{dataid}_cycles_colored_matplotlib.png', dpi=PLOT_DPI)
                    plt.close()
            
            # 返回改善的結果 - 使用正確的頻率值
            final_frequency = best_frequency / x_factor if freq_analysis['frequency_source'] == 'lomb_scargle' else f_scaled
            
            return {
                'dataid': dataid,
                'success': True,
                'I_c': I_c_scaled,
                'phi_0': phi_0_scaled,
                'f': final_frequency,  # 使用正確的頻率
                'T': T_scaled,
                'r': r_scaled,
                'C': C_scaled,
                'r_squared': r_squared_original,
                'adj_r_squared': adj_r_squared_original,
                'rmse': rmse_original,
                'mae': mae_original,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                # 添加頻率分析詳情
                'frequency_analysis': freq_analysis,
                'frequency_source': freq_analysis['frequency_source'],
                'frequency_reliable': freq_analysis['frequency_reliable']
            }
            
        except Exception as e:
            self.update_progress(Path(csv_file_path).stem, False, f"Error: {str(e)[:50]}...")
            return {
                'dataid': Path(csv_file_path).stem,
                'success': False,
                'error': str(e)
            }


def main():
    """主入口點"""
    print("=" * 60)
    print("改善版本的 CPR 分析系統")
    print("主要改進：修復頻率計算問題")
    print("=" * 60)
    
    processor = ImprovedJosephsonProcessor()
    
    # 示例：處理單一文件
    file_path = "data/Ic/435Ic.csv"
    output_dir = "output/improved_analysis_435Ic"
    
    if os.path.exists(file_path):
        print(f"正在分析檔案：{file_path}")
        result = processor.process_single_file(file_path, output_dir)
        
        if result['success']:
            print("\n✅ 分析成功完成！")
            print(f"檔案 ID：{result['dataid']}")
            print(f"頻率來源：{result['frequency_source']}")
            print(f"頻率可靠性：{result['frequency_reliable']}")
            print(f"最終頻率：{result['f']:.6e} Hz")
            print(f"透明度：{result['T']:.4f} ({result['T']*100:.2f}%)")
            print(f"R²：{result['r_squared']:.6f}")
        else:
            print(f"❌ 分析失敗：{result['error']}")
    else:
        print(f"❌ 檔案不存在：{file_path}")


if __name__ == "__main__":
    main()
