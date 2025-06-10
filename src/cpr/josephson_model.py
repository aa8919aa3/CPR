"""
Josephson Junction Physics Model with Numba Optimization
"""
import numpy as np
import pandas as pd
import numba
from numba import jit, prange
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings
from scipy import stats

# Suppress numba warnings
try:
    from numba.core.errors import NumbaWarning
    warnings.filterwarnings('ignore', category=NumbaWarning)
except:
    pass

# Check numba availability
try:
    # Test numba compilation
    @jit(nopython=True)
    def _test_numba():
        return 1.0
    _test_numba()
    HAS_NUMBA = True
except:
    HAS_NUMBA = False

@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def josephson_model_numba(Phi_ext: np.ndarray, I_c: float, phi_0: float, 
                         f: float, T: float, r: float, C: float) -> np.ndarray:
    """
    Optimized Josephson junction model using Numba JIT compilation
    
    Parameters:
    -----------
    Phi_ext : External magnetic flux array
    I_c : Critical current
    phi_0 : Phase offset
    f : Frequency
    T : Transparency parameter
    r : Linear resistance term
    C : Constant offset
    
    Returns:
    --------
    Supercurrent array
    """
    main_phase = 2 * np.pi * f * Phi_ext - phi_0
    half_phase = main_phase / 2
    sin_half = np.sin(half_phase)
    sin_main = np.sin(main_phase)
    
    # Numerical stability protection
    denominator_term = 1 - T * sin_half**2
    denominator_term = np.maximum(denominator_term, 1e-12)
    denominator = np.sqrt(denominator_term)
    
    return I_c * sin_main / denominator + r * Phi_ext + C

def josephson_model_wrapper(Phi_ext: np.ndarray, I_c: float, phi_0: float, 
                           f: float, T: float, r: float, C: float) -> np.ndarray:
    """Wrapper for scipy.optimize.curve_fit compatibility"""
    return josephson_model_numba(Phi_ext, I_c, phi_0, f, T, r, C)

@jit(nopython=True, cache=True, fastmath=True)
def calculate_statistics_numba(y_observed: np.ndarray, y_predicted: np.ndarray, 
                              n_params: int) -> Tuple[float, float, float, float, float, float, float]:
    """
    Fast statistical calculations using Numba
    
    Returns:
    --------
    r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std
    """
    n = len(y_observed)
    y_mean = np.mean(y_observed)
    
    # Sum of squares
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - y_mean) ** 2)
    
    # R-squared metrics
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1) if n > n_params + 1 else 0.0
    
    # Error metrics
    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(y_observed - y_predicted))
    
    # Residual statistics
    residuals = y_observed - y_predicted
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    return float(r_squared), float(adj_r_squared), float(rmse), float(mae), float(ss_res), float(residual_mean), float(residual_std)

@jit(nopython=True, cache=True, fastmath=True)
def calculate_mode_magnitude(values: np.ndarray) -> float:
    """
    計算數量級眾數的 Numba 兼容版本
    """
    if len(values) == 0:
        return 1.0
    
    # 移除零值和負值
    positive_values = values[values > 0]
    if len(positive_values) == 0:
        return 1.0
    
    # 計算 log10 數量級
    log_values = np.log10(positive_values)
    magnitudes = np.floor(log_values)
    
    # 手動計算眾數（Numba 兼容）
    if len(magnitudes) == 0:
        return 1.0
    
    # 找到唯一值和計數
    unique_mags = np.unique(magnitudes)
    max_count = 0
    mode_magnitude = magnitudes[0]
    
    for mag in unique_mags:
        count = np.sum(magnitudes == mag)
        if count > max_count:
            max_count = count
            mode_magnitude = mag
    
    return 10.0 ** mode_magnitude

def calculate_mode_magnitude_fallback(values: np.ndarray) -> float:
    """
    計算數量級眾數的非 Numba 版本（回退實現）
    """
    if len(values) == 0:
        return 1.0
    
    # 移除零值和負值
    positive_values = values[values > 0]
    if len(positive_values) == 0:
        return 1.0
    
    # 計算 log10 數量級
    log_values = np.log10(positive_values)
    magnitudes = np.floor(log_values)
    
    # 使用 scipy.stats.mode 計算眾數
    if len(magnitudes) == 0:
        return 1.0
    
    try:
        mode_result = stats.mode(magnitudes, keepdims=True)
        mode_magnitude = mode_result.mode[0]
    except:
        # 如果 scipy.stats.mode 失敗，手動計算
        unique_mags, counts = np.unique(magnitudes, return_counts=True)
        mode_magnitude = unique_mags[np.argmax(counts)]
    
    return 10.0 ** mode_magnitude

def preprocess_data_numba(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    改進的數據預處理 - 使用您指定的精確流程
    
    流程：
    1. 平移資料使起點或最小值為 0
    2. 計算差值、log10 數量級、四捨五入後取眾數
    3. 縮放因子
    4. 正規化資料
    
    Returns:
    --------
    x_normalized, y_normalized, x_factor, y_factor
    """
    import pandas as pd
    
    # 轉換為 pandas Series
    x_data = pd.Series(x_data)
    y_data = pd.Series(y_data)
    
    # 1. 平移資料使起點或最小值為 0
    x_shifted = x_data - x_data.iloc[0]
    y_shifted = y_data - y_data.min()

    # 2. 計算差值、log10 數量級、四捨五入後取眾數
    # 3. 縮放因子
    try:
        x_diffs = x_shifted.diff().abs().replace(0, np.nan).dropna()
        if len(x_diffs) > 0:
            x_modes = x_diffs.apply(lambda x: round(np.log10(x))).mode()
            if len(x_modes) > 0:
                x_factor = 10.0 ** x_modes.iloc[0]
            else:
                x_factor = 1.0
        else:
            x_factor = 1.0
    except:
        x_factor = 1.0
    
    try:
        y_diffs = y_shifted.diff().abs().replace(0, np.nan).dropna()
        if len(y_diffs) > 0:
            y_modes = y_diffs.apply(lambda y: round(np.log10(y))).mode()
            if len(y_modes) > 0:
                y_factor = 10.0 ** y_modes.iloc[0]
            else:
                y_factor = 1.0
        else:
            y_factor = 1.0
    except:
        y_factor = 1.0

    # 4. 正規化資料
    x_normalized = x_shifted / x_factor
    y_normalized = y_shifted / y_factor
    
    return x_normalized.values, y_normalized.values, float(x_factor), float(y_factor)

def preprocess_data_fallback(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    改進的數據預處理 - 使用 pandas 風格的差值和數量級眾數計算縮放因子（非 Numba 版本）
    
    Returns:
    --------
    x_normalized, y_normalized, x_factor, y_factor
    """
    import pandas as pd
    
    # 轉換為 pandas Series 以使用 .diff() 和 .mode()
    x_series = pd.Series(x_data)
    y_series = pd.Series(y_data)
    
    # 1. 平移資料使起點或最小值為 0
    x_shifted = x_series - x_series.iloc[0]
    y_shifted = y_series - y_series.min()
    
    # 2. 計算差值、log10 數量級、四捨五入後取眾數
    # 3. 縮放因子
    try:
        # X 方向的縮放因子
        x_diffs = x_shifted.diff().abs().replace(0, np.nan).dropna()
        if len(x_diffs) > 0:
            x_log_values = x_diffs.apply(lambda x: round(np.log10(x)) if x > 0 else 0)
            x_mode = x_log_values.mode()
            x_factor = 10.0 ** x_mode.iloc[0] if len(x_mode) > 0 else 1e-6
        else:
            x_factor = 1e-6
        
        # Y 方向的縮放因子
        y_diffs = y_shifted.diff().abs().replace(0, np.nan).dropna()
        if len(y_diffs) > 0:
            y_log_values = y_diffs.apply(lambda y: round(np.log10(y)) if y > 0 else 0)
            y_mode = y_log_values.mode()
            y_factor = 10.0 ** y_mode.iloc[0] if len(y_mode) > 0 else 1e-6
        else:
            y_factor = 1e-6
            
    except Exception:
        # 回退到安全值
        x_factor = 1e-6
        y_factor = 1e-6
    
    # 避免除零和過小的值
    x_factor = max(float(x_factor), 1e-12)
    y_factor = max(float(y_factor), 1e-12)
    
    # 4. 正規化資料
    x_normalized = x_shifted.values / x_factor
    y_normalized = y_shifted.values / y_factor
    
    return x_normalized, y_normalized, x_factor, y_factor

@jit(nopython=True, cache=True, fastmath=True)
def calculate_diff_magnitude_mode(values: np.ndarray) -> float:
    """
    使用相鄰點差值計算數量級眾數的 Numba 兼容版本
    
    Parameters:
    -----------
    values : np.ndarray
        輸入數據陣列
        
    Returns:
    --------
    float
        基於差值的數量級眾數（10的冪）
    """
    if len(values) <= 1:
        return 1.0
    
    # 計算相鄰點差值
    diffs = np.diff(values)
    
    # 取絕對值，移除零值
    abs_diffs = np.abs(diffs)
    positive_diffs = abs_diffs[abs_diffs > 0]
    
    if len(positive_diffs) == 0:
        return 1.0
    
    # 計算 log10 數量級
    log_diffs = np.log10(positive_diffs)
    magnitudes = np.floor(log_diffs)
    
    # 手動計算眾數（Numba 兼容）
    if len(magnitudes) == 0:
        return 1.0
    
    # 找到唯一值和計數
    unique_mags = np.unique(magnitudes)
    max_count = 0
    mode_magnitude = magnitudes[0]
    
    for mag in unique_mags:
        count = np.sum(magnitudes == mag)
        if count > max_count:
            max_count = count
            mode_magnitude = mag
    
    return 10.0 ** mode_magnitude

def calculate_diff_magnitude_mode_fallback(values: np.ndarray) -> float:
    """
    使用相鄰點差值計算數量級眾數的非 Numba 版本（回退實現）
    
    Parameters:
    -----------
    values : np.ndarray
        輸入數據陣列
        
    Returns:
    --------
    float
        基於差值的數量級眾數（10的冪）
    """
    if len(values) <= 1:
        return 1.0
    
    # 計算相鄰點差值
    diffs = np.diff(values)
    
    # 取絕對值，移除零值
    abs_diffs = np.abs(diffs)
    positive_diffs = abs_diffs[abs_diffs > 0]
    
    if len(positive_diffs) == 0:
        return 1.0
    
    # 計算 log10 數量級
    log_diffs = np.log10(positive_diffs)
    magnitudes = np.floor(log_diffs)
    
    if len(magnitudes) == 0:
        return 1.0
    
    try:
        # 使用 scipy.stats.mode 計算眾數
        mode_result = stats.mode(magnitudes, keepdims=True)
        mode_magnitude = mode_result.mode[0]
    except:
        # 如果 scipy.stats.mode 失敗，手動計算
        unique_mags, counts = np.unique(magnitudes, return_counts=True)
        mode_magnitude = unique_mags[np.argmax(counts)]
    
    return 10.0 ** mode_magnitude

@jit(nopython=True, cache=True, fastmath=True)
def preprocess_data_diff_numba(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    使用差值基礎的數據預處理 - 使用相鄰點差值的數量級眾數計算縮放因子
    
    Parameters:
    -----------
    x_data : np.ndarray
        X軸數據
    y_data : np.ndarray
        Y軸數據
        
    Returns:
    --------
    x_normalized : np.ndarray
        歸一化的X數據
    y_normalized : np.ndarray
        歸一化的Y數據
    x_factor : float
        X數據的縮放因子
    y_factor : float
        Y數據的縮放因子
    """
    # 平移數據到零點
    x_shifted = x_data - x_data[0]
    y_shifted = y_data - np.min(y_data)
    
    # 使用相鄰點差值的數量級眾數計算縮放因子
    x_factor = calculate_diff_magnitude_mode(x_shifted)
    y_factor = calculate_diff_magnitude_mode(y_shifted)
    
    # 避免除零和過小的值
    x_factor = max(x_factor, 1e-12)
    y_factor = max(y_factor, 1e-12)
    
    # 歸一化
    x_normalized = x_shifted / x_factor
    y_normalized = y_shifted / y_factor
    
    return x_normalized, y_normalized, x_factor, y_factor

def preprocess_data_diff_fallback(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    使用差值基礎的數據預處理 - 非 Numba 版本（回退實現）
    
    Parameters:
    -----------
    x_data : np.ndarray
        X軸數據
    y_data : np.ndarray
        Y軸數據
        
    Returns:
    --------
    x_normalized : np.ndarray
        歸一化的X數據
    y_normalized : np.ndarray
        歸一化的Y數據
    x_factor : float
        X數據的縮放因子
    y_factor : float
        Y數據的縮放因子
    """
    # 平移數據到零點
    x_shifted = x_data - x_data[0]
    y_shifted = y_data - np.min(y_data)
    
    # 使用相鄰點差值的數量級眾數計算縮放因子
    x_factor = calculate_diff_magnitude_mode_fallback(x_shifted)
    y_factor = calculate_diff_magnitude_mode_fallback(y_shifted)
    
    # 避免除零和過小的值
    x_factor = max(x_factor, 1e-12)
    y_factor = max(y_factor, 1e-12)
    
    # 歸一化
    x_normalized = x_shifted / x_factor
    y_normalized = y_shifted / y_factor
    
    return x_normalized, y_normalized, x_factor, y_factor

class JosephsonFitter:
    """Enhanced Josephson junction model fitter with robust parameter estimation"""
    
    def __init__(self, config):
        self.config = config
        self.fitted_params = None
        self.param_errors = None
        self.fit_statistics = None
        
    def estimate_initial_parameters(self, x_data: np.ndarray, y_data: np.ndarray, 
                                  best_frequency: float) -> list:
        """Intelligent initial parameter estimation"""
        # Critical current estimate
        I_c_init = 3 * np.std(y_data)
        
        # Phase offset (try to align with data maximum)
        max_idx = np.argmax(y_data)
        phi_0_init = 2 * np.pi * best_frequency * x_data[max_idx] if best_frequency > 0 else 0.0
        phi_0_init = phi_0_init % (2 * np.pi)
        if phi_0_init > np.pi:
            phi_0_init -= 2 * np.pi
            
        # Frequency
        f_init = best_frequency
        
        # Transparency (moderate coupling)
        T_init = 0.5
        
        # Linear term and offset
        r_init = np.polyfit(x_data, y_data, 1)[0] if len(x_data) > 1 else 0.0
        C_init = np.mean(y_data)
        
        return [I_c_init, phi_0_init, f_init, T_init, r_init, C_init]
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray, best_frequency: float) -> dict:
        """
        Fit Josephson junction model with robust error handling
        
        Returns:
        --------
        Dictionary with fit results and statistics
        """
        try:
            # Initial parameter estimation
            p0 = self.estimate_initial_parameters(x_data, y_data, best_frequency)
            
            # Parameter bounds
            bounds = (
                [0, -np.pi, 0, 0, -np.inf, 0],  # Lower bounds
                [np.inf, np.pi, np.inf, 1, np.inf, np.inf]  # Upper bounds
            )
            
            # Perform fit
            popt, pcov = curve_fit(
                josephson_model_wrapper, 
                x_data, 
                y_data, 
                p0=p0, 
                bounds=bounds, 
                maxfev=self.config.get('MAX_ITERATIONS', 50000),
                method='trf'  # Trust Region Reflective algorithm
            )
            
            # Extract parameters
            I_c, phi_0, f, T, r, C = popt
            
            # Calculate parameter errors
            param_errors = np.sqrt(np.diag(pcov))
            
            # Generate fitted data
            y_fitted = josephson_model_numba(x_data, I_c, phi_0, f, T, r, C)
            
            # Calculate statistics
            stats = calculate_statistics_numba(y_data, y_fitted, len(popt))
            r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std = stats
            
            # Store results
            self.fitted_params = popt
            self.param_errors = param_errors
            self.fit_statistics = {
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'rmse': rmse,
                'mae': mae,
                'ss_res': ss_res,
                'residual_mean': residual_mean,
                'residual_std': residual_std
            }
            
            return {
                'success': True,
                'parameters': {
                    'I_c': I_c,
                    'phi_0': phi_0,
                    'f': f,
                    'T': T,
                    'r': r,
                    'C': C
                },
                'parameter_errors': {
                    'I_c_err': param_errors[0],
                    'phi_0_err': param_errors[1],
                    'f_err': param_errors[2],
                    'T_err': param_errors[3],
                    'r_err': param_errors[4],
                    'C_err': param_errors[5]
                },
                'statistics': self.fit_statistics,
                'fitted_data': y_fitted
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'parameters': None,
                'statistics': None,
                'fitted_data': None
            }
    
    def scale_parameters(self, x_factor: float, y_factor: float, y_min: float) -> Optional[dict]:
        """Scale parameters back to original units"""
        if self.fitted_params is None:
            return None
            
        I_c, phi_0, f, T, r, C = self.fitted_params
        
        return {
            'I_c': I_c * y_factor,
            'phi_0': phi_0,  # Phase is dimensionless
            'f': f / x_factor,
            'T': T,  # Transparency is dimensionless
            'r': r * y_factor / x_factor,
            'C': C * y_factor + y_min
        }