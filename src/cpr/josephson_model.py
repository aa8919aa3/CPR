"""
Josephson Junction Physics Model with Numba Optimization
"""
import numpy as np
import numba
from numba import jit, prange
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings

# Suppress numba warnings
warnings.filterwarnings('ignore', category=numba.NumbaWarning)

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
    
    return r_squared, adj_r_squared, rmse, mae, ss_res, residual_mean, residual_std

@jit(nopython=True, cache=True, fastmath=True)
def preprocess_data_numba(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fast data preprocessing using Numba
    
    Returns:
    --------
    x_normalized, y_normalized, x_factor, y_factor
    """
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
    
    def scale_parameters(self, x_factor: float, y_factor: float, y_min: float) -> dict:
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