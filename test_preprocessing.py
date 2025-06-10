#!/usr/bin/env python3
"""
測試改進的數據預處理函數
"""
import numpy as np
import sys
import os

# 添加 src 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.cpr.josephson_model import preprocess_data_numba, preprocess_data_fallback

def test_preprocess_mode_magnitude():
    """測試使用數量級眾數的預處理函數"""
    print("=== 測試改進的數據預處理函數 ===")
    
    # 創建測試數據
    print("\n1. 測試正常數據...")
    x_data = np.linspace(0, 1000, 100)  # 數量級約為 10^2
    y_data = np.sin(x_data) * 0.001 + 0.002  # 數量級約為 10^-3
    
    # 測試 Numba 版本
    x_norm, y_norm, x_factor, y_factor = preprocess_data_numba(x_data, y_data)
    print(f"Numba版本:")
    print(f"  x_factor: {x_factor:.2e} (預期約為 10^2)")
    print(f"  y_factor: {y_factor:.2e} (預期約為 10^-3)")
    print(f"  x_normalized 範圍: [{np.min(x_norm):.3f}, {np.max(x_norm):.3f}]")
    print(f"  y_normalized 範圍: [{np.min(y_norm):.3f}, {np.max(y_norm):.3f}]")
    
    # 測試回退版本
    x_norm_fb, y_norm_fb, x_factor_fb, y_factor_fb = preprocess_data_fallback(x_data, y_data)
    print(f"回退版本:")
    print(f"  x_factor: {x_factor_fb:.2e}")
    print(f"  y_factor: {y_factor_fb:.2e}")
    print(f"  結果一致性: {np.allclose(x_norm, x_norm_fb) and np.allclose(y_norm, y_norm_fb)}")
    
    # 測試邊界情況
    print("\n2. 測試邊界情況...")
    
    # 全零數據
    print("  a) 全零 y 數據:")
    y_zeros = np.zeros_like(x_data)
    try:
        x_norm_z, y_norm_z, x_factor_z, y_factor_z = preprocess_data_numba(x_data, y_zeros)
        print(f"    處理成功: x_factor={x_factor_z:.2e}, y_factor={y_factor_z:.2e}")
    except Exception as e:
        print(f"    錯誤: {e}")
    
    # 負值數據
    print("  b) 包含負值的數據:")
    y_negative = np.linspace(-0.001, 0.001, 100)
    try:
        x_norm_n, y_norm_n, x_factor_n, y_factor_n = preprocess_data_numba(x_data, y_negative)
        print(f"    處理成功: x_factor={x_factor_n:.2e}, y_factor={y_factor_n:.2e}")
    except Exception as e:
        print(f"    錯誤: {e}")
    
    # 極小值數據
    print("  c) 極小值數據:")
    y_tiny = np.full_like(x_data, 1e-15)
    try:
        x_norm_t, y_norm_t, x_factor_t, y_factor_t = preprocess_data_numba(x_data, y_tiny)
        print(f"    處理成功: x_factor={x_factor_t:.2e}, y_factor={y_factor_t:.2e}")
    except Exception as e:
        print(f"    錯誤: {e}")
    
    print("\n3. 與舊方法比較...")
    # 模擬舊的預處理方法
    x_shifted_old = x_data - x_data[0]
    y_shifted_old = y_data - np.min(y_data)
    x_factor_old = abs(x_shifted_old[2] - x_shifted_old[1]) if len(x_shifted_old) > 2 else 1.0
    y_factor_old = abs(y_shifted_old[2] - y_shifted_old[1]) if len(y_shifted_old) > 2 else 1.0
    x_factor_old = max(x_factor_old, 1e-12)
    y_factor_old = max(y_factor_old, 1e-12)
    
    print(f"舊方法 x_factor: {x_factor_old:.2e}")
    print(f"新方法 x_factor: {x_factor:.2e}")
    print(f"舊方法 y_factor: {y_factor_old:.2e}")
    print(f"新方法 y_factor: {y_factor:.2e}")
    
    improvement_ratio_x = x_factor / x_factor_old
    improvement_ratio_y = y_factor / y_factor_old
    print(f"改進比率 - x: {improvement_ratio_x:.2f}, y: {improvement_ratio_y:.2f}")

if __name__ == "__main__":
    test_preprocess_mode_magnitude()
