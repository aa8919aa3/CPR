#!/usr/bin/env python3
"""
修復預處理邏輯，處理只有2個唯一值的情況
"""

import numpy as np
import pandas as pd
import os

def improved_validate_data_array(data, name="data", min_unique=2, tolerance=1e-15):
    """
    改進的數據驗證函數，處理數值精度問題
    """
    if not np.all(np.isfinite(data)):
        return False, f"{name}: contains non-finite values"
    
    unique_values = np.unique(data)
    n_unique = len(unique_values)
    
    if n_unique < min_unique:
        # 如果唯一值不足，檢查是否由於數值精度問題
        if n_unique == 1:
            return False, f"{name}: constant data (all values same: {unique_values[0]:.6e})"
        else:
            # 檢查唯一值之間的差異是否足夠大
            min_diff = np.min(np.diff(unique_values))
            max_val = np.max(np.abs(unique_values))
            relative_diff = min_diff / max_val if max_val > 0 else min_diff
            
            if relative_diff < tolerance:
                return False, f"{name}: values too close (relative diff: {relative_diff:.2e})"
    
    return True, f"{name}: valid ({n_unique} unique values)"

def improved_preprocess_data(x_data, y_data, remove_first_n=10):
    """
    改進的預處理函數，更好地處理邊界情況
    """
    print(f"原始數據: x={len(x_data)}, y={len(y_data)}")
    print(f"x 範圍: {x_data.min():.6e} 到 {x_data.max():.6e}")
    print(f"y 範圍: {y_data.min():.6e} 到 {y_data.max():.6e}")
    print(f"y 唯一值: {len(np.unique(y_data))}")
    
    # 移除前 N 個點
    if len(x_data) > remove_first_n:
        x_data = x_data[remove_first_n:]
        y_data = y_data[remove_first_n:]
        print(f"移除前 {remove_first_n} 個點後: {len(x_data)} 個點")
    
    # 在移除點後檢查 y 數據的變化性
    y_unique_after_removal = np.unique(y_data)
    print(f"移除點後 y 唯一值: {len(y_unique_after_removal)}")
    print(f"y 唯一值: {y_unique_after_removal}")
    
    if len(y_unique_after_removal) < 2:
        return None, None, "移除前N個點後y數據變為常數"
    
    # 平移數據到原點
    x_min, y_min = x_data.min(), y_data.min()
    x_shifted = x_data - x_min
    y_shifted = y_data - y_min
    
    print(f"平移後 x 範圍: {x_shifted.min():.6e} 到 {x_shifted.max():.6e}")
    print(f"平移後 y 範圍: {y_shifted.min():.6e} 到 {y_shifted.max():.6e}")
    print(f"平移後 y 唯一值: {len(np.unique(y_shifted))}")
    
    # 計算縮放因子
    # 對於 x 數據
    x_range = x_shifted.max() - x_shifted.min()
    x_factor = max(x_range, 1e-6)  # 確保不會太小
    
    # 對於 y 數據 - 改進的邏輯
    y_range = y_shifted.max() - y_shifted.min()
    if y_range > 1e-15:  # 如果有足夠的範圍
        y_factor = y_range
    else:
        # 如果範圍太小，使用其他方法
        y_std = np.std(y_shifted)
        if y_std > 1e-15:
            y_factor = y_std
        else:
            # 使用唯一值之間的最小差異
            y_unique = np.unique(y_shifted)
            if len(y_unique) > 1:
                min_diff = np.min(np.diff(y_unique))
                y_factor = max(min_diff, 1e-6)
            else:
                y_factor = 1.0
    
    print(f"縮放因子: x_factor={x_factor:.6e}, y_factor={y_factor:.6e}")
    
    # 歸一化
    x_normalized = x_shifted / x_factor
    y_normalized = y_shifted / y_factor
    
    print(f"歸一化後 x 範圍: {x_normalized.min():.6e} 到 {x_normalized.max():.6e}")
    print(f"歸一化後 y 範圍: {y_normalized.min():.6e} 到 {y_normalized.max():.6e}")
    print(f"歸一化後 x 唯一值: {len(np.unique(x_normalized))}")
    print(f"歸一化後 y 唯一值: {len(np.unique(y_normalized))}")
    
    return x_normalized, y_normalized, "成功"

def test_improved_preprocessing():
    """測試改進的預處理邏輯"""
    files = [
        '/Users/albert-mac/Code/GitHub/CPR/data/Ic/228Ic.csv',
        '/Users/albert-mac/Code/GitHub/CPR/data/Ic/130Ic-.csv'
    ]
    
    for file_path in files:
        print(f"\n{'='*70}")
        print(f"測試文件: {os.path.basename(file_path)}")
        print('='*70)
        
        try:
            # 讀取數據
            df = pd.read_csv(file_path)
            x_data = df.iloc[:, 0].values
            y_data = df.iloc[:, 1].values
            
            # 應用改進的預處理
            x_norm, y_norm, message = improved_preprocess_data(x_data, y_data)
            
            if x_norm is not None and y_norm is not None:
                # 使用改進的驗證函數
                x_valid, x_msg = improved_validate_data_array(x_norm, "x_normalized")
                y_valid, y_msg = improved_validate_data_array(y_norm, "y_normalized")
                
                print(f"\n驗證結果:")
                print(f"x: {x_msg}")
                print(f"y: {y_msg}")
                
                if x_valid and y_valid:
                    print("✅ 預處理成功!")
                else:
                    print("❌ 驗證失敗")
            else:
                print(f"❌ 預處理失敗: {message}")
                
        except Exception as e:
            print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    test_improved_preprocessing()
