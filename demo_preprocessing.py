#!/usr/bin/env python3
"""
測試和展示新的 log10 量級眾數預處理方法
使用 .diff() 方法計算相鄰點差值
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# 設置 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def demonstrate_preprocessing_method(filename="435Ic.csv"):
    """
    展示新的預處理方法的工作原理
    """
    print("🧪 展示 log10 量級眾數預處理方法")
    print("=" * 50)
    
    # 載入數據
    file_path = f"data/Ic/{filename}"
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    x_data = df['y_field'].values.astype(np.float64)
    y_data = df['Ic'].values.astype(np.float64)
    
    print(f"📁 檔案: {filename}")
    print(f"📊 數據點數: {len(x_data)}")
    print()
    
    # 顯示原始數據範圍
    print("📈 原始數據範圍:")
    print(f"   X: [{x_data.min():.3e}, {x_data.max():.3e}]")
    print(f"   Y: [{y_data.min():.3e}, {y_data.max():.3e}]")
    print()
    
    # 步驟 1: 數據平移
    x_shifted = x_data - x_data[0]
    y_shifted = y_data - np.min(y_data)
    
    print("🔄 步驟 1: 數據平移")
    print(f"   X 平移後: [{x_shifted.min():.3e}, {x_shifted.max():.3e}]")
    print(f"   Y 平移後: [{y_shifted.min():.3e}, {y_shifted.max():.3e}]")
    print()
    
    # 步驟 2: 計算相鄰點差值
    x_diffs = np.diff(x_shifted)
    y_diffs = np.diff(y_shifted)
    
    print("📏 步驟 2: 計算相鄰點差值 (.diff())")
    print(f"   X 差值數量: {len(x_diffs)}")
    print(f"   Y 差值數量: {len(y_diffs)}")
    print(f"   X 差值範圍: [{x_diffs.min():.3e}, {x_diffs.max():.3e}]")
    print(f"   Y 差值範圍: [{y_diffs.min():.3e}, {y_diffs.max():.3e}]")
    print()
    
    # 步驟 3: 取絕對值，移除零值
    x_abs_diffs = np.abs(x_diffs)
    y_abs_diffs = np.abs(y_diffs)
    
    x_positive_diffs = x_abs_diffs[x_abs_diffs > 0]
    y_positive_diffs = y_abs_diffs[y_abs_diffs > 0]
    
    print("🔢 步驟 3: 處理差值")
    print(f"   X 正值差值數量: {len(x_positive_diffs)}")
    print(f"   Y 正值差值數量: {len(y_positive_diffs)}")
    print()
    
    # 步驟 4: 計算 log10 數量級
    x_log_diffs = np.log10(x_positive_diffs)
    y_log_diffs = np.log10(y_positive_diffs)
    
    x_magnitudes = np.floor(x_log_diffs)
    y_magnitudes = np.floor(y_log_diffs)
    
    print("📐 步驟 4: 計算 log10 數量級")
    print(f"   X 數量級範圍: [{x_magnitudes.min():.0f}, {x_magnitudes.max():.0f}]")
    print(f"   Y 數量級範圍: [{y_magnitudes.min():.0f}, {y_magnitudes.max():.0f}]")
    
    # 顯示數量級分布
    x_unique, x_counts = np.unique(x_magnitudes, return_counts=True)
    y_unique, y_counts = np.unique(y_magnitudes, return_counts=True)
    
    print(f"   X 數量級分布:")
    for mag, count in zip(x_unique, x_counts):
        print(f"     10^{mag:.0f}: {count} 次")
    
    print(f"   Y 數量級分布:")
    for mag, count in zip(y_unique, y_counts):
        print(f"     10^{mag:.0f}: {count} 次")
    print()
    
    # 步驟 5: 計算眾數 (mode())
    x_mode_result = stats.mode(x_magnitudes, keepdims=True)
    y_mode_result = stats.mode(y_magnitudes, keepdims=True)
    
    x_mode_magnitude = x_mode_result.mode[0]
    y_mode_magnitude = y_mode_result.mode[0]
    
    x_factor = 10.0 ** x_mode_magnitude
    y_factor = 10.0 ** y_mode_magnitude
    
    print("🎯 步驟 5: 計算眾數 (mode())")
    print(f"   X 眾數數量級: 10^{x_mode_magnitude:.0f}")
    print(f"   Y 眾數數量級: 10^{y_mode_magnitude:.0f}")
    print(f"   X 縮放因子: {x_factor:.3e}")
    print(f"   Y 縮放因子: {y_factor:.3e}")
    print()
    
    # 步驟 6: 歸一化
    x_normalized = x_shifted / x_factor
    y_normalized = y_shifted / y_factor
    
    print("✅ 步驟 6: 最終歸一化")
    print(f"   X 歸一化範圍: [{x_normalized.min():.6f}, {x_normalized.max():.6f}]")
    print(f"   Y 歸一化範圍: [{y_normalized.min():.6f}, {y_normalized.max():.6f}]")
    print()
    
    # 與系統實際使用的方法比較
    try:
        from cpr.josephson_model import preprocess_data_numba
        
        # 使用系統的預處理方法
        x_norm_sys, y_norm_sys, x_fact_sys, y_fact_sys = preprocess_data_numba(x_data, y_data)
        
        print("🔄 與系統方法比較:")
        print(f"   系統 X 縮放因子: {x_fact_sys:.3e}")
        print(f"   系統 Y 縮放因子: {y_fact_sys:.3e}")
        print(f"   手動 X 縮放因子: {x_factor:.3e}")
        print(f"   手動 Y 縮放因子: {y_factor:.3e}")
        
        if abs(x_fact_sys - x_factor) < 1e-10 and abs(y_fact_sys - y_factor) < 1e-10:
            print("✅ 完全一致!")
        else:
            print("⚠️ 有差異（可能是實現細節不同）")
            
    except Exception as e:
        print(f"⚠️ 無法比較系統方法: {e}")
    
    print()
    print("🎉 預處理方法展示完成!")

def main():
    """主程式"""
    filename = sys.argv[1] if len(sys.argv) > 1 else "435Ic.csv"
    demonstrate_preprocessing_method(filename)

if __name__ == "__main__":
    main()
