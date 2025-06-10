#!/usr/bin/env python3
"""
檢查 435Ic.csv 中兩個頻率值的關係
"""

import pandas as pd
import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import sys
from pathlib import Path

# 添加 src 目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from cpr.josephson_model import preprocess_data_numba

def analyze_frequency_difference():
    """分析兩個頻率值的來源和關係"""
    
    print("=" * 60)
    print("435Ic.csv 頻率分析調試")
    print("=" * 60)
    
    # 加載數據
    file_path = "data/Ic/435Ic.csv"
    df = pd.read_csv(file_path)
    x_data = df['y_field'].values.astype(np.float64)
    y_data = df['Ic'].values.astype(np.float64)
    
    print(f"原始數據點數：{len(x_data)}")
    print(f"X 數據範圍：{x_data.min():.6e} 到 {x_data.max():.6e}")
    print(f"Y 數據範圍：{y_data.min():.6e} 到 {y_data.max():.6e}")
    
    # 清理數據（複製主處理器的邏輯）
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    # 移除前10個數據點
    if len(x_data) >= 10:
        x_data = x_data[10:]
        y_data = y_data[10:]
    
    print(f"處理後數據點數：{len(x_data)}")
    
    # 數據歸一化（複製主處理器的邏輯）
    x_data_normalized, y_data_normalized, x_factor, y_factor = preprocess_data_numba(x_data, y_data)
    
    print(f"\n歸一化因子：")
    print(f"x_factor：{x_factor:.6e}")
    print(f"y_factor：{y_factor:.6e}")
    print(f"歸一化後 X 範圍：{x_data_normalized.min():.6e} 到 {x_data_normalized.max():.6e}")
    print(f"歸一化後 Y 範圍：{y_data_normalized.min():.6e} 到 {y_data_normalized.max():.6e}")
    
    # Lomb-Scargle 功率譜分析
    print(f"\n=== Lomb-Scargle 功率譜分析 ===")
    
    # 頻率範圍計算（複製主處理器邏輯）
    median_diff = np.median(np.diff(x_data_normalized))
    print(f"median_diff：{median_diff:.6e}")
    
    freq_min = max(1e-5, 1.0 / (len(x_data_normalized) * median_diff))
    freq_max = 1 / (2 * median_diff)
    
    print(f"頻率範圍：{freq_min:.6e} 到 {freq_max:.6e}")
    
    frequencies = np.linspace(freq_min, freq_max, 10000)
    
    # 計算功率譜
    ls = LombScargle(x_data_normalized, y_data_normalized)
    power = ls.power(frequencies)
    
    # 尋找峰值
    height_threshold = np.max(power) * 0.1
    peaks, properties = find_peaks(power, height=height_threshold, distance=100)
    
    # 找到最強峰值
    peak_frequencies = frequencies[peaks]
    peak_powers = power[peaks]
    
    if len(peak_powers) > 0:
        sorted_indices = np.argsort(peak_powers)[::-1]
        top_frequencies = peak_frequencies[sorted_indices]
        best_frequency_ls = top_frequencies[0]
        
        print(f"Lomb-Scargle 最佳頻率：{best_frequency_ls:.6e} Hz")
        print(f"對應週期：{1/best_frequency_ls:.6f}")
    else:
        best_frequency_ls = frequencies[np.argmax(power)]
        print(f"Lomb-Scargle 全局最大頻率：{best_frequency_ls:.6e} Hz")
    
    # 從分析結果中獲取擬合頻率（這裡我們使用分析腳本的結果）
    # 根據之前的輸出結果
    f_opt_fitted = 6.5754e+04  # 從分析結果中得到的擬合頻率
    f_scaled = f_opt_fitted / x_factor  # 換算到原始單位
    
    print(f"\n=== 擬合結果分析 ===")
    print(f"擬合得到的頻率（歸一化數據）：{f_opt_fitted:.6e} Hz")
    print(f"換算到原始單位：{f_scaled:.6e} Hz")
    
    print(f"\n=== 頻率對比 ===")
    print(f"Lomb-Scargle 頻率：{best_frequency_ls:.6e} Hz")
    print(f"擬合換算頻率：    {f_scaled:.6e} Hz")
    print(f"頻率比值：        {f_opt_fitted / best_frequency_ls:.2f}")
    print(f"差異：            {abs(best_frequency_ls - f_scaled):.6e} Hz")
    
    if abs(best_frequency_ls - f_scaled) < best_frequency_ls * 0.1:
        print("✅ 兩個頻率值基本一致（差異 < 10%）")
    else:
        print("❌ 兩個頻率值存在顯著差異")
        
    print(f"\n=== 結論 ===")
    print(f"• phase_folded_with_drift 圖表顯示的是 Lomb-Scargle 分析的頻率")
    print(f"• fitted_curve_normalized_plot 顯示的是非線性擬合的頻率（歸一化數據）")
    print(f"• 正確的物理頻率應該是換算後的值：{f_scaled:.6e} Hz")
    print(f"• x_factor = {x_factor:.6e} 是造成數值差異的歸一化因子")

if __name__ == "__main__":
    analyze_frequency_difference()
