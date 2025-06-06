#!/usr/bin/env python3
"""
詳細跟蹤預處理過程，找出真正的問題所在
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def debug_preprocessing_step_by_step(csv_file):
    """逐步調試預處理過程"""
    print(f"\n=== 逐步調試: {Path(csv_file).name} ===")
    
    # 讀取原始數據
    data = pd.read_csv(csv_file)
    x_data = data.iloc[:, 0].values.astype(np.float64)
    y_data = data.iloc[:, 1].values.astype(np.float64)
    
    print(f"1. 原始數據:")
    print(f"   x_data 長度: {len(x_data)}")
    print(f"   y_data 長度: {len(y_data)}")
    print(f"   x_data 範圍: {np.min(x_data):.6e} 到 {np.max(x_data):.6e}")
    print(f"   y_data 範圍: {np.min(y_data):.6e} 到 {np.max(y_data):.6e}")
    
    # 移除 NaN 和無限值
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    print(f"2. 有效性檢查:")
    print(f"   有效數據點: {np.sum(valid_mask)}/{len(x_data)}")
    
    if not np.any(valid_mask):
        print("   ❌ 沒有有效數據點")
        return
    
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    # 檢查數據點數量
    print(f"3. 數據點檢查:")
    print(f"   過濾後長度: {len(x_data)}")
    
    if len(x_data) < 20:
        print("   ❌ 數據點不足 20 個")
        return
        
    # 移除前10個數據點
    if len(x_data) >= 10:
        x_data = x_data[10:]
        y_data = y_data[10:]
        print(f"4. 移除前10個點後:")
        print(f"   剩餘長度: {len(x_data)}")
    
    if len(x_data) < 10:
        print("   ❌ 移除前10個點後數據不足")
        return
    
    # 開始預處理 - 精確模擬 preprocess_data_numba
    print(f"5. 預處理步驟:")
    
    # 步驟 1: 數據平移
    x_data_shifted = x_data - x_data[0]
    y_data_shifted = y_data - np.min(y_data)
    
    print(f"   平移後 x_data: {np.min(x_data_shifted):.6e} 到 {np.max(x_data_shifted):.6e}")
    print(f"   平移後 y_data: {np.min(y_data_shifted):.6e} 到 {np.max(y_data_shifted):.6e}")
    print(f"   y_data_shifted 唯一值: {len(np.unique(y_data_shifted))}")
    
    # 步驟 2: 計算縮放因子
    print(f"6. 縮放因子計算:")
    
    # x_factor 計算
    if len(x_data_shifted) > 2:
        x_factor = abs(x_data_shifted[2] - x_data_shifted[1])
        print(f"   初始 x_factor (|x[2]-x[1]|): {x_factor:.15e}")
    else:
        x_factor = 1.0
        print(f"   初始 x_factor (默認): {x_factor}")
    
    if x_factor < 1e-12:
        x_std = np.std(x_data_shifted)
        print(f"   x_factor 太小，嘗試標準差: {x_std:.15e}")
        if x_std > 1e-12:
            x_factor = x_std
            print(f"   使用標準差作為 x_factor: {x_factor:.15e}")
        else:
            x_range = np.max(x_data_shifted) - np.min(x_data_shifted)
            print(f"   標準差也太小，嘗試範圍: {x_range:.15e}")
            x_factor = x_range if x_range > 1e-12 else 1.0
            print(f"   最終 x_factor: {x_factor:.15e}")
    
    # y_factor 計算 - 關鍵部分
    if len(y_data_shifted) > 2:
        y_factor = abs(y_data_shifted[2] - y_data_shifted[1])
        print(f"   初始 y_factor (|y[2]-y[1]|): {y_factor:.15e}")
        print(f"   y_data_shifted[0]: {y_data_shifted[0]:.15e}")
        print(f"   y_data_shifted[1]: {y_data_shifted[1]:.15e}")
        print(f"   y_data_shifted[2]: {y_data_shifted[2]:.15e}")
    else:
        y_factor = 1.0
        print(f"   初始 y_factor (默認): {y_factor}")
    
    if y_factor < 1e-12:
        y_std = np.std(y_data_shifted)
        print(f"   y_factor 太小，嘗試標準差: {y_std:.15e}")
        if y_std > 1e-12:
            y_factor = y_std
            print(f"   使用標準差作為 y_factor: {y_factor:.15e}")
        else:
            y_range = np.max(y_data_shifted) - np.min(y_data_shifted)
            print(f"   標準差也太小，嘗試範圍: {y_range:.15e}")
            y_factor = y_range if y_range > 1e-12 else 1.0
            print(f"   最終 y_factor: {y_factor:.15e}")
    
    # 最終安全檢查
    x_factor = max(x_factor, 1e-6)
    y_factor = max(y_factor, 1e-6)
    print(f"   安全檢查後 x_factor: {x_factor:.15e}")
    print(f"   安全檢查後 y_factor: {y_factor:.15e}")
    
    # 步驟 3: 歸一化
    x_data_normalized = x_data_shifted / x_factor
    y_data_normalized = y_data_shifted / y_factor
    
    print(f"7. 歸一化結果:")
    print(f"   x_normalized 範圍: {np.min(x_data_normalized):.6e} 到 {np.max(x_data_normalized):.6e}")
    print(f"   y_normalized 範圍: {np.min(y_data_normalized):.6e} 到 {np.max(y_data_normalized):.6e}")
    print(f"   x_normalized 唯一值: {len(np.unique(x_data_normalized))}")
    print(f"   y_normalized 唯一值: {len(np.unique(y_data_normalized))}")
    
    # 步驟 4: 驗證
    print(f"8. 驗證步驟:")
    
    # 檢查 NaN/inf
    x_finite = np.isfinite(x_data_normalized).all()
    y_finite = np.isfinite(y_data_normalized).all()
    print(f"   x_normalized 全部有限: {x_finite}")
    print(f"   y_normalized 全部有限: {y_finite}")
    
    if not x_finite:
        nan_count = np.isnan(x_data_normalized).sum()
        inf_count = np.isinf(x_data_normalized).sum()
        print(f"   x_normalized: {nan_count} NaN, {inf_count} inf")
    
    if not y_finite:
        nan_count = np.isnan(y_data_normalized).sum()
        inf_count = np.isinf(y_data_normalized).sum()
        print(f"   y_normalized: {nan_count} NaN, {inf_count} inf")
    
    # 檢查唯一值
    x_unique_count = len(np.unique(x_data_normalized))
    y_unique_count = len(np.unique(y_data_normalized))
    
    print(f"   x_normalized 唯一值檢查: {x_unique_count} {'≥' if x_unique_count >= 2 else '<'} 2")
    print(f"   y_normalized 唯一值檢查: {y_unique_count} {'≥' if y_unique_count >= 2 else '<'} 2")
    
    if x_unique_count < 2:
        print(f"   ❌ x_normalized 被認為是常數")
        print(f"   前5個 x_normalized 值: {x_data_normalized[:5]}")
    
    if y_unique_count < 2:
        print(f"   ❌ y_normalized 被認為是常數")
        print(f"   前5個 y_normalized 值: {y_data_normalized[:5]}")
        print(f"   全部 y_normalized 唯一值: {np.unique(y_data_normalized)}")

def main():
    """主函數"""
    print("詳細跟蹤預處理過程，找出真正的問題所在")
    print("="*70)
    
    problem_files = [
        "data/Ic/228Ic.csv",
        "data/Ic/130Ic-.csv"
    ]
    
    for filename in problem_files:
        csv_file = project_root / filename
        if csv_file.exists():
            debug_preprocessing_step_by_step(str(csv_file))
        else:
            print(f"❌ 文件不存在: {filename}")
    
    print(f"\n{'='*70}")
    print("調試完成")

if __name__ == "__main__":
    main()
