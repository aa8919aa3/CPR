#!/usr/bin/env python3
"""
測試統一的預處理方法
使用您指定的流程：pandas .diff() + log10 + mode()
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 設置 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_unified_preprocessing():
    """測試統一的預處理方法"""
    print("🧪 測試統一的預處理方法")
    print("=" * 50)
    
    # 讀取 435Ic.csv 資料
    file_path = "data/Ic/435Ic.csv"
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在: {file_path}")
        return
    
    # 讀取資料
    df = pd.read_csv(file_path)
    x_data = df['y_field']  # pandas Series
    y_data = df['Ic']       # pandas Series
    
    print(f"📁 檔案: {file_path}")
    print(f"📊 數據點數: {len(x_data)}")
    print()
    
    # 顯示原始數據範圍
    print("📈 原始數據範圍:")
    print(f"   X: [{x_data.min():.6e}, {x_data.max():.6e}]")
    print(f"   Y: [{y_data.min():.6e}, {y_data.max():.6e}]")
    print()
    
    # 按照您指定的流程進行預處理
    print("🔄 執行您指定的預處理流程:")
    
    # 1. 平移資料使起點或最小值為 0
    x_shifted = x_data - x_data.iloc[0]
    y_shifted = y_data - y_data.min()
    
    print("步驟 1: 平移資料")
    print(f"   X 平移後範圍: [{x_shifted.min():.6e}, {x_shifted.max():.6e}]")
    print(f"   Y 平移後範圍: [{y_shifted.min():.6e}, {y_shifted.max():.6e}]")
    print()
    
    # 2. 計算差值、log10 數量級、四捨五入後取眾數
    # 3. 縮放因子
    print("步驟 2-3: 計算縮放因子")
    
    # X 方向
    x_diffs = x_shifted.diff().abs().replace(0, np.nan).dropna()
    x_log_values = x_diffs.apply(lambda x: round(np.log10(x)) if x > 0 else 0)
    x_mode = x_log_values.mode()
    x_factor = 10.0 ** x_mode.iloc[0]
    
    print(f"   X 差值數量: {len(x_diffs)}")
    print(f"   X 數量級眾數: {x_mode.iloc[0]:.0f}")
    print(f"   X 縮放因子: {x_factor:.6e}")
    
    # Y 方向
    y_diffs = y_shifted.diff().abs().replace(0, np.nan).dropna()
    y_log_values = y_diffs.apply(lambda y: round(np.log10(y)) if y > 0 else 0)
    y_mode = y_log_values.mode()
    y_factor = 10.0 ** y_mode.iloc[0]
    
    print(f"   Y 差值數量: {len(y_diffs)}")
    print(f"   Y 數量級眾數: {y_mode.iloc[0]:.0f}")
    print(f"   Y 縮放因子: {y_factor:.6e}")
    print()
    
    # 4. 正規化資料
    x_normalized = x_shifted / x_factor
    y_normalized = y_shifted / y_factor
    
    print("步驟 4: 正規化資料")
    print(f"   X 正規化範圍: [{x_normalized.min():.6f}, {x_normalized.max():.6f}]")
    print(f"   Y 正規化範圍: [{y_normalized.min():.6f}, {y_normalized.max():.6f}]")
    print()
    
    # 與系統方法比較
    try:
        from cpr.josephson_model import preprocess_data_numba
        
        # 轉換為 numpy array 以供系統方法使用
        x_array = x_data.values
        y_array = y_data.values
        
        x_norm_sys, y_norm_sys, x_fact_sys, y_fact_sys = preprocess_data_numba(x_array, y_array)
        
        print("🔄 與更新後的系統方法比較:")
        print(f"   手動 X 縮放因子: {x_factor:.6e}")
        print(f"   系統 X 縮放因子: {x_fact_sys:.6e}")
        print(f"   手動 Y 縮放因子: {y_factor:.6e}")
        print(f"   系統 Y 縮放因子: {y_fact_sys:.6e}")
        
        x_match = abs(x_factor - x_fact_sys) < 1e-10
        y_match = abs(y_factor - y_fact_sys) < 1e-10
        
        if x_match and y_match:
            print("✅ 完全一致！")
        else:
            print(f"⚠️ 有差異:")
            print(f"   X 差異: {abs(x_factor - x_fact_sys):.2e}")
            print(f"   Y 差異: {abs(y_factor - y_fact_sys):.2e}")
            
    except Exception as e:
        print(f"⚠️ 無法比較系統方法: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🎉 預處理測試完成!")

def main():
    """主程式"""
    test_unified_preprocessing()

if __name__ == "__main__":
    main()
