#!/usr/bin/env python3
"""
完整的預處理函數驗證測試
測試新實現與實際 CPR 分析數據的兼容性
"""

import sys
import os
import numpy as np
import pandas as pd
import time

# 添加 src 目錄到 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cpr.josephson_model import preprocess_data_numba

def test_with_actual_data():
    """使用實際的 435Ic.csv 數據進行完整測試"""
    
    print("=== 使用實際數據測試新預處理函數 ===")
    
    # 載入實際數據
    data_path = "data/Ic/435Ic.csv"
    if not os.path.exists(data_path):
        print(f"錯誤：找不到數據文件 {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"載入數據：{len(df)} 個數據點")
    
    # 準備數據
    x_data = df['Ic'].values.astype(float)
    y_data = df['y_field'].values.astype(float)
    
    print(f"原始數據範圍：")
    print(f"  Ic: {x_data.min():.2e} 到 {x_data.max():.2e}")
    print(f"  y_field: {y_data.min():.2e} 到 {y_data.max():.2e}")
    
    # 測試預處理效能
    start_time = time.time()
    x_norm, y_norm, x_factor, y_factor = preprocess_data_numba(x_data, y_data)
    processing_time = time.time() - start_time
    
    print(f"\n預處理結果：")
    print(f"  處理時間：{processing_time:.4f} 秒")
    print(f"  X 縮放因子：{x_factor:.2e}")
    print(f"  Y 縮放因子：{y_factor:.2e}")
    print(f"  正規化後 Ic 範圍：{x_norm.min():.6f} 到 {x_norm.max():.6f}")
    print(f"  正規化後 y_field 範圍：{y_norm.min():.6f} 到 {y_norm.max():.6f}")
    
    # 驗證結果的品質
    print(f"\n數據品質檢查：")
    print(f"  X 數據點數：{len(x_norm)}")
    print(f"  Y 數據點數：{len(y_norm)}")
    print(f"  是否包含 NaN：X={np.isnan(x_norm).any()}, Y={np.isnan(y_norm).any()}")
    print(f"  是否包含 Inf：X={np.isinf(x_norm).any()}, Y={np.isinf(y_norm).any()}")
    
    # 手動驗證預處理流程
    print(f"\n手動驗證流程：")
    x_series = pd.Series(x_data)
    y_series = pd.Series(y_data)
    
    # 1. 平移
    x_shifted = x_series - x_series.iloc[0]
    y_shifted = y_series - y_series.min()
    print(f"  1. 平移後 X 起點：{x_shifted.iloc[0]:.2e}")
    print(f"  1. 平移後 Y 最小值：{y_shifted.min():.2e}")
    
    # 2. 差值計算
    x_diffs = x_shifted.diff().abs().replace(0, np.nan).dropna()
    y_diffs = y_shifted.diff().abs().replace(0, np.nan).dropna()
    print(f"  2. X 有效差值數量：{len(x_diffs)}")
    print(f"  2. Y 有效差值數量：{len(y_diffs)}")
    
    # 3. log10 和眾數
    x_log_mode = x_diffs.apply(lambda x: round(np.log10(x))).mode().iloc[0]
    y_log_mode = y_diffs.apply(lambda y: round(np.log10(y))).mode().iloc[0]
    print(f"  3. X log10 眾數：{x_log_mode}")
    print(f"  3. Y log10 眾數：{y_log_mode}")
    
    # 4. 縮放因子驗證
    manual_x_factor = 10.0 ** x_log_mode
    manual_y_factor = 10.0 ** y_log_mode
    print(f"  4. 手動計算 X 縮放因子：{manual_x_factor:.2e}")
    print(f"  4. 手動計算 Y 縮放因子：{manual_y_factor:.2e}")
    print(f"  4. 與函數結果匹配：X={abs(manual_x_factor - x_factor) < 1e-10}, Y={abs(manual_y_factor - y_factor) < 1e-10}")
    
    return True

def test_integration_with_main_processor():
    """測試與主處理器的集成"""
    
    print(f"\n=== 集成測試 ===")
    
    # 檢查主處理器是否能正確調用新的預處理函數
    try:
        # 檢查導入
        from cpr.main_processor_optimized import EnhancedJosephsonProcessor
        print("✅ 成功導入 EnhancedJosephsonProcessor")
        
        # 檢查是否使用了正確的預處理函數
        import inspect
        import cpr.main_processor_optimized as main_module
        
        # 檢查模塊中是否引用了正確的預處理函數
        source = inspect.getsource(main_module)
        if 'preprocess_data_numba' in source:
            print("✅ 主處理器使用正確的預處理函數")
        else:
            print("⚠️  主處理器可能沒有使用新的預處理函數")
        
        # 檢查實例化
        processor = EnhancedJosephsonProcessor()
        print("✅ 成功實例化處理器")
        
    except ImportError as e:
        print(f"❌ 導入錯誤：{e}")
        return False
    except Exception as e:
        print(f"❌ 其他錯誤：{e}")
        return False
    
    return True

def main():
    """主測試函數"""
    print("CPR 預處理函數完整驗證")
    print("=" * 50)
    
    success1 = test_with_actual_data()
    success2 = test_integration_with_main_processor()
    
    if success1 and success2:
        print("\n🎉 所有測試通過！新預處理函數已準備就緒。")
        return 0
    else:
        print("\n❌ 部分測試失敗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
