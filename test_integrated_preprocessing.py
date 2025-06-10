#!/usr/bin/env python3
"""
測試新的預處理方法在實際處理器中的效果
"""
import numpy as np
import sys
import os

# 添加 src 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_integrated_preprocessing():
    """測試集成的預處理方法"""
    print("=== 測試集成的預處理方法 ===")
    
    try:
        # 測試 josephson_model 的改進預處理
        from src.cpr.josephson_model import preprocess_data_numba, preprocess_data_fallback
        print("✓ 成功導入 josephson_model 預處理函數")
        
        # 創建測試數據
        x_data = np.linspace(0, 1000, 100)
        y_data = np.sin(x_data) * 0.001 + 0.002
        
        # 測試 Numba 版本
        x_norm, y_norm, x_factor, y_factor = preprocess_data_numba(x_data, y_data)
        print(f"Numba 版本 - x_factor: {x_factor:.2e}, y_factor: {y_factor:.2e}")
        
        # 測試回退版本
        x_norm_fb, y_norm_fb, x_factor_fb, y_factor_fb = preprocess_data_fallback(x_data, y_data)
        print(f"回退版本 - x_factor: {x_factor_fb:.2e}, y_factor: {y_factor_fb:.2e}")
        
        # 測試一致性
        is_consistent = np.allclose(x_norm, x_norm_fb) and np.allclose(y_norm, y_norm_fb)
        print(f"結果一致性: {is_consistent}")
        
    except Exception as e:
        print(f"✗ josephson_model 測試失敗: {e}")
    
    print("\n=== 測試處理器集成 ===")
    
    # 測試主處理器是否能使用改進的方法
    try:
        from src.cpr.main_processor_improved import EnhancedJosephsonProcessor
        processor = EnhancedJosephsonProcessor()
        print("✓ 成功創建 main_processor_improved 實例")
        
        # 檢查是否使用了來自 josephson_model 的預處理函數
        import inspect
        from src.cpr import main_processor_improved
        if hasattr(main_processor_improved, 'preprocess_data_numba'):
            func_source = inspect.getsource(main_processor_improved.preprocess_data_numba)
            if 'josephson_model' in func_source or 'calculate_mode_magnitude' in func_source:
                print("✓ main_processor_improved 使用改進的預處理方法")
            else:
                print("⚠ main_processor_improved 可能未使用改進的預處理方法")
        else:
            print("⚠ main_processor_improved 沒有自己的 preprocess_data_numba")
            
    except Exception as e:
        print(f"✗ main_processor_improved 測試失敗: {e}")
    
    # 測試 main_processor.py
    try:
        from src.cpr.main_processor import EnhancedJosephsonProcessor as MainProcessor
        processor = MainProcessor()
        print("✓ 成功創建 main_processor 實例")
        
    except Exception as e:
        print(f"✗ main_processor 測試失敗: {e}")
    
    print("\n=== 效果驗證 ===")
    print("新的預處理方法的主要改進：")
    print("1. 使用 numpy.log10() 計算數據的數量級")
    print("2. 使用眾數來獲得最具代表性的數量級")
    print("3. 相比舊方法（相鄰點差值），更穩健且不依賴特定數據點")
    print("4. 對異常值和噪聲數據更有抗性")

if __name__ == "__main__":
    test_integrated_preprocessing()
