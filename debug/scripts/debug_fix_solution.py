#!/usr/bin/env python3
"""
針對 NaN/inf 問題的具體修復方案
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_current_implementation():
    """測試當前實現對問題文件的處理結果"""
    from cpr.main_processor_optimized import EnhancedJosephsonProcessor
    
    problem_files = [
        "data/Ic/228Ic.csv",
        "data/Ic/130Ic-.csv"
    ]
    
    print("=== 當前實現測試結果 ===")
    processor = EnhancedJosephsonProcessor()
    output_dir = project_root / "debug" / "output" / "current_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in problem_files:
        csv_file = project_root / filename
        if csv_file.exists():
            print(f"\n測試文件: {filename}")
            result = processor.process_single_file(str(csv_file), str(output_dir))
            print(f"  結果: {'成功' if result['success'] else '失敗'}")
            if not result['success']:
                print(f"  錯誤: {result['error']}")

def analyze_unique_values_precision():
    """分析唯一值的精度問題"""
    print("\n=== 唯一值精度分析 ===")
    
    for filename in ["data/Ic/228Ic.csv", "data/Ic/130Ic-.csv"]:
        csv_file = project_root / filename
        if csv_file.exists():
            print(f"\n分析文件: {filename}")
            data = pd.read_csv(csv_file)
            current = data.iloc[:, 1].values
            
            print(f"原始電流值:")
            unique_current = np.unique(current)
            for i, val in enumerate(unique_current):
                print(f"  {i+1}: {val:.15e}")
            
            # 模擬預處理過程
            current_shifted = current - np.min(current)
            print(f"\n減去最小值後:")
            unique_shifted = np.unique(current_shifted)
            for i, val in enumerate(unique_shifted):
                print(f"  {i+1}: {val:.15e}")
            
            # 計算歸一化因子
            if len(current_shifted) > 2:
                y_factor = abs(current_shifted[2] - current_shifted[1])
            else:
                y_factor = 1.0
            
            if y_factor < 1e-12:
                y_std = np.std(current_shifted)
                if y_std > 1e-12:
                    y_factor = y_std
                else:
                    y_range = np.max(current_shifted) - np.min(current_shifted)
                    y_factor = y_range if y_range > 1e-12 else 1.0
            
            y_factor = max(y_factor, 1e-6)
            print(f"\n歸一化因子: {y_factor:.15e}")
            
            # 歸一化
            current_normalized = current_shifted / y_factor
            print(f"\n歸一化後:")
            unique_normalized = np.unique(current_normalized)
            for i, val in enumerate(unique_normalized):
                print(f"  {i+1}: {val:.15e}")
            
            print(f"唯一值數量: {len(unique_normalized)}")
            print(f"是否被認為是常數: {len(unique_normalized) < 2}")

def create_fixed_validation():
    """創建修復後的驗證函數"""
    print("\n=== 修復方案 ===")
    
    def validate_data_array_fixed(data, name, tolerance=1e-12):
        """修復後的數據驗證函數"""
        if not np.isfinite(data).all():
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            return False, f"{name}: {nan_count} NaN, {inf_count} inf values"
        
        # 使用更寬鬆的唯一值檢查
        unique_values = np.unique(data)
        
        # 如果只有一個唯一值，確實是常數
        if len(unique_values) == 1:
            return False, f"{name}: constant data (all values same)"
        
        # 如果有2個或更多唯一值，檢查它們之間的差異是否足夠大
        if len(unique_values) >= 2:
            min_diff = np.min(np.diff(np.sort(unique_values)))
            if min_diff < tolerance:
                # 值之間的差異太小，可能是數值誤差
                data_range = np.max(data) - np.min(data)
                relative_diff = min_diff / (data_range + 1e-15)  # 避免除零
                
                if relative_diff < 1e-10:  # 相對差異小於 1e-10
                    return False, f"{name}: quasi-constant data (min_diff={min_diff:.2e}, relative={relative_diff:.2e})"
        
        # 檢查數據的變異性
        data_std = np.std(data)
        data_mean = np.mean(np.abs(data))
        
        if data_std / (data_mean + 1e-15) < 1e-10:  # 變異係數過小
            return False, f"{name}: insufficient variation (CV={data_std/(data_mean+1e-15):.2e})"
            
        return True, "OK"
    
    # 測試修復後的函數
    print("測試修復後的驗證函數:")
    for filename in ["data/Ic/228Ic.csv", "data/Ic/130Ic-.csv"]:
        csv_file = project_root / filename
        if csv_file.exists():
            print(f"\n測試文件: {filename}")
            data = pd.read_csv(csv_file)
            current = data.iloc[:, 1].values
            
            # 模擬預處理
            current_shifted = current - np.min(current)
            if len(current_shifted) > 2:
                y_factor = abs(current_shifted[2] - current_shifted[1])
            else:
                y_factor = 1.0
            if y_factor < 1e-12:
                y_std = np.std(current_shifted)
                y_factor = y_std if y_std > 1e-12 else 1.0
            y_factor = max(y_factor, 1e-6)
            current_normalized = current_shifted / y_factor
            
            # 測試原始驗證函數
            unique_count = len(np.unique(current_normalized))
            original_result = "PASS" if unique_count >= 2 else "FAIL"
            
            # 測試修復後的驗證函數
            fixed_valid, fixed_msg = validate_data_array_fixed(current_normalized, "y_normalized")
            fixed_result = "PASS" if fixed_valid else "FAIL"
            
            print(f"  唯一值數量: {unique_count}")
            print(f"  原始驗證: {original_result}")
            print(f"  修復後驗證: {fixed_result}")
            if not fixed_valid:
                print(f"  修復後錯誤: {fixed_msg}")

def main():
    """主函數"""
    print("針對 NaN/inf 問題的具體修復方案")
    print("="*60)
    
    test_current_implementation()
    analyze_unique_values_precision()
    create_fixed_validation()
    
    print(f"\n{'='*60}")
    print("分析完成")

if __name__ == "__main__":
    main()
