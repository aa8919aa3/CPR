#!/usr/bin/env python3
"""
完整集成測試 - 驗證新的數量級眾數預處理方法在整個系統中的集成
"""

import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor
from cpr.josephson_model import preprocess_data_numba, preprocess_data_fallback

def create_test_csv(filename, x_data, y_data):
    """創建測試用的 CSV 檔案"""
    df = pd.DataFrame({
        'y_field': x_data,  # External magnetic flux
        'Ic': y_data        # Supercurrent
    })
    df.to_csv(filename, index=False)
    return filename

def test_complete_integration():
    """測試完整的系統集成"""
    print("=" * 60)
    print("完整集成測試 - 數量級眾數預處理方法")
    print("=" * 60)
    
    # 創建具有不同數量級的測試數據
    np.random.seed(42)
    n_points = 100
    
    # 測試案例 1: 正常數據
    x_data1 = np.linspace(0, 10, n_points)
    y_data1 = 2.5 * np.sin(2 * np.pi * 0.8 * x_data1) + 0.1 * x_data1 + 1.2 + 0.1 * np.random.normal(0, 1, n_points)
    
    # 測試案例 2: 大數量級差異的數據
    x_data2 = np.linspace(0, 1e-3, n_points)  # 微小範圍
    y_data2 = 1e6 * np.sin(2 * np.pi * 500 * x_data2) + 1e5 + 1e4 * np.random.normal(0, 1, n_points)  # 大數值
    
    # 測試案例 3: 混合數量級數據
    x_data3 = np.concatenate([
        np.linspace(0, 1e-6, n_points//3),
        np.linspace(1e-3, 1e-2, n_points//3),
        np.linspace(1, 10, n_points//3)
    ])
    y_data3 = np.concatenate([
        1e9 * np.sin(2 * np.pi * 1e3 * x_data3[:n_points//3]) + 1e8,
        1e3 * np.sin(2 * np.pi * 10 * x_data3[n_points//3:2*n_points//3]) + 1e2,
        10 * np.sin(2 * np.pi * 0.5 * x_data3[2*n_points//3:]) + 5
    ])
    
    test_cases = [
        ("normal_data", x_data1, y_data1),
        ("large_scale_diff", x_data2, y_data2),
        ("mixed_scales", x_data3, y_data3)
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"使用臨時目錄: {temp_dir}")
        
        # 創建測試檔案
        csv_files = []
        for name, x_data, y_data in test_cases:
            csv_file = create_test_csv(
                os.path.join(temp_dir, f"{name}.csv"),
                x_data, y_data
            )
            csv_files.append(csv_file)
            print(f"✓ 創建測試檔案: {name}.csv")
        
        # 創建輸出目錄
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 40)
        print("測試預處理方法比較")
        print("=" * 40)
        
        # 直接測試預處理方法
        for name, x_data, y_data in test_cases:
            print(f"\n測試案例: {name}")
            print("-" * 30)
            
            # 使用新的數量級眾數方法
            try:
                x_norm_new, y_norm_new, x_factor_new, y_factor_new = preprocess_data_numba(x_data, y_data)
                print(f"新方法 - x_factor: {x_factor_new:.2e}, y_factor: {y_factor_new:.2e}")
                print(f"新方法 - x歸一化範圍: [{np.min(x_norm_new):.2e}, {np.max(x_norm_new):.2e}]")
                print(f"新方法 - y歸一化範圍: [{np.min(y_norm_new):.2e}, {np.max(y_norm_new):.2e}]")
            except Exception as e:
                print(f"新方法失敗: {e}")
                continue
            
            # 使用回退方法比較
            try:
                x_norm_old, y_norm_old, x_factor_old, y_factor_old = preprocess_data_fallback(x_data, y_data)
                print(f"舊方法 - x_factor: {x_factor_old:.2e}, y_factor: {y_factor_old:.2e}")
                print(f"舊方法 - x歸一化範圍: [{np.min(x_norm_old):.2e}, {np.max(x_norm_old):.2e}]")
                print(f"舊方法 - y歸一化範圍: [{np.min(y_norm_old):.2e}, {np.max(y_norm_old):.2e}]")
                
                # 計算改進比率
                x_improvement = x_factor_new / x_factor_old if x_factor_old != 0 else float('inf')
                y_improvement = y_factor_new / y_factor_old if y_factor_old != 0 else float('inf')
                print(f"改進比率 - x方向: {x_improvement:.2f}, y方向: {y_improvement:.2f}")
                
            except Exception as e:
                print(f"舊方法失敗: {e}")
        
        print("\n" + "=" * 40)
        print("測試完整系統集成")
        print("=" * 40)
        
        # 初始化處理器
        processor = EnhancedJosephsonProcessor()
        
        # 處理所有測試檔案
        results = processor.process_files(csv_files, output_dir)
        
        print(f"\n處理結果:")
        print(f"總計檔案: {len(results)}")
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        print(f"成功: {successful}")
        print(f"失敗: {failed}")
        
        # 檢查輸出檔案
        print(f"\n生成的檔案:")
        for result in results:
            if result['success']:
                dataid = result['dataid']
                expected_files = [
                    f"{dataid}_fitted_curve_normalized_plot.png",
                    f"{dataid}_fitted_curve_plot.png",
                    f"{dataid}_residuals_plot.png",
                    f"{dataid}_phase_folded_with_drift.png",
                    f"{dataid}_cycles_colored_matplotlib.png"
                ]
                
                print(f"\n{dataid}:")
                for filename in expected_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        size = os.path.getsize(filepath)
                        print(f"  ✓ {filename} ({size} bytes)")
                    else:
                        print(f"  ✗ {filename} (missing)")
                
                # 顯示擬合參數
                params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
                param_str = ", ".join([f"{p}: {result.get(p, 'N/A'):.2e}" if isinstance(result.get(p), (int, float)) 
                                     else f"{p}: {result.get(p, 'N/A')}" for p in params])
                print(f"  參數: {param_str}")
                print(f"  R²: {result.get('r_squared', 'N/A'):.4f}")
            else:
                print(f"\n{result['dataid']}: 失敗 - {result.get('error', 'Unknown error')}")
        
        # 檢查分析摘要檔案
        summary_file = os.path.join(output_dir, 'analysis_summary.csv')
        if os.path.exists(summary_file):
            print(f"\n✓ 分析摘要檔案已創建: analysis_summary.csv")
            summary_df = pd.read_csv(summary_file)
            print(f"摘要包含 {len(summary_df)} 條記錄")
        else:
            print(f"\n✗ 分析摘要檔案未找到")
        
        print("\n" + "=" * 60)
        print("集成測試完成!")
        print("=" * 60)
        
        return results

if __name__ == "__main__":
    results = test_complete_integration()
    
    # 簡單的成功/失敗統計
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n最終結果: {successful}/{total} 檔案處理成功")
    
    if successful == total:
        print("🎉 所有測試通過！新的數量級眾數預處理方法已成功集成到整個系統中。")
        sys.exit(0)
    else:
        print("⚠️ 部分測試失敗，需要進一步調查。")
        sys.exit(1)
