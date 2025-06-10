#!/usr/bin/env python3
"""
測試改善版本的處理器 - 驗證頻率計算修復
"""

import sys
import os
from pathlib import Path

# 添加路徑
sys.path.append('/Users/albert-mac/Code/GitHub/CPR/src')

from cpr.main_processor_improved import ImprovedJosephsonProcessor

def test_improved_processor():
    """測試改善版本的處理器"""
    
    # 測試文件路徑
    csv_file = '/Users/albert-mac/Code/GitHub/CPR/data/Ic/435Ic.csv'
    output_dir = '/Users/albert-mac/Code/GitHub/CPR/output/improved_test_435Ic'
    
    print("=== 測試改善版本的處理器 ===")
    print(f"輸入文件: {csv_file}")
    print(f"輸出目錄: {output_dir}")
    
    # 創建處理器
    processor = ImprovedJosephsonProcessor()
    
    # 處理文件
    result = processor.process_single_file(csv_file, output_dir)
    
    print("\n=== 分析結果 ===")
    if result['success']:
        print(f"✓ 分析成功")
        print(f"  頻率: {result['f']:.6e} Hz")
        print(f"  頻率來源: {result.get('frequency_source', 'unknown')}")
        print(f"  頻率可靠性: {result.get('frequency_reliable', 'unknown')}")
        
        if 'frequency_analysis' in result:
            freq_analysis = result['frequency_analysis']
            print(f"\n=== 頻率分析詳情 ===")
            print(f"  Lomb-Scargle 頻率: {freq_analysis.get('ls_frequency', 'N/A')}")
            print(f"  擬合頻率 (歸一化): {freq_analysis.get('fit_frequency_normalized', 'N/A')}")
            print(f"  擬合頻率 (縮放): {freq_analysis.get('fit_frequency_scaled', 'N/A')}")
            print(f"  最佳頻率: {freq_analysis.get('best_frequency', 'N/A')}")
            
        print(f"\n=== 其他參數 ===")
        print(f"  I_c: {result['I_c']:.2e}")
        print(f"  phi_0: {result['phi_0']:.2f}")
        print(f"  T: {result['T']:.2%}")
        print(f"  R²: {result['r_squared']:.4f}")
        
        print(f"\n=== 輸出文件 ===")
        for file in os.listdir(output_dir):
            if file.endswith('.png'):
                print(f"  {file}")
                
    else:
        print(f"✗ 分析失敗: {result.get('error', 'unknown error')}")
    
    return result

if __name__ == "__main__":
    result = test_improved_processor()
