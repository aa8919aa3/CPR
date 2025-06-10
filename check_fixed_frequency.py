#!/usr/bin/env python3
"""
快速檢查修復後的頻率顯示
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cpr.main_processor_improved import ImprovedJosephsonProcessor

def main():
    print("=== 檢查修復後的頻率顯示 ===")
    
    # 測試文件
    test_file = "/Users/albert-mac/Code/GitHub/CPR/data/Ic/435Ic.csv"
    output_dir = "/Users/albert-mac/Code/GitHub/CPR/output/frequency_fix_final"
    
    processor = ImprovedJosephsonProcessor()
    result = processor.process_single_file(test_file, output_dir)
    
    if result['success']:
        freq_analysis = result['frequency_analysis']
        print(f"✅ 處理成功")
        print(f"Lomb-Scargle 頻率 (歸一化): {freq_analysis.get('ls_frequency', 'N/A'):.6e}")
        print(f"最終頻率 (原始單位): {result['f']:.6e} Hz")
        print(f"頻率來源: {result['frequency_source']}")
        
        print("\n現在應該看到:")
        print(f"1. fitted_curve_normalized_plot: f = {freq_analysis.get('ls_frequency', 0):.2e} (norm.)")
        print(f"2. fitted_curve_plot: f = {result['f']:.2e} Hz")
        print(f"3. phase_folded_with_drift: 頻率 = {result['f']:.6e} Hz")
        
    else:
        print(f"❌ 處理失敗: {result.get('error', '未知錯誤')}")

if __name__ == "__main__":
    main()
