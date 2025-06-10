#!/usr/bin/env python3
"""
驗證頻率修復的腳本
檢查所有圖表中的頻率是否一致
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cpr.main_processor_improved import ImprovedJosephsonProcessor

def main():
    print("=== 頻率修復最終驗證 ===")
    
    # 測試文件
    test_file = "/Users/albert-mac/Code/GitHub/CPR/data/Ic/435Ic.csv"
    output_dir = "/Users/albert-mac/Code/GitHub/CPR/output/frequency_fix_verification"
    
    if not os.path.exists(test_file):
        print(f"❌ 測試文件不存在: {test_file}")
        return
    
    # 創建處理器
    processor = ImprovedJosephsonProcessor()
    
    # 處理文件
    print(f"測試文件: {test_file}")
    print(f"輸出目錄: {output_dir}")
    
    result = processor.process_single_file(test_file, output_dir)
    
    if result['success']:
        print("\n=== ✅ 處理成功 ===")
        print(f"數據ID: {result['dataid']}")
        print(f"最終頻率: {result['f']:.6e} Hz")
        print(f"頻率來源: {result['frequency_source']}")
        print(f"頻率可靠性: {result['frequency_analysis']['frequency_reliable']}")
        
        # 顯示頻率分析詳情
        freq_analysis = result['frequency_analysis']
        print("\n=== 頻率分析詳情 ===")
        print(f"Lomb-Scargle 頻率: {freq_analysis.get('ls_frequency', 'N/A')}")
        print(f"擬合頻率 (歸一化): {freq_analysis.get('fit_frequency_normalized', 'N/A')}")
        print(f"擬合頻率 (縮放): {freq_analysis.get('fit_frequency_scaled', 'N/A')}")
        print(f"使用的最佳頻率: {freq_analysis['best_frequency']}")
        
        # 檢查輸出文件
        print("\n=== 輸出文件檢查 ===")
        expected_files = [
            f"{result['dataid']}_fitted_curve_normalized_plot.png",
            f"{result['dataid']}_fitted_curve_plot.png", 
            f"{result['dataid']}_phase_folded_with_drift.png",
            f"{result['dataid']}_cycles_colored_matplotlib.png",
            f"{result['dataid']}_residuals_plot.png"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"✅ {filename}")
            else:
                print(f"❌ {filename} (未找到)")
        
        print("\n=== 關鍵檢查點 ===")
        print("1. fitted_curve_plot 中的頻率應為原始單位的最佳頻率")
        print("2. fitted_curve_normalized_plot 中的頻率信息應包含來源")
        print("3. phase_folded_with_drift 中的頻率應與 fitted_curve_plot 一致")
        print("4. 所有圖表應使用相同的頻率進行週期計算")
        
        print(f"\n✅ 驗證完成！請檢查 {output_dir} 中的圖片")
        
    else:
        print(f"❌ 處理失敗: {result.get('error', '未知錯誤')}")

if __name__ == "__main__":
    main()

