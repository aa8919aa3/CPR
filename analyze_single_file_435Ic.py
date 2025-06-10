#!/usr/bin/env python3
"""
單一檔案分析腳本 - 分析 435Ic.csv
使用現有的CPR分析系統分析單一檔案
"""

import os
import sys
from pathlib import Path

# 添加 src 目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor

def analyze_single_file():
    """分析單一檔案 435Ic.csv"""
    
    # 初始化處理器
    print("=" * 60)
    print("CPR 單一檔案分析")
    print("=" * 60)
    print("正在初始化分析器...")
    
    processor = EnhancedJosephsonProcessor()
    
    # 檔案路徑
    file_path = "data/Ic/435Ic.csv"
    output_dir = "output/single_file_analysis_435Ic"
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"❌ 錯誤：檔案 {file_path} 不存在")
        return
    
    print(f"✓ 找到檔案：{file_path}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 輸出目錄：{output_dir}")
    
    print("\n" + "=" * 60)
    print("開始分析...")
    print("=" * 60)
    
    # 執行分析
    try:
        result = processor.process_single_file(file_path, output_dir)
        
        # 顯示結果
        print("\n" + "=" * 60)
        print("分析結果")
        print("=" * 60)
        
        if result.get('success', False):
            print("✅ 分析成功完成！")
            print(f"\n檔案 ID：{result['dataid']}")
            print(f"\n擬合參數：")
            print(f"  • 臨界電流 (I_c)：{result['I_c']:.4e} A")
            print(f"  • 相位偏移 (φ₀)：{result['phi_0']:.4f} rad")
            print(f"  • 頻率 (f)：{result['f']:.4e} Hz")
            print(f"  • 透明度 (T)：{result['T']:.4f} ({result['T']*100:.2f}%)")
            print(f"  • 線性項 (r)：{result['r']:.4e}")
            print(f"  • 常數項 (C)：{result['C']:.4e}")
            
            print(f"\n統計指標：")
            print(f"  • R²：{result['r_squared']:.6f}")
            print(f"  • 調整 R²：{result['adj_r_squared']:.6f}")
            print(f"  • RMSE：{result['rmse']:.6f}")
            print(f"  • MAE：{result['mae']:.6f}")
            print(f"  • 殘差平均：{result['residual_mean']:.6e}")
            print(f"  • 殘差標準差：{result['residual_std']:.6e}")
            
        else:
            print("❌ 分析失敗")
            print(f"錯誤信息：{result.get('error', '未知錯誤')}")
            
    except Exception as e:
        print(f"❌ 分析過程中發生錯誤：{str(e)}")
        return
    
    # 顯示生成的檔案
    print(f"\n" + "=" * 60)
    print("生成的檔案")
    print("=" * 60)
    
    if result.get('success', False):
        output_files = [
            f"{result['dataid']}_fitted_curve_normalized_plot.png",
            f"{result['dataid']}_fitted_curve_plot.png", 
            f"{result['dataid']}_residuals_plot.png",
            f"{result['dataid']}_phase_folded_with_drift.png",
            f"{result['dataid']}_cycles_colored_matplotlib.png"
        ]
        
        print("分析生成了以下視覺化檔案：")
        for i, filename in enumerate(output_files, 1):
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"  {i}. ✓ {filename}")
            else:
                print(f"  {i}. ✗ {filename} (未生成)")
                
        print(f"\n所有檔案保存在：{os.path.abspath(output_dir)}")
        
        print(f"\n圖表說明：")
        print(f"  1. fitted_curve_normalized_plot.png - 標準化數據與擬合曲線")
        print(f"  2. fitted_curve_plot.png - 原始尺度數據與擬合曲線")
        print(f"  3. residuals_plot.png - 殘差分析（4個子圖）")
        print(f"  4. phase_folded_with_drift.png - 相位摺疊圖與漂移分析")
        print(f"  5. cycles_colored_matplotlib.png - 按週期著色的原始數據")
    
    print(f"\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    analyze_single_file()
