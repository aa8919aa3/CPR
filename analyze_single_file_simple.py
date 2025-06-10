#!/usr/bin/env python3
"""
簡單的單檔案 CPR 分析腳本
使用改進的 log10 量級眾數預處理方法
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 設置 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def analyze_single_file(filename):
    """
    分析單一檔案的簡化版本
    
    Parameters:
    -----------
    filename : str
        要分析的CSV檔案名稱 (例如: "435Ic.csv")
    """
    
    try:
        from cpr.main_processor_improved import ImprovedJosephsonProcessor
        
        # 檔案路徑設定
        file_path = f"data/Ic/{filename}"
        dataid = Path(filename).stem
        output_dir = f"output/simple_analysis_{dataid}"
        
        # 檢查檔案是否存在
        if not os.path.exists(file_path):
            print(f"❌ 檔案不存在: {file_path}")
            print("💡 請確認檔案名稱正確，包括 .csv 副檔名")
            return
        
        print("🔬 CPR 單檔案分析")
        print("=" * 50)
        print(f"📁 分析檔案: {filename}")
        print(f"📂 輸出目錄: {output_dir}")
        print()
        
        # 創建處理器並執行分析
        processor = ImprovedJosephsonProcessor()
        result = processor.process_single_file(file_path, output_dir)
        
        # 顯示分析結果
        print("📊 分析結果:")
        print("-" * 30)
        
        if result['success']:
            print("✅ 狀態: 分析成功")
            print(f"🆔 檔案ID: {result['dataid']}")
            print(f"📈 頻率: {result['f']:.6e} Hz")
            print(f"🔍 頻率來源: {result['frequency_source']}")
            print(f"📊 頻率可靠性: {'是' if result['frequency_reliable'] else '否'}")
            print(f"⚡ 臨界電流 (I_c): {result['I_c']:.4e} A")
            print(f"🌊 相位偏移 (φ₀): {result['phi_0']:.4f} rad")
            print(f"🔲 透明度 (T): {result['T']:.2%}")
            print(f"📏 線性項 (r): {result['r']:.4e}")
            print(f"📍 常數項 (C): {result['C']:.4e}")
            print(f"📈 R²: {result['r_squared']:.4f}")
            print(f"📊 調整 R²: {result['adj_r_squared']:.4f}")
            print(f"🎯 RMSE: {result['rmse']:.4e}")
            print(f"📐 MAE: {result['mae']:.4e}")
            
            print()
            print("🖼️ 生成的圖表:")
            print(f"   • {output_dir}/{dataid}_fitted_curve_normalized_plot.png")
            print(f"   • {output_dir}/{dataid}_fitted_curve_plot.png")
            print(f"   • {output_dir}/{dataid}_residuals_plot.png")
            print(f"   • {output_dir}/{dataid}_phase_folded_with_drift.png")
            print(f"   • {output_dir}/{dataid}_cycles_colored_matplotlib.png")
            
            print()
            print("🎉 分析完成！請查看輸出目錄中的圖表檔案。")
            
        else:
            print("❌ 狀態: 分析失敗")
            print(f"🚫 錯誤: {result.get('error', '未知錯誤')}")
            
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("💡 請確認您在正確的專案目錄中，且所有依賴已安裝")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

def interactive_mode():
    """互動模式：讓用戶選擇要分析的檔案"""
    print("🔬 CPR 互動式單檔案分析")
    print("=" * 50)
    
    # 顯示可用檔案的範例
    try:
        import glob
        files = glob.glob("data/Ic/*.csv")
        if files:
            files.sort()
            print(f"📁 找到 {len(files)} 個CSV檔案")
            print("\n前10個檔案範例:")
            for i, file_path in enumerate(files[:10], 1):
                filename = Path(file_path).name
                print(f"  {i:2d}. {filename}")
            
            if len(files) > 10:
                print(f"  ... 還有 {len(files)-10} 個檔案")
        else:
            print("❌ 在 data/Ic/ 目錄中未找到CSV檔案")
            return
            
    except Exception as e:
        print(f"⚠️ 無法列出檔案: {e}")
    
    print()
    filename = input("請輸入要分析的檔案名稱 (例如: 435Ic.csv): ").strip()
    
    if filename:
        analyze_single_file(filename)
    else:
        print("❌ 未輸入檔案名稱")

def main():
    """主程式入口"""
    
    # 檢查命令列參數
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        analyze_single_file(filename)
    else:
        # 如果沒有提供檔案名稱，進入互動模式
        interactive_mode()

if __name__ == "__main__":
    main()
