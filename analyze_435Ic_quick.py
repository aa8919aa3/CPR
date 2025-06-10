#!/usr/bin/env python3
"""
專門分析 435Ic.csv 的快速腳本
展示新的 log10 量級眾數預處理方法
"""
import sys
import os
from pathlib import Path

# 設置 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🔬 分析 435Ic.csv")
    print("=" * 40)
    
    try:
        from cpr.main_processor_improved import ImprovedJosephsonProcessor
        
        # 檔案設定
        filename = "435Ic.csv"
        file_path = f"data/Ic/{filename}"
        output_dir = "output/analysis_435Ic"
        
        # 檢查檔案
        if not os.path.exists(file_path):
            print(f"❌ 檔案不存在: {file_path}")
            return
        
        print(f"📁 檔案: {filename}")
        print(f"📂 輸出: {output_dir}")
        print()
        
        # 執行分析
        processor = ImprovedJosephsonProcessor()
        print("🚀 開始分析...")
        result = processor.process_single_file(file_path, output_dir)
        
        # 顯示結果
        if result['success']:
            print("✅ 分析成功!")
            print()
            print("📊 主要結果:")
            print(f"   頻率: {result['f']:.6e} Hz")
            print(f"   來源: {result['frequency_source']}")
            print(f"   R²: {result['r_squared']:.4f}")
            print(f"   透明度: {result['T']:.2%}")
            print()
            print("🖼️ 圖表已生成在輸出目錄中")
        else:
            print(f"❌ 失敗: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    main()
