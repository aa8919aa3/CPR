#!/usr/bin/env python3
"""
單一檔案 CPR 分析腳本
"""
import sys
import os
from pathlib import Path

# 設置 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from cpr.main_processor_improved import ImprovedJosephsonProcessor

def analyze_single_file(filename):
    """分析單一檔案"""
    # 文件路徑
    file_path = f"data/Ic/{filename}"
    output_dir = f"output/single_file_analysis_{Path(filename).stem}"
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在: {file_path}")
        return
    
    print(f"🔬 分析檔案: {filename}")
    print(f"📂 輸出目錄: {output_dir}")
    
    # 創建處理器
    processor = ImprovedJosephsonProcessor()
    
    # 執行分析
    result = processor.process_single_file(file_path, output_dir)
    
    # 顯示結果
    if result['success']:
        print("✅ 分析成功!")
        print(f"   頻率: {result['f']:.6e} Hz")
        print(f"   頻率來源: {result['frequency_source']}")
        print(f"   R²: {result['r_squared']:.4f}")
        print(f"   透明度: {result['T']:.2%}")
    else:
        print(f"❌ 分析失敗: {result.get('error')}")

if __name__ == "__main__":
    # 您可以修改這裡的檔名
    analyze_single_file("435Ic.csv")