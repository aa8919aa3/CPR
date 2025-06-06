#!/usr/bin/env python3
"""
CPR - 示例使用腳本
演示如何使用 Josephson Junction Analysis Suite 進行數據分析
"""

import sys
import os
from pathlib import Path

# 添加 src 到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cpr import EnhancedJosephsonProcessor, config


def main():
    """示例主函數"""
    print("=" * 60)
    print("CPR - Current-Phase Relation")
    print("Josephson Junction Analysis Suite")
    print("=" * 60)
    
    # 創建處理器實例
    processor = EnhancedJosephsonProcessor()
    
    # 顯示配置信息
    print(f"\n📁 輸入資料夾: {config.get('INPUT_FOLDER')}")
    print(f"📁 輸出資料夾: {config.get('OUTPUT_FOLDER')}")
    print(f"🔧 工作線程數: {config.get('N_WORKERS', '自動檢測')}")
    
    # 檢查數據文件
    input_folder = Path(config.get('INPUT_FOLDER'))
    if not input_folder.exists():
        print(f"\n❌ 錯誤: 輸入資料夾 '{input_folder}' 不存在")
        return
    
    csv_files = list(input_folder.glob("*.csv"))
    print(f"\n📊 找到 {len(csv_files)} 個 CSV 文件")
    
    if len(csv_files) == 0:
        print("❌ 未找到 CSV 數據文件")
        return
    
    # 處理第一個文件作為示例
    sample_file = csv_files[0]
    print(f"\n🔬 處理示例文件: {sample_file.name}")
    
    try:
        result = processor.process_single_file(str(sample_file))
        
        if result['status'] == 'success':
            print("✅ 處理成功!")
            print(f"   數據點數: {result['n_points']}")
            print(f"   處理時間: {result['processing_time']:.3f} 秒")
            print(f"   X 範圍: {result['x_range']}")
            print(f"   Y 範圍: {result['y_range']}")
        else:
            print(f"❌ 處理失敗: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
    
    print(f"\n✨ 分析完成! 結果保存在 '{config.get('OUTPUT_FOLDER')}' 資料夾中")
    print("\n💡 提示: 使用 'python run_analysis.py' 處理所有文件")


if __name__ == "__main__":
    main()
