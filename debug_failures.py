#!/usr/bin/env python3
"""
調試失敗文件的具體錯誤原因
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor

def debug_single_file(csv_file):
    """調試單個文件的處理過程"""
    print(f"\n=== 調試文件: {Path(csv_file).name} ===")
    
    processor = EnhancedJosephsonProcessor()
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        result = processor.process_single_file(csv_file, output_dir)
        
        if result['success']:
            print(f"✅ 成功處理")
            print(f"  I_c: {result.get('I_c', 'N/A')}")
            print(f"  R²: {result.get('r_squared', 'N/A')}")
        else:
            print(f"❌ 處理失敗")
            print(f"  錯誤: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        print(f"❌ 異常錯誤: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    # 測試失敗的文件
    failed_files = [
        "data/Ic/394Ic.csv",
        "data/Ic/175Ic.csv", 
        "data/Ic/401Ic.csv"
    ]
    
    print("調試失敗文件的錯誤原因")
    print("="*50)
    
    for csv_file in failed_files:
        if os.path.exists(csv_file):
            debug_single_file(csv_file)
        else:
            print(f"❌ 文件不存在: {csv_file}")

if __name__ == "__main__":
    main()
