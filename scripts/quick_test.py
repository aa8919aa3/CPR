#!/usr/bin/env python3
"""
快速測試分析腳本，用於驗證功能
"""
import sys
import os
from pathlib import Path

# 設置Python路徑
project_root = Path(__file__).parent.parent.absolute()
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def main():
    """快速測試"""
    print("🧪 快速測試分析腳本")
    print("=" * 50)
    
    # 測試模組導入
    try:
        from cpr.main_processor_optimized import EnhancedJosephsonProcessor
        print("✓ 模組導入成功")
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        return 1
    
    # 檢查數據目錄
    data_dir = project_root / 'data' / 'Ic'
    csv_files = list(data_dir.glob('*.csv'))
    print(f"✓ 找到 {len(csv_files)} 個CSV文件")
    
    if len(csv_files) == 0:
        print("❌ 沒有找到CSV文件")
        return 1
    
    # 測試樣本分析（前5個文件）
    test_files = csv_files[:5]
    print(f"🔬 測試處理前 {len(test_files)} 個文件:")
    for f in test_files:
        print(f"   - {f.name}")
    
    # 運行測試
    from scripts.analyze_all_csv import main as analyze_main
    import tempfile
    
    # 創建臨時輸出目錄
    with tempfile.TemporaryDirectory() as temp_dir:
        # 模擬命令行參數
        sys.argv = [
            'analyze_all_csv.py',
            '--sample-size', '5',
            '--output-dir', temp_dir,
            '--max-workers', '2'
        ]
        
        try:
            result = analyze_main()
            if result == 0:
                print("✅ 測試成功!")
            else:
                print("❌ 測試失敗")
                return 1
        except Exception as e:
            print(f"❌ 測試異常: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n🎉 快速測試完成，分析腳本功能正常!")
    return 0

if __name__ == "__main__":
    exit(main())
