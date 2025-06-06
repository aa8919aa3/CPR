#!/usr/bin/env python3
"""
測試跳過邏輯是否正常工作
"""
import sys
import os
import multiprocessing
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable for subprocess/threading
os.environ['PYTHONPATH'] = str(src_path) + os.pathsep + os.environ.get('PYTHONPATH', '')

# 設置多進程啟動方法為 'spawn' 以避免模組導入問題
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

def main():
    print("=== 測試跳過邏輯 ===")
    
    # Import here to ensure path is set
    try:
        from cpr.main_processor_optimized import EnhancedJosephsonProcessor
    except ImportError as e:
        print(f"導入錯誤: {e}")
        print("嘗試使用單線程模式...")
        return
    
    # 創建處理器實例
    processor = EnhancedJosephsonProcessor()
    
    # 測試這幾個特定文件
    test_files = ['data/Ic/228Ic.csv', 'data/Ic/130Ic-.csv', 'data/Ic/394Ic.csv']
    print(f"測試文件: {[Path(f).name for f in test_files]}")
    
    # 創建輸出目錄
    output_dir = 'output/skip_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理這些文件 - 先逐個處理避免多線程問題
    results = []
    for file_path in test_files:
        print(f"\n處理文件: {Path(file_path).name}")
        try:
            result = processor.process_single_file(file_path, output_dir)
            results.append(result)
        except Exception as e:
            print(f"處理 {Path(file_path).name} 時發生錯誤: {e}")
            results.append({
                'dataid': Path(file_path).stem,
                'success': False,
                'error': str(e)
            })
    
    print(f"\n=== 處理結果摘要 ===")
    for result in results:
        dataid = result['dataid']
        success = result['success']
        skipped = result.get('skipped', False)
        error = result.get('error', '')
        
        if skipped:
            print(f"⏭️  {dataid}: SKIPPED - {error}")
        elif success:
            I_c = result.get('I_c', 'N/A')
            r_squared = result.get('r_squared', 'N/A')
            print(f"✅ {dataid}: SUCCESS - I_c: {I_c:.3e}, R²: {r_squared:.4f}")
        else:
            print(f"❌ {dataid}: FAILED - {error}")
    
    # 統計結果
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    
    print(f"\n=== 統計結果 ===")
    print(f"總檔案數: {len(results)}")
    print(f"成功處理: {len(successful)}")
    print(f"處理失敗: {len(failed)}")
    print(f"數據質量跳過: {len(skipped)}")
    print(f"成功率: {len(successful)/len(results)*100:.1f}%")
    
    if skipped:
        print(f"\n🎯 跳過的文件 (數據質量不佳):")
        for result in skipped:
            print(f"  {result['dataid']}: {result.get('error', '')}")
    
    # 測試多線程批處理
    print(f"\n=== 測試多線程批處理 ===")
    try:
        batch_results = processor.process_files(test_files, output_dir)
        print(f"批處理結果: {len(batch_results)} 個文件")
        batch_successful = sum(1 for r in batch_results if r.get('success', False))
        batch_skipped = sum(1 for r in batch_results if r.get('skipped', False))
        print(f"批處理成功: {batch_successful}, 跳過: {batch_skipped}")
    except Exception as e:
        print(f"批處理失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
