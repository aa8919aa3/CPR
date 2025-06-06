#!/usr/bin/env python3
"""
修復版本：測試跳過邏輯是否正常工作
"""
import sys
import os
from pathlib import Path

def setup_python_path():
    """設置Python路徑以確保模組能被正確導入"""
    # Add the src directory to the Python path
    project_root = Path(__file__).parent.absolute()
    src_path = project_root / 'src'
    
    # Add to sys.path if not already there
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set PYTHONPATH environment variable for subprocesses
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(src_path) not in current_pythonpath:
        os.environ['PYTHONPATH'] = str(src_path) + os.pathsep + current_pythonpath
    
    print(f"✓ Python路徑設置完成: {src_path}")
    return src_path

def test_module_import():
    """測試模組導入是否正常"""
    try:
        from cpr.main_processor_optimized import EnhancedJosephsonProcessor
        print("✓ 成功導入 EnhancedJosephsonProcessor")
        assert True, "模組導入成功"
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        assert False, f"模組導入失敗: {e}"

def main():
    print("=" * 60)
    print("測試跳過邏輯 - 修復版本")
    print("=" * 60)
    
    # 設置路徑
    setup_python_path()
    
    # 測試模組導入
    if not test_module_import():
        print("❌ 模組導入失敗，退出測試")
        return
    
    # 現在導入模組
    from cpr.main_processor_optimized import EnhancedJosephsonProcessor
    
    # 創建處理器實例
    print("🔧 創建處理器實例...")
    processor = EnhancedJosephsonProcessor()
    
    # 測試這幾個特定文件
    test_files = [
        'data/Ic/228Ic.csv',    # 應該被跳過（數據質量不佳）
        'data/Ic/130Ic-.csv',   # 應該被跳過（數據質量不佳）
        'data/Ic/394Ic.csv',    # 應該成功處理
        'data/Ic/175Ic.csv',    # 應該成功處理
        'data/Ic/401Ic.csv'     # 應該成功處理
    ]
    
    # 檢查文件是否存在
    existing_files = []
    for file_path in test_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    if not existing_files:
        print("❌ 沒有找到測試文件")
        return
    
    print(f"📁 找到 {len(existing_files)} 個測試文件")
    for f in existing_files:
        print(f"   - {Path(f).name}")
    
    # 創建輸出目錄
    output_dir = 'output/skip_test_fixed'
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 輸出目錄: {output_dir}")
    
    print("\n" + "=" * 60)
    print("開始單個文件測試")
    print("=" * 60)
    
    # 先逐個處理文件（單線程，避免多線程問題）
    results = []
    for file_path in existing_files:
        filename = Path(file_path).name
        print(f"\n🔄 處理文件: {filename}")
        try:
            result = processor.process_single_file(file_path, output_dir)
            results.append(result)
            
            # 分析結果
            if result.get('skipped', False):
                print(f"   ⏭️  跳過: {result.get('error', '未知原因')}")
            elif result.get('success', False):
                I_c = result.get('I_c', 'N/A')
                r_squared = result.get('r_squared', 'N/A')
                print(f"   ✅ 成功: I_c={I_c:.3e}, R²={r_squared:.4f}")
            else:
                print(f"   ❌ 失敗: {result.get('error', '未知錯誤')}")
                
        except Exception as e:
            print(f"   💥 異常: {e}")
            results.append({
                'dataid': Path(file_path).stem,
                'success': False,
                'error': str(e)
            })
    
    print("\n" + "=" * 60)
    print("處理結果摘要")
    print("=" * 60)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False) and not r.get('skipped', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    
    print(f"總檔案數: {len(results)}")
    print(f"成功處理: {len(successful)}")
    print(f"處理失敗: {len(failed)}")
    print(f"數據質量跳過: {len(skipped)}")
    if len(results) > 0:
        print(f"成功率: {len(successful)/len(results)*100:.1f}%")
    
    if skipped:
        print(f"\n🎯 跳過的文件（數據質量不佳）:")
        for result in skipped:
            print(f"   {result['dataid']}: {result.get('error', '')}")
    
    if failed:
        print(f"\n❌ 失敗的文件:")
        for result in failed:
            print(f"   {result['dataid']}: {result.get('error', '')}")
    
    if successful:
        print(f"\n✅ 成功處理的文件:")
        for result in successful:
            I_c = result.get('I_c', 'N/A')
            r_squared = result.get('r_squared', 'N/A')
            print(f"   {result['dataid']}: I_c={I_c:.3e}, R²={r_squared:.4f}")
    
    print("\n" + "=" * 60)
    print("測試批處理功能")
    print("=" * 60)
    
    # 測試批處理（使用ThreadPoolExecutor）
    try:
        print("🔄 開始批處理測試...")
        batch_results = processor.process_files(existing_files, output_dir)
        
        batch_successful = sum(1 for r in batch_results if r.get('success', False))
        batch_skipped = sum(1 for r in batch_results if r.get('skipped', False))
        batch_failed = len(batch_results) - batch_successful - batch_skipped
        
        print(f"✅ 批處理完成:")
        print(f"   - 總數: {len(batch_results)}")
        print(f"   - 成功: {batch_successful}")
        print(f"   - 跳過: {batch_skipped}")
        print(f"   - 失敗: {batch_failed}")
        
    except Exception as e:
        print(f"❌ 批處理失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
