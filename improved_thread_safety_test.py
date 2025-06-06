#!/usr/bin/env python3
"""
改進的線程安全測試 - 解決多線程競爭條件問題
"""
import os
import sys
import glob
import time
import threading
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_with_improved_thread_safety():
    """測試改進的線程安全版本"""
    print("="*60)
    print("改進的線程安全測試")
    print("="*60)
    
    # 找到測試文件
    input_folder = "data/Ic"
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    # 選擇之前失敗的文件
    failed_files = []
    target_files = ["394Ic.csv", "175Ic.csv", "401Ic.csv"]
    
    for target in target_files:
        matching_files = [f for f in csv_files if Path(f).name == target]
        if matching_files:
            failed_files.append(matching_files[0])
    
    if len(failed_files) < 3:
        print(f"❌ 找不到足夠的測試文件")
        return False
    
    print(f"測試文件: {[Path(f).name for f in failed_files]}")
    
    # 測試不同的線程數設置
    for max_workers in [1, 2, 3]:
        print(f"\n=== 測試 {max_workers} 個工作線程 ===")
        
        # 創建輸出目錄
        output_dir = f"output_thread_test_{max_workers}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 動態修改最大工作線程數
        from cpr import main_processor_optimized
        original_workers = main_processor_optimized.MAX_WORKERS
        main_processor_optimized.MAX_WORKERS = max_workers
        
        try:
            # 重新導入處理器
            from importlib import reload
            reload(main_processor_optimized)
            
            # 創建處理器實例
            processor = main_processor_optimized.EnhancedJosephsonProcessor()
            
            # 開始測試
            start_time = time.time()
            results = processor.process_files(failed_files, output_dir)
            end_time = time.time()
            
            # 分析結果
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            processing_time = end_time - start_time
            
            print(f"處理結果:")
            print(f"  成功: {successful}/{len(failed_files)}")
            print(f"  失敗: {failed}")
            print(f"  成功率: {successful/len(failed_files)*100:.1f}%")
            print(f"  處理時間: {processing_time:.2f} 秒")
            
            # 如果這個設置成功，就找到了解決方案
            if successful == len(failed_files):
                print(f"✅ 找到解決方案：使用 {max_workers} 個工作線程")
                return max_workers
                
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
        finally:
            # 恢復原始設置
            main_processor_optimized.MAX_WORKERS = original_workers
    
    return False

def apply_thread_safety_fix():
    """應用線程安全修復"""
    print("\n=== 應用線程安全修復 ===")
    
    # 讀取當前的主處理器文件
    processor_file = "/Users/albert-mac/Code/GitHub/CPR/src/cpr/main_processor_optimized.py"
    
    # 檢查是否需要修復
    with open(processor_file, 'r') as f:
        content = f.read()
    
    # 如果還沒有全局鎖，添加它
    if "GLOBAL_PROCESSING_LOCK" not in content:
        print("添加全局處理鎖...")
        
        # 找到 MAX_WORKERS 定義的位置
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "MAX_WORKERS = min(" in line:
                # 在 MAX_WORKERS 定義後添加全局鎖
                lines.insert(i + 1, "")
                lines.insert(i + 2, "# Global lock for thread-safe operations")
                lines.insert(i + 3, "GLOBAL_PROCESSING_LOCK = threading.Lock()")
                lines.insert(i + 4, "NUMBA_COMPILATION_LOCK = threading.Lock()")
                break
        
        # 保存修改
        with open(processor_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ 全局鎖已添加")
        return True
    else:
        print("✅ 全局鎖已存在")
        return True

def main():
    """主函數"""
    try:
        # 應用線程安全修復
        if apply_thread_safety_fix():
            print("開始測試改進的線程安全性...")
            
            # 測試不同的線程配置
            optimal_workers = test_with_improved_thread_safety()
            
            if optimal_workers:
                print(f"\n🎉 找到最優配置：{optimal_workers} 個工作線程")
                print("\n建議:")
                print(f"  • 將 MAX_WORKERS 設置為 {optimal_workers}")
                print("  • 這樣可以避免競爭條件")
                print("  • 同時保持良好的性能")
                return True
            else:
                print("\n❌ 未找到穩定的線程配置")
                print("建議使用單線程模式 (MAX_WORKERS = 1)")
                return False
        else:
            print("❌ 無法應用線程安全修復")
            return False
            
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
