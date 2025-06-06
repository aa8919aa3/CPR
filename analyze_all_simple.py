#!/usr/bin/env python3
"""
簡化版本的所有CSV分析腳本 - 避免Numba兼容性問題
直接使用處理器的批量處理功能
"""
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """主函數"""
    print("🚀 CPR 簡化版批量分析")
    print("=" * 60)
    
    try:
        # 直接設置 HAS_NUMBA = False 來避免 Numba 問題
        import src.cpr.main_processor_optimized as mpo
        mpo.HAS_NUMBA = False
        
        from src.cpr.main_processor_optimized import EnhancedJosephsonProcessor
        print("✓ 成功導入處理器 (無Numba模式)")
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        return 1
    
    # 檢查輸入檔案
    input_dir = project_root / "data" / "Ic"
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ 在 {input_dir} 中未找到CSV檔案")
        return 1
    
    print(f"📁 找到 {len(csv_files)} 個CSV檔案")
    
    # 創建處理器
    processor = EnhancedJosephsonProcessor()
    
    # 開始處理
    print(f"\n🚀 開始批量處理...")
    start_time = time.time()
    
    try:
        # 使用內建的批量處理方法
        processor.batch_process_files()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ 處理完成！")
        print(f"⏱️ 總時間: {total_time:.2f} 秒")
        print(f"📈 平均速度: {total_time/len(csv_files):.3f} 秒/檔案")
        
        # 檢查輸出
        output_dir = "output"
        if os.path.exists(output_dir):
            png_files = list(Path(output_dir).glob("*.png"))
            print(f"📊 生成了 {len(png_files)} 個圖表檔案")
            
        return 0
        
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
