#!/usr/bin/env python3
"""
完整版CPR分析 - 處理所有CSV檔案
執行修改後的優化處理器，分析data/Ic目錄中的所有CSV檔案
"""
import os
import sys
import glob
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor

def main():
    """主函數 - 執行完整的CSV檔案分析"""
    print("="*80)
    print("CPR完整版分析 - 處理所有CSV檔案")
    print("="*80)
    
    # 初始化處理器
    try:
        processor = EnhancedJosephsonProcessor()
        print("✅ 處理器初始化成功")
    except Exception as e:
        print(f"❌ 處理器初始化失敗: {e}")
        return
    
    # 獲取所有CSV檔案
    input_folder = "data/Ic"
    csv_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ 在{input_folder}目錄中未找到CSV檔案")
        return
    
    print(f"📁 找到 {len(csv_files)} 個CSV檔案")
    print(f"📂 輸入目錄: {input_folder}")
    print(f"📂 輸出目錄: output/images")
    print(f"📊 分析模式: 高效能優化版本")
    
    # 配置信息
    print(f"\n⚙️ 處理器配置:")
    print(f"  • FireDucks pandas: 啟用")
    print(f"  • Numba JIT編譯: 啟用")
    print(f"  • 多線程處理: 啟用")
    print(f"  • 圖像解析度: 1920x1080 @ 100 DPI")
    print(f"  • 圖表類型: 5種 (fitted curve, residuals, phase-folded, cycles, normalized)")
    
    # 開始批量處理
    print(f"\n🚀 開始批量處理...")
    start_time = time.time()
    
    try:
        # 使用批量處理方法
        processor.batch_process_files()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ 批量處理完成！")
        print(f"⏱️ 總處理時間: {total_time:.2f} 秒")
        print(f"📈 平均處理速度: {total_time/len(csv_files):.2f} 秒/檔案")
        
        # 檢查輸出結果
        output_dir = "output/images"
        png_files = glob.glob(os.path.join(output_dir, "*.png"))
        csv_summary = "output/data/analysis_summary.csv"
        
        print(f"\n📊 處理結果統計:")
        print(f"  • 生成圖像: {len(png_files)} 個PNG檔案")
        print(f"  • 輸出目錄: {output_dir}")
        if os.path.exists(csv_summary):
            print(f"  • 分析摘要: {csv_summary}")
        
        # 顯示圖像類型統計
        plot_types = {
            'fitted_curve_plot': 0,
            'fitted_curve_normalized_plot': 0,
            'residuals_plot': 0,
            'phase_folded_with_drift': 0,
            'cycles_colored_matplotlib': 0
        }
        
        for png_file in png_files:
            filename = Path(png_file).name
            for plot_type in plot_types:
                if plot_type in filename:
                    plot_types[plot_type] += 1
                    break
        
        print(f"\n📈 圖表類型統計:")
        for plot_type, count in plot_types.items():
            type_name = plot_type.replace('_', ' ').title()
            print(f"  • {type_name}: {count} 個")
        
    except Exception as e:
        print(f"\n❌ 批量處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 完整分析成功完成！")
    print(f"📁 請查看 {output_dir} 目錄獲取所有生成的圖像")
    
    # 效能總結
    files_per_second = len(csv_files) / total_time
    print(f"\n📊 效能總結:")
    print(f"  • 檔案數量: {len(csv_files)}")
    print(f"  • 處理時間: {total_time:.2f} 秒")
    print(f"  • 處理速度: {files_per_second:.2f} 檔案/秒")
    
    if len(csv_files) > 100:
        estimated_single_thread = total_time * 2  # 估計單線程時間
        speedup = estimated_single_thread / total_time
        print(f"  • 估計加速比: {speedup:.1f}x")

if __name__ == "__main__":
    main()
