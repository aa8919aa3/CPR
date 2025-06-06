#!/usr/bin/env python3
"""
全面分析所有CSV文件的腳本
支持跳過邏輯、批量處理、詳細報告和性能監控
"""
import sys
import os
from pathlib import Path
import time
import json
import pandas as pd
from datetime import datetime
import argparse

def setup_python_path():
    """設置Python路徑以確保模組能被正確導入"""
    project_root = Path(__file__).parent.parent.absolute()
    src_path = project_root / 'src'
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(src_path) not in current_pythonpath:
        os.environ['PYTHONPATH'] = str(src_path) + os.pathsep + current_pythonpath
    
    return src_path, project_root

def create_output_directories(base_output_dir):
    """創建輸出目錄結構"""
    dirs = [
        base_output_dir,
        base_output_dir / 'images',
        base_output_dir / 'reports',
        base_output_dir / 'data'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def scan_csv_files(data_dir):
    """掃描所有CSV文件"""
    csv_files = list(Path(data_dir).glob('*.csv'))
    return sorted(csv_files)

def generate_analysis_report(results, output_dir, processing_time):
    """生成詳細的分析報告"""
    
    # 統計數據
    total_files = len(results)
    successful = [r for r in results if r.get('success', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    failed = [r for r in results if not r.get('success', False) and not r.get('skipped', False)]
    
    # 創建統計摘要
    stats = {
        'total_files': total_files,
        'successful': len(successful),
        'skipped': len(skipped),
        'failed': len(failed),
        'success_rate': len(successful) / total_files * 100 if total_files > 0 else 0,
        'skip_rate': len(skipped) / total_files * 100 if total_files > 0 else 0,
        'processing_time': processing_time,
        'average_time_per_file': processing_time / total_files if total_files > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存統計摘要
    with open(output_dir / 'reports' / 'analysis_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 創建詳細結果CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'reports' / 'detailed_results.csv', index=False)
    
    # 創建成功文件的詳細分析
    if successful:
        success_data = []
        for result in successful:
            success_data.append({
                'dataid': result.get('dataid'),
                'I_c': result.get('I_c'),
                'I_c_error': result.get('I_c_error'),
                'r_squared': result.get('r_squared'),
                'chi_squared': result.get('chi_squared'),
                'processing_time': result.get('processing_time', 0)
            })
        
        success_df = pd.DataFrame(success_data)
        success_df.to_csv(output_dir / 'reports' / 'successful_analysis.csv', index=False)
        
        # 計算統計指標
        i_c_stats = {
            'mean': float(success_df['I_c'].mean()) if len(success_df) > 0 else 0,
            'std': float(success_df['I_c'].std()) if len(success_df) > 0 else 0,
            'min': float(success_df['I_c'].min()) if len(success_df) > 0 else 0,
            'max': float(success_df['I_c'].max()) if len(success_df) > 0 else 0,
            'median': float(success_df['I_c'].median()) if len(success_df) > 0 else 0
        }
        
        r_squared_stats = {
            'mean': float(success_df['r_squared'].mean()) if len(success_df) > 0 else 0,
            'std': float(success_df['r_squared'].std()) if len(success_df) > 0 else 0,
            'min': float(success_df['r_squared'].min()) if len(success_df) > 0 else 0,
            'max': float(success_df['r_squared'].max()) if len(success_df) > 0 else 0,
            'median': float(success_df['r_squared'].median()) if len(success_df) > 0 else 0
        }
        
        stats['i_c_statistics'] = i_c_stats
        stats['r_squared_statistics'] = r_squared_stats
    
    # 創建跳過文件分析
    if skipped:
        skip_reasons = {}
        for result in skipped:
            error = result.get('error', 'Unknown')
            if error in skip_reasons:
                skip_reasons[error] += 1
            else:
                skip_reasons[error] = 1
        
        stats['skip_reasons'] = skip_reasons
    
    # 創建失敗文件分析
    if failed:
        failure_reasons = {}
        for result in failed:
            error = result.get('error', 'Unknown')
            if error in failure_reasons:
                failure_reasons[error] += 1
            else:
                failure_reasons[error] = 1
        
        stats['failure_reasons'] = failure_reasons
    
    # 重新保存完整統計
    with open(output_dir / 'reports' / 'analysis_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def print_progress_report(stats):
    """打印進度報告"""
    print("\n" + "=" * 80)
    print("📊 全面分析完成報告")
    print("=" * 80)
    
    print(f"📁 總文件數: {stats['total_files']}")
    print(f"✅ 成功處理: {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"⏭️  質量跳過: {stats['skipped']} ({stats['skip_rate']:.1f}%)")
    print(f"❌ 處理失敗: {stats['failed']}")
    print(f"⏱️  總處理時間: {stats['processing_time']:.2f} 秒")
    print(f"⚡ 平均處理時間: {stats['average_time_per_file']:.3f} 秒/文件")
    
    if 'i_c_statistics' in stats:
        print(f"\n🔬 I_c 統計 (成功處理的文件):")
        i_c_stats = stats['i_c_statistics']
        print(f"   平均值: {i_c_stats['mean']:.3e}")
        print(f"   標準差: {i_c_stats['std']:.3e}")
        print(f"   範圍: {i_c_stats['min']:.3e} ~ {i_c_stats['max']:.3e}")
        print(f"   中位數: {i_c_stats['median']:.3e}")
    
    if 'r_squared_statistics' in stats:
        print(f"\n📈 R² 統計 (擬合質量):")
        r2_stats = stats['r_squared_statistics']
        print(f"   平均值: {r2_stats['mean']:.4f}")
        print(f"   標準差: {r2_stats['std']:.4f}")
        print(f"   範圍: {r2_stats['min']:.4f} ~ {r2_stats['max']:.4f}")
        print(f"   中位數: {r2_stats['median']:.4f}")
    
    if 'skip_reasons' in stats:
        print(f"\n⏭️  跳過原因分析:")
        for reason, count in stats['skip_reasons'].items():
            print(f"   {count:3d} 文件: {reason}")
    
    if 'failure_reasons' in stats:
        print(f"\n❌ 失敗原因分析:")
        for reason, count in stats['failure_reasons'].items():
            print(f"   {count:3d} 文件: {reason}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='全面分析所有CSV文件')
    parser.add_argument('--data-dir', type=str, default='data/Ic', 
                       help='CSV文件目錄 (默認: data/Ic)')
    parser.add_argument('--output-dir', type=str, default='output/full_analysis', 
                       help='輸出目錄 (默認: output/full_analysis)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大工作線程數 (默認: 自動檢測)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='樣本大小限制 (用於測試, 默認: 處理所有文件)')
    parser.add_argument('--dry-run', action='store_true',
                       help='乾運行模式，只顯示統計不實際處理')
    
    args = parser.parse_args()
    
    print("🔧 全面CSV分析腳本啟動")
    print("=" * 60)
    
    # 設置Python路徑
    src_path, project_root = setup_python_path()
    print(f"✓ Python路徑設置: {src_path}")
    
    # 測試模組導入
    try:
        from cpr.main_processor_optimized import EnhancedJosephsonProcessor, MAX_WORKERS
        print("✓ 成功導入 EnhancedJosephsonProcessor 和 MAX_WORKERS")
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return 1
    
    # 掃描CSV文件
    data_dir = project_root / args.data_dir
    if not data_dir.exists():
        print(f"❌ 數據目錄不存在: {data_dir}")
        return 1
    
    csv_files = scan_csv_files(data_dir)
    print(f"📁 找到 {len(csv_files)} 個CSV文件")
    
    if not csv_files:
        print("❌ 沒有找到CSV文件")
        return 1
    
    # 樣本限制（用於測試）
    if args.sample_size and args.sample_size < len(csv_files):
        csv_files = csv_files[:args.sample_size]
        print(f"🔬 測試模式：只處理前 {len(csv_files)} 個文件")
    
    # 乾運行模式
    if args.dry_run:
        print("🏃 乾運行模式：分析文件但不實際處理")
        file_sizes = []
        for file_path in csv_files:
            try:
                size = file_path.stat().st_size
                file_sizes.append(size)
            except:
                file_sizes.append(0)
        
        total_size = sum(file_sizes)
        print(f"📊 文件統計:")
        print(f"   總數: {len(csv_files)}")
        print(f"   總大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"   平均大小: {total_size / len(csv_files) / 1024:.2f} KB")
        print(f"   最大文件: {max(file_sizes) / 1024:.2f} KB")
        print(f"   最小文件: {min(file_sizes) / 1024:.2f} KB")
        return 0
    
    # 創建輸出目錄
    output_dir = project_root / args.output_dir
    create_output_directories(output_dir)
    print(f"📂 輸出目錄: {output_dir}")
    
    # 創建處理器
    processor = EnhancedJosephsonProcessor()
    
    if args.max_workers:
        print(f"🔧 注意: max_workers 參數 ({args.max_workers}) 已提供，但處理器使用固定的 MAX_WORKERS = {MAX_WORKERS}")
    
    print(f"⚙️  處理器配置:")
    print(f"   最大工作線程: {MAX_WORKERS}")
    print(f"   使用Numba優化: {hasattr(processor, 'config')}")
    
    # 開始處理
    print("\n" + "=" * 60)
    print("🚀 開始全面分析")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 轉換為字符串路徑
        file_paths = [str(f) for f in csv_files]
        
        # 處理文件
        results = processor.process_files(file_paths, str(output_dir / 'images'))
        
        processing_time = time.time() - start_time
        
        # 生成報告
        print("\n🔄 生成分析報告...")
        stats = generate_analysis_report(results, output_dir, processing_time)
        
        # 打印報告
        print_progress_report(stats)
        
        # 保存結果摘要
        print(f"\n💾 報告已保存到:")
        print(f"   統計摘要: {output_dir / 'reports' / 'analysis_stats.json'}")
        print(f"   詳細結果: {output_dir / 'reports' / 'detailed_results.csv'}")
        if stats['successful'] > 0:
            print(f"   成功分析: {output_dir / 'reports' / 'successful_analysis.csv'}")
        print(f"   圖片輸出: {output_dir / 'images'}")
        
        print(f"\n🎉 分析完成！總共處理 {len(results)} 個文件")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
