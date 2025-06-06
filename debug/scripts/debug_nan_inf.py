#!/usr/bin/env python3
"""
專門調試 NaN/inf 值問題的工具
分析數據預處理過程中出現的 NaN/inf 值並提供解決方案
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor

def analyze_data_quality(csv_file):
    """分析數據質量，查找NaN/inf問題的根源"""
    print(f"\n{'='*60}")
    print(f"數據質量分析: {Path(csv_file).name}")
    print(f"{'='*60}")
    
    try:
        # 讀取原始數據
        df = pd.read_csv(csv_file)
        print(f"原始數據形狀: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 檢查必要的列
        required_columns = ['y_field', 'Ic']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 缺少必要的列: {missing_columns}")
            return False
            
        # 提取數據
        y_field = df['y_field'].values
        Ic = df['Ic'].values
        
        print(f"\n--- 原始數據統計 ---")
        print(f"y_field: min={np.min(y_field):.6f}, max={np.max(y_field):.6f}")
        print(f"Ic: min={np.min(Ic):.6e}, max={np.max(Ic):.6e}")
        
        # 檢查NaN值
        y_nan_count = np.isnan(y_field).sum()
        ic_nan_count = np.isnan(Ic).sum()
        print(f"NaN值: y_field={y_nan_count}, Ic={ic_nan_count}")
        
        # 檢查inf值
        y_inf_count = np.isinf(y_field).sum()
        ic_inf_count = np.isinf(Ic).sum()
        print(f"Inf值: y_field={y_inf_count}, Ic={ic_inf_count}")
        
        # 檢查零值
        y_zero_count = (y_field == 0).sum()
        ic_zero_count = (Ic == 0).sum()
        print(f"零值: y_field={y_zero_count}, Ic={ic_zero_count}")
        
        # 模擬預處理步驟
        print(f"\n--- 預處理模擬 ---")
        
        # 1. 移除無效值
        valid_mask = np.isfinite(y_field) & np.isfinite(Ic) & (Ic != 0)
        print(f"有效數據點: {valid_mask.sum()}/{len(valid_mask)}")
        
        if valid_mask.sum() < 10:
            print(f"❌ 有效數據點太少: {valid_mask.sum()}")
            return False
            
        y_clean = y_field[valid_mask]
        ic_clean = Ic[valid_mask]
        
        # 2. 數據範圍檢查
        y_range = np.max(y_clean) - np.min(y_clean)
        ic_range = np.max(ic_clean) - np.min(ic_clean)
        print(f"數據範圍: y_field={y_range:.6f}, Ic={ic_range:.6e}")
        
        if y_range == 0:
            print(f"❌ y_field範圍為零")
            return False
        if ic_range == 0:
            print(f"❌ Ic範圍為零")
            return False
            
        # 3. 正規化模擬
        try:
            y_normalized = (y_clean - np.min(y_clean)) / (np.max(y_clean) - np.min(y_clean))
            ic_normalized = (ic_clean - np.min(ic_clean)) / (np.max(ic_clean) - np.min(ic_clean))
            
            print(f"正規化後範圍: y=[{np.min(y_normalized):.6f}, {np.max(y_normalized):.6f}]")
            print(f"正規化後範圍: Ic=[{np.min(ic_normalized):.6f}, {np.max(ic_normalized):.6f}]")
            
            # 檢查正規化後的NaN/inf
            y_norm_nan = np.isnan(y_normalized).sum()
            y_norm_inf = np.isinf(y_normalized).sum()
            ic_norm_nan = np.isnan(ic_normalized).sum()
            ic_norm_inf = np.isinf(ic_normalized).sum()
            
            print(f"正規化後NaN: y_norm={y_norm_nan}, ic_norm={ic_norm_nan}")
            print(f"正規化後Inf: y_norm={y_norm_inf}, ic_norm={ic_norm_inf}")
            
            if y_norm_nan > 0 or y_norm_inf > 0 or ic_norm_nan > 0 or ic_norm_inf > 0:
                print(f"❌ 正規化後出現NaN/Inf值")
                return False
                
        except Exception as e:
            print(f"❌ 正規化過程出錯: {e}")
            return False
            
        # 4. 數據分布檢查
        print(f"\n--- 數據分布分析 ---")
        y_std = np.std(y_clean)
        ic_std = np.std(ic_clean)
        print(f"標準差: y_field={y_std:.6f}, Ic={ic_std:.6e}")
        
        # 檢查異常值
        y_q1, y_q3 = np.percentile(y_clean, [25, 75])
        ic_q1, ic_q3 = np.percentile(ic_clean, [25, 75])
        y_iqr = y_q3 - y_q1
        ic_iqr = ic_q3 - ic_q1
        
        y_outliers = ((y_clean < y_q1 - 1.5*y_iqr) | (y_clean > y_q3 + 1.5*y_iqr)).sum()
        ic_outliers = ((ic_clean < ic_q1 - 1.5*ic_iqr) | (ic_clean > ic_q3 + 1.5*ic_iqr)).sum()
        
        print(f"異常值: y_field={y_outliers}, Ic={ic_outliers}")
        
        print(f"✅ 數據質量檢查通過")
        return True
        
    except Exception as e:
        print(f"❌ 數據分析錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_nan_inf_files():
    """調試特定的NaN/inf問題文件"""
    
    # 問題文件列表
    problem_files = [
        "228Ic.csv",
        "130Ic-.csv"
    ]
    
    print("CPR NaN/Inf 問題調試工具")
    print("="*60)
    
    data_dir = "data/Ic"
    found_files = []
    
    # 查找問題文件
    for problem_file in problem_files:
        file_path = os.path.join(data_dir, problem_file)
        if os.path.exists(file_path):
            found_files.append(file_path)
            print(f"✅ 找到問題文件: {problem_file}")
        else:
            print(f"❌ 未找到文件: {problem_file}")
    
    # 分析每個問題文件
    for csv_file in found_files:
        success = analyze_data_quality(csv_file)
        
        if not success:
            print(f"\n--- 嘗試修復數據 ---")
            try_repair_data(csv_file)

def try_repair_data(csv_file):
    """嘗試修復有問題的數據"""
    print(f"嘗試修復: {Path(csv_file).name}")
    
    try:
        df = pd.read_csv(csv_file)
        original_shape = df.shape
        
        # 移除包含NaN或Inf的行
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"數據修復: {original_shape} -> {df_clean.shape}")
        
        if len(df_clean) < 10:
            print(f"❌ 修復後數據點太少: {len(df_clean)}")
            return False
            
        # 保存修復後的數據
        repair_dir = "debug/output/repaired_data"
        os.makedirs(repair_dir, exist_ok=True)
        
        repair_file = os.path.join(repair_dir, Path(csv_file).name)
        df_clean.to_csv(repair_file, index=False)
        
        print(f"✅ 修復的數據已保存: {repair_file}")
        
        # 測試修復後的數據
        print(f"--- 測試修復後的數據 ---")
        processor = EnhancedJosephsonProcessor()
        
        try:
            result = processor.process_single_file(repair_file, "debug/output")
            if result['success']:
                print(f"✅ 修復成功！可以正常處理")
                return True
            else:
                print(f"❌ 修復後仍然失敗: {result.get('error', '未知錯誤')}")
                return False
        except Exception as e:
            print(f"❌ 測試修復數據時出錯: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 數據修復失敗: {e}")
        return False

def scan_all_files_for_issues():
    """掃描所有文件，找出有NaN/inf問題的文件"""
    print(f"\n{'='*60}")
    print("掃描所有CSV文件，查找數據質量問題")
    print(f"{'='*60}")
    
    data_dir = "data/Ic"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"總共找到 {len(csv_files)} 個CSV文件")
    
    problem_files = []
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] 檢查: {Path(csv_file).name}", end=" ")
        
        try:
            df = pd.read_csv(csv_file)
            
            if 'y_field' not in df.columns or 'Ic' not in df.columns:
                print("❌ 缺少必要列")
                problem_files.append((csv_file, "missing_columns"))
                continue
                
            y_field = df['y_field'].values
            ic = df['Ic'].values
            
            # 檢查NaN/inf
            has_nan = np.isnan(y_field).any() or np.isnan(ic).any()
            has_inf = np.isinf(y_field).any() or np.isinf(ic).any()
            
            # 檢查數據範圍
            valid_mask = np.isfinite(y_field) & np.isfinite(ic) & (ic != 0)
            valid_count = valid_mask.sum()
            
            if has_nan or has_inf:
                print("❌ 包含NaN/Inf")
                problem_files.append((csv_file, "nan_inf"))
            elif valid_count < 10:
                print("❌ 有效數據點太少")
                problem_files.append((csv_file, "insufficient_data"))
            elif len(np.unique(y_field[valid_mask])) < 5:
                print("❌ y_field值過少")
                problem_files.append((csv_file, "insufficient_y_values"))
            else:
                print("✅")
                
        except Exception as e:
            print(f"❌ 讀取錯誤: {e}")
            problem_files.append((csv_file, f"read_error: {e}"))
    
    print(f"\n{'='*60}")
    print(f"掃描完成！發現 {len(problem_files)} 個問題文件:")
    print(f"{'='*60}")
    
    issue_types = {}
    for file_path, issue_type in problem_files:
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(Path(file_path).name)
    
    for issue_type, files in issue_types.items():
        print(f"\n{issue_type} ({len(files)} 個文件):")
        for file_name in files[:10]:  # 只顯示前10個
            print(f"  - {file_name}")
        if len(files) > 10:
            print(f"  ... 還有 {len(files)-10} 個文件")
    
    return problem_files

def main():
    """主函數"""
    print("CPR 數據質量調試工具")
    print("="*60)
    
    # 首先調試特定的問題文件
    debug_nan_inf_files()
    
    # 然後掃描所有文件
    problem_files = scan_all_files_for_issues()
    
    # 提供修復建議
    if problem_files:
        print(f"\n{'='*60}")
        print("修復建議:")
        print(f"{'='*60}")
        print("1. NaN/Inf問題: 使用數據清理功能移除無效值")
        print("2. 數據點不足: 檢查原始測量數據")
        print("3. y_field值過少: 檢查磁場掃描範圍")
        print("4. 讀取錯誤: 檢查文件格式和編碼")

if __name__ == "__main__":
    main()
