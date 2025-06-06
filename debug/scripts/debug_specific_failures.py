#!/usr/bin/env python3
"""
調試特定問題文件的 NaN/inf 值問題
專門針對：228Ic.csv, 130Ic-.csv 等文件
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor

def analyze_data_issues(csv_file):
    """詳細分析數據問題"""
    print(f"\n=== 詳細分析: {Path(csv_file).name} ===")
    
    try:
        # 讀取原始數據
        data = pd.read_csv(csv_file)
        print(f"原始數據形狀: {data.shape}")
        print(f"列名: {list(data.columns)}")
        
        # 檢查每列的基本統計
        print("\n數據統計:")
        print(data.describe())
        
        # 檢查 NaN 值
        nan_count = data.isnull().sum()
        print(f"\nNaN 值統計:")
        for col, count in nan_count.items():
            if count > 0:
                print(f"  {col}: {count} NaN 值 ({count/len(data)*100:.1f}%)")
        
        # 檢查 inf 值
        print(f"\ninf 值檢查:")
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32]:
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    print(f"  {col}: {inf_count} inf 值")
        
        # 檢查零值
        print(f"\n零值檢查:")
        for col in data.columns:
            zero_count = (data[col] == 0).sum()
            if zero_count > 0:
                print(f"  {col}: {zero_count} 零值 ({zero_count/len(data)*100:.1f}%)")
        
        # 檢查數據範圍
        print(f"\n數據範圍:")
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32]:
                valid_data = data[col].dropna()
                if len(valid_data) > 0:
                    print(f"  {col}: {valid_data.min():.6e} 到 {valid_data.max():.6e}")
        
        return data
        
    except Exception as e:
        print(f"❌ 讀取文件失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_processing_steps(csv_file):
    """逐步調試處理過程"""
    print(f"\n=== 處理步驟調試: {Path(csv_file).name} ===")
    
    try:
        processor = EnhancedJosephsonProcessor()
        
        # 讀取數據
        data = pd.read_csv(csv_file)
        print(f"1. 數據讀取成功: {data.shape}")
        
        # 檢查數據列
        if len(data.columns) < 2:
            print(f"❌ 數據列不足: {len(data.columns)} < 2")
            return
        
        voltage = data.iloc[:, 0].values
        current = data.iloc[:, 1].values
        
        print(f"2. 電壓數據: {len(voltage)} 點")
        print(f"   範圍: {np.min(voltage):.6e} 到 {np.max(voltage):.6e}")
        print(f"   NaN: {np.isnan(voltage).sum()}")
        print(f"   inf: {np.isinf(voltage).sum()}")
        
        print(f"3. 電流數據: {len(current)} 點")
        print(f"   範圍: {np.min(current):.6e} 到 {np.max(current):.6e}")
        print(f"   NaN: {np.isnan(current).sum()}")
        print(f"   inf: {np.isinf(current).sum()}")
        
        # 檢查有效數據點
        valid_mask = ~(np.isnan(voltage) | np.isnan(current) | 
                      np.isinf(voltage) | np.isinf(current))
        valid_count = np.sum(valid_mask)
        
        print(f"4. 有效數據點: {valid_count}/{len(voltage)} ({valid_count/len(voltage)*100:.1f}%)")
        
        if valid_count < 10:
            print(f"❌ 有效數據點不足: {valid_count} < 10")
            return
        
        # 嘗試處理
        output_dir = project_root / "debug" / "output"
        output_dir.mkdir(exist_ok=True)
        
        result = processor.process_single_file(csv_file, str(output_dir))
        
        if result['success']:
            print(f"✅ 處理成功")
            print(f"   I_c: {result.get('I_c', 'N/A')}")
            print(f"   R²: {result.get('r_squared', 'N/A')}")
        else:
            print(f"❌ 處理失敗: {result.get('error', '未知錯誤')}")
            
    except Exception as e:
        print(f"❌ 處理異常: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    # 測試特定問題文件
    problem_files = [
        "data/Ic/228Ic.csv",
        "data/Ic/130Ic-.csv",
        "data/Ic/394Ic.csv",
        "data/Ic/175Ic.csv", 
        "data/Ic/401Ic.csv"
    ]
    
    print("調試特定問題文件的 NaN/inf 值問題")
    print("="*60)
    
    data_dir = project_root / "data" / "Ic"
    
    for filename in problem_files:
        csv_file = project_root / filename
        
        if csv_file.exists():
            print(f"\n{'='*60}")
            analyze_data_issues(str(csv_file))
            debug_processing_steps(str(csv_file))
        else:
            print(f"❌ 文件不存在: {csv_file}")
    
    print(f"\n{'='*60}")
    print("調試完成")

if __name__ == "__main__":
    main()