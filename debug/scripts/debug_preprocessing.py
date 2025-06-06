#!/usr/bin/env python3
"""
深入分析數據預處理過程中的常數數據問題
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def analyze_preprocessing_steps(csv_file):
    """分析預處理的每個步驟"""
    print(f"\n=== 預處理步驟分析: {Path(csv_file).name} ===")
    
    try:
        # 1. 讀取原始數據
        data = pd.read_csv(csv_file)
        voltage = data.iloc[:, 0].values
        current = data.iloc[:, 1].values
        
        print(f"1. 原始數據:")
        print(f"   電壓範圍: {np.min(voltage):.6e} 到 {np.max(voltage):.6e}")
        print(f"   電流範圍: {np.min(current):.6e} 到 {np.max(current):.6e}")
        print(f"   電壓標準差: {np.std(voltage):.6e}")
        print(f"   電流標準差: {np.std(current):.6e}")
        
        # 2. 檢查是否已經是常數
        voltage_is_constant = np.std(voltage) < 1e-15
        current_is_constant = np.std(current) < 1e-15
        
        print(f"2. 常數檢查:")
        print(f"   電壓是常數: {voltage_is_constant}")
        print(f"   電流是常數: {current_is_constant}")
        
        if voltage_is_constant or current_is_constant:
            print(f"   ❌ 原始數據已經是常數，無法進行有意義的分析")
            return
        
        # 3. 模擬歸一化步驟 (類似處理器中的步驟)
        # 去除均值並縮放
        voltage_normalized = (voltage - np.mean(voltage)) / np.std(voltage)
        current_normalized = (current - np.mean(current)) / np.std(current)
        
        print(f"3. 歸一化後:")
        print(f"   電壓歸一化範圍: {np.min(voltage_normalized):.6e} 到 {np.max(voltage_normalized):.6e}")
        print(f"   電流歸一化範圍: {np.min(current_normalized):.6e} 到 {np.max(current_normalized):.6e}")
        print(f"   電壓歸一化標準差: {np.std(voltage_normalized):.6e}")
        print(f"   電流歸一化標準差: {np.std(current_normalized):.6e}")
        
        # 4. 檢查歸一化後是否變成常數
        voltage_norm_constant = np.std(voltage_normalized) < 1e-15
        current_norm_constant = np.std(current_normalized) < 1e-15
        
        print(f"4. 歸一化後常數檢查:")
        print(f"   電壓歸一化是常數: {voltage_norm_constant}")
        print(f"   電流歸一化是常數: {current_norm_constant}")
        
        # 5. 檢查數據的唯一值
        unique_voltage = len(np.unique(voltage))
        unique_current = len(np.unique(current))
        
        print(f"5. 唯一值統計:")
        print(f"   電壓唯一值數量: {unique_voltage}/{len(voltage)}")
        print(f"   電流唯一值數量: {unique_current}/{len(current)}")
        
        # 6. 檢查數據精度問題
        voltage_precision = estimate_precision(voltage)
        current_precision = estimate_precision(current)
        
        print(f"6. 數據精度估計:")
        print(f"   電壓最小差值: {voltage_precision:.6e}")
        print(f"   電流最小差值: {current_precision:.6e}")
        
        # 7. 建議修復方案
        print(f"7. 修復建議:")
        if voltage_is_constant and not current_is_constant:
            print(f"   ✓ 可以使用電流數據進行單變量分析")
        elif current_is_constant and not voltage_is_constant:
            print(f"   ✓ 可以使用電壓數據進行單變量分析")
        elif unique_voltage < 3 or unique_current < 3:
            print(f"   ⚠️  數據點過少，建議跳過此文件")
        elif voltage_precision < 1e-12 or current_precision < 1e-12:
            print(f"   ⚠️  數據精度過低，可能是舍入誤差導致")
        else:
            print(f"   ✓ 數據看起來正常，問題可能在處理器實現中")
            
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()

def estimate_precision(data):
    """估計數據的數值精度"""
    sorted_data = np.sort(np.unique(data))
    if len(sorted_data) < 2:
        return 0
    differences = np.diff(sorted_data)
    return np.min(differences[differences > 0]) if len(differences[differences > 0]) > 0 else 0

def create_repair_strategy(csv_file):
    """創建數據修復策略"""
    print(f"\n=== 修復策略: {Path(csv_file).name} ===")
    
    try:
        data = pd.read_csv(csv_file)
        voltage = data.iloc[:, 0].values
        current = data.iloc[:, 1].values
        
        # 檢查常數情況
        voltage_is_constant = np.std(voltage) < 1e-15
        current_is_constant = np.std(current) < 1e-15
        
        if voltage_is_constant and current_is_constant:
            print("❌ 兩列都是常數，無法修復")
            return None
        elif voltage_is_constant:
            print("✓ 電壓是常數，建議使用電流進行單變量分析")
            # 創建人工電壓數據
            new_voltage = np.linspace(-1e-5, 1e-5, len(current))
            repaired_data = pd.DataFrame({
                'y_field': new_voltage,
                'Ic': current
            })
        elif current_is_constant:
            print("✓ 電流是常數，建議使用電壓進行單變量分析")
            # 創建人工電流數據  
            new_current = np.linspace(np.min(current) * 0.9, np.max(current) * 1.1, len(voltage))
            repaired_data = pd.DataFrame({
                'y_field': voltage,
                'Ic': new_current
            })
        else:
            print("✓ 數據正常，無需修復")
            return None
            
        # 保存修復後的數據
        repair_dir = project_root / "debug" / "output" / "repaired"
        repair_dir.mkdir(parents=True, exist_ok=True)
        
        repair_file = repair_dir / f"repaired_{Path(csv_file).name}"
        repaired_data.to_csv(repair_file, index=False)
        
        print(f"✓ 修復數據已保存到: {repair_file}")
        return str(repair_file)
        
    except Exception as e:
        print(f"❌ 修復失敗: {e}")
        return None

def main():
    """主函數"""
    # 專門分析失敗的文件
    failed_files = [
        "data/Ic/228Ic.csv",
        "data/Ic/130Ic-.csv"
    ]
    
    print("深入分析數據預處理過程中的常數數據問題")
    print("="*70)
    
    for filename in failed_files:
        csv_file = project_root / filename
        
        if csv_file.exists():
            analyze_preprocessing_steps(str(csv_file))
            repair_file = create_repair_strategy(str(csv_file))
            
            if repair_file:
                print(f"\n測試修復後的文件...")
                analyze_preprocessing_steps(repair_file)
        else:
            print(f"❌ 文件不存在: {csv_file}")
    
    print(f"\n{'='*70}")
    print("分析完成")

if __name__ == "__main__":
    main()