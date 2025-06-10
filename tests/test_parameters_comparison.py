#!/usr/bin/env python3
"""
分析特定檔案的參數和統計數據
從 analysis_summary.csv 中提取並比較指定檔案的結果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def load_analysis_data(csv_path):
    """讀取分析結果CSV文件"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ 成功讀取 {len(df)} 行數據")
        return df
    except Exception as e:
        print(f"❌ 讀取CSV文件失敗: {e}")
        return None

def extract_file_data(df, file_ids):
    """提取指定檔案的數據"""
    # 清理檔案ID，移除可能的後綴
    clean_ids = []
    for fid in file_ids:
        # 將字符串轉換為整數再轉回字符串，去除前導零等
        try:
            clean_id = str(int(str(fid).replace('Ic', '')))
            clean_ids.append(clean_id)
        except:
            clean_ids.append(str(fid))
    
    print(f"🔍 搜尋檔案ID: {clean_ids}")
    
    # 搜尋匹配的記錄
    matches = []
    for clean_id in clean_ids:
        # 嘗試多種匹配模式
        patterns = [
            f"{clean_id}Ic",
            f"{clean_id}Ic+", 
            f"{clean_id}Ic-",
            clean_id
        ]
        
        found = False
        for pattern in patterns:
            mask = df['dataid'].str.contains(pattern, na=False)
            if mask.any():
                matched_rows = df[mask]
                print(f"  ✓ 找到 {pattern}: {len(matched_rows)} 行")
                matches.append(matched_rows)
                found = True
                break
        
        if not found:
            print(f"  ❌ 未找到 {clean_id}")
    
    if matches:
        result_df = pd.concat(matches, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()

def analyze_parameters(df):
    """分析參數統計"""
    if df.empty:
        print("❌ 沒有數據可分析")
        return
    
    print(f"\n📊 分析結果 ({len(df)} 個檔案)")
    print("=" * 80)
    
    # 只分析成功的記錄
    success_df = df[df['success'] == True]
    if success_df.empty:
        print("❌ 沒有成功處理的檔案")
        return
    
    print(f"✓ 成功處理的檔案: {len(success_df)}")
    
    # 參數統計
    parameters = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    statistics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    
    print(f"\n📈 參數統計:")
    print("-" * 60)
    
    for param in parameters:
        if param in success_df.columns:
            values = success_df[param].dropna()
            if len(values) > 0:
                print(f"{param:8s}: "
                      f"平均={values.mean():.6e}, "
                      f"標準差={values.std():.6e}, "
                      f"範圍=[{values.min():.6e}, {values.max():.6e}]")
    
    print(f"\n📊 統計指標:")
    print("-" * 60)
    
    for stat in statistics:
        if stat in success_df.columns:
            values = success_df[stat].dropna()
            if len(values) > 0:
                print(f"{stat:15s}: "
                      f"平均={values.mean():.6f}, "
                      f"標準差={values.std():.6f}, "
                      f"範圍=[{values.min():.6f}, {values.max():.6f}]")

def display_detailed_results(df):
    """顯示詳細結果"""
    if df.empty:
        return
    
    print(f"\n📋 詳細結果:")
    print("=" * 120)
    
    # 設置顯示選項
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    for idx, row in df.iterrows():
        print(f"\n檔案: {row['dataid']}")
        print(f"狀態: {'✓ 成功' if row['success'] else '❌ 失敗'}")
        
        if row['success']:
            print(f"參數:")
            print(f"  I_c    = {row['I_c']:.6e}")
            print(f"  phi_0  = {row['phi_0']:.6f}")
            print(f"  f      = {row['f']:.6e}")
            print(f"  T      = {row['T']:.6f}")
            print(f"  r      = {row['r']:.6e}")
            print(f"  C      = {row['C']:.6e}")
            
            print(f"統計:")
            print(f"  R²     = {row['r_squared']:.6f}")
            print(f"  Adj R² = {row['adj_r_squared']:.6f}")
            print(f"  RMSE   = {row['rmse']:.6e}")
            print(f"  MAE    = {row['mae']:.6e}")
        else:
            if pd.notna(row.get('error')):
                print(f"錯誤: {row['error']}")

def create_comparison_plots(df, output_dir):
    """創建比較圖表"""
    if df.empty:
        return
    
    success_df = df[df['success'] == True]
    if len(success_df) < 2:
        print("⚠️ 成功檔案數量不足，跳過圖表生成")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 設置matplotlib中文字體
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建參數比較圖
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Parameters Comparison', fontsize=16)
    
    parameters = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    
    for i, param in enumerate(parameters):
        ax = axes[i//3, i%3]
        values = success_df[param].dropna()
        files = success_df[success_df[param].notna()]['dataid']
        
        if len(values) > 0:
            bars = ax.bar(range(len(values)), values)
            ax.set_title(f'{param}')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([f.replace('Ic', '') for f in files], rotation=45)
            ax.tick_params(axis='x', labelsize=8)
            
            # 使用科學記數法顯示y軸
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path / 'parameters_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 參數比較圖已保存: {output_path / 'parameters_comparison.png'}")
    
    # 創建統計指標比較圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Statistics Comparison', fontsize=16)
    
    statistics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    
    for i, stat in enumerate(statistics):
        ax = axes[i//2, i%2]
        values = success_df[stat].dropna()
        files = success_df[success_df[stat].notna()]['dataid']
        
        if len(values) > 0:
            bars = ax.bar(range(len(values)), values)
            ax.set_title(f'{stat}')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([f.replace('Ic', '') for f in files], rotation=45)
            ax.tick_params(axis='x', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 統計指標比較圖已保存: {output_path / 'statistics_comparison.png'}")
    
    plt.close('all')

def analyze_parameters_with_angles(df, angle_file_map):
    """分析參數統計（帶角度資訊）"""
    if df.empty:
        print("❌ 沒有數據可分析")
        return
    
    print(f"\n📊 CPR 參數角度依賴性分析 ({len(df)} 個測量點)")
    print("=" * 80)
    
    # 只分析成功的記錄
    success_df = df[df['success'] == True]
    if success_df.empty:
        print("❌ 沒有成功處理的檔案")
        return
    
    print(f"✓ 成功處理的測量點: {len(success_df)}")
    
    # 為每個成功的檔案添加角度資訊
    success_df = success_df.copy()
    file_to_angle = {str(fid): angle for angle, fid in angle_file_map.items()}
    
    angles = []
    for _, row in success_df.iterrows():
        file_id = str(row['dataid']).replace('Ic', '').replace('+', '').replace('-', '')
        angle = file_to_angle.get(file_id, None)
        angles.append(angle)
    
    success_df['angle'] = angles
    success_df = success_df[success_df['angle'].notna()]  # 只保留有角度資訊的記錄
    success_df = success_df.sort_values('angle')  # 按角度排序
    
    # 參數統計
    parameters = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    statistics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    
    print(f"\n📈 約瑟夫森結參數統計:")
    print("-" * 60)
    print("說明: T = Transparency(透明度), I_c = 臨界電流, phi_0 = 相位偏移")
    print("      f = 特徵頻率, r = 阻尼係數, C = 電容")
    print()
    
    for param in parameters:
        if param in success_df.columns:
            values = success_df[param].dropna()
            if len(values) > 0:
                param_desc = {
                    'I_c': '臨界電流',
                    'phi_0': '相位偏移', 
                    'f': '特徵頻率',
                    'T': '透明度',
                    'r': '阻尼係數',
                    'C': '電容'
                }
                print(f"{param:8s} ({param_desc[param]:8s}): "
                      f"平均={values.mean():.6e}, "
                      f"標準差={values.std():.6e}, "
                      f"範圍=[{values.min():.6e}, {values.max():.6e}]")
    
    print(f"\n📊 擬合品質統計:")
    print("-" * 60)
    
    for stat in statistics:
        if stat in success_df.columns:
            values = success_df[stat].dropna()
            if len(values) > 0:
                stat_desc = {
                    'r_squared': 'R²決定係數',
                    'adj_r_squared': '調整R²',
                    'rmse': '均方根誤差',
                    'mae': '平均絕對誤差'
                }
                print(f"{stat:15s} ({stat_desc[stat]:10s}): "
                      f"平均={values.mean():.6f}, "
                      f"標準差={values.std():.6f}, "
                      f"範圍=[{values.min():.6f}, {values.max():.6f}]")

def display_detailed_results_with_angles(df, angle_file_map):
    """顯示詳細結果（帶角度資訊）"""
    if df.empty:
        return
    
    # 創建檔案ID到角度的映射
    file_to_angle = {str(fid): angle for angle, fid in angle_file_map.items()}
    
    # 為DataFrame添加角度資訊並排序
    df_with_angles = df.copy()
    angles = []
    for _, row in df_with_angles.iterrows():
        file_id = str(row['dataid']).replace('Ic', '').replace('+', '').replace('-', '')
        angle = file_to_angle.get(file_id, None)
        angles.append(angle)
    
    df_with_angles['angle'] = angles
    df_with_angles = df_with_angles[df_with_angles['angle'].notna()]
    df_with_angles = df_with_angles.sort_values('angle')
    
    print(f"\n📋 磁場角度掃描詳細結果:")
    print("=" * 120)
    print("樣品: 003-2, 條件: CPR@30mT (Current-Phase Relation at 30mT parallel field)")
    print()
    
    for idx, row in df_with_angles.iterrows():
        angle = row['angle']
        print(f"🧲 磁場角度: {angle:6.1f}° | 檔案: {row['dataid']}")
        print(f"   狀態: {'✓ 成功' if row['success'] else '❌ 失敗'}")
        
        if row['success']:
            print(f"   約瑟夫森結參數:")
            print(f"     I_c (臨界電流)    = {row['I_c']:.6e} A")
            print(f"     phi_0 (相位偏移)  = {row['phi_0']:.6f} rad")
            print(f"     f (特徵頻率)      = {row['f']:.6e} Hz")
            print(f"     T (透明度)        = {row['T']:.6f}")
            print(f"     r (阻尼係數)      = {row['r']:.6e}")
            print(f"     C (電容)          = {row['C']:.6e} F")
            
            print(f"   擬合品質:")
            print(f"     R² (決定係數)     = {row['r_squared']:.6f}")
            print(f"     調整 R²           = {row['adj_r_squared']:.6f}")
            print(f"     RMSE (均方根誤差) = {row['rmse']:.6e}")
            print(f"     MAE (平均絕對誤差)= {row['mae']:.6e}")
        else:
            if pd.notna(row.get('error')):
                print(f"   錯誤: {row['error']}")
        print()

def create_comparison_plots_with_angles(df, angle_file_map, experiment_info, output_dir):
    """創建帶角度資訊的比較圖表"""
    if df.empty:
        return
    
    success_df = df[df['success'] == True]
    if len(success_df) < 2:
        print("⚠️ 成功檔案數量不足，跳過圖表生成")
        return
    
    # 添加角度資訊並排序
    file_to_angle = {str(fid): angle for angle, fid in angle_file_map.items()}
    angles = []
    for _, row in success_df.iterrows():
        file_id = str(row['dataid']).replace('Ic', '').replace('+', '').replace('-', '')
        angle = file_to_angle.get(file_id, None)
        angles.append(angle)
    
    success_df = success_df.copy()
    success_df['angle'] = angles
    success_df = success_df[success_df['angle'].notna()]
    success_df = success_df.sort_values('angle')
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 設置matplotlib
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建參數隨角度變化圖
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Josephson Junction Parameters vs Magnetic Field Angle\n'
                 f'Sample: {experiment_info["sample_id"]}, Condition: {experiment_info["condition"]}', 
                 fontsize=14)
    
    parameters = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    param_labels = {
        'I_c': 'Critical Current I_c (A)',
        'phi_0': 'Phase Offset φ₀ (rad)', 
        'f': 'Characteristic Frequency f (Hz)',
        'T': 'Transparency T',
        'r': 'Damping Coefficient r',
        'C': 'Capacitance C (F)'
    }
    
    for i, param in enumerate(parameters):
        ax = axes[i//2, i%2]
        values = success_df[param].dropna()
        angles_for_param = success_df[success_df[param].notna()]['angle']
        
        if len(values) > 0:
            # 用線條和點連接，顯示角度依賴性
            ax.plot(angles_for_param, values, 'o-', linewidth=2, markersize=6)
            ax.set_title(param_labels[param])
            ax.set_xlabel('Magnetic Field Angle (°)')
            ax.set_ylabel(param_labels[param].split(' (')[0])
            ax.grid(True, alpha=0.3)
            
            # 設置x軸刻度
            ax.set_xlim(-10, 325)
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315])
            
            # 使用科學記數法顯示y軸（如果需要）
            if param in ['I_c', 'f', 'r', 'C']:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path / 'parameters_vs_angle.png', dpi=300, bbox_inches='tight')
    print(f"✓ 參數角度依賴性圖已保存: {output_path / 'parameters_vs_angle.png'}")
    
    # 創建統計指標圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Fitting Quality vs Magnetic Field Angle\n'
                 f'Sample: {experiment_info["sample_id"]}, Condition: {experiment_info["condition"]}', 
                 fontsize=14)
    
    statistics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    stat_labels = {
        'r_squared': 'R² (Coefficient of Determination)',
        'adj_r_squared': 'Adjusted R²',
        'rmse': 'RMSE (Root Mean Square Error)',
        'mae': 'MAE (Mean Absolute Error)'
    }
    
    for i, stat in enumerate(statistics):
        ax = axes[i//2, i%2]
        values = success_df[stat].dropna()
        angles_for_stat = success_df[success_df[stat].notna()]['angle']
        
        if len(values) > 0:
            ax.plot(angles_for_stat, values, 'o-', linewidth=2, markersize=6)
            ax.set_title(stat_labels[stat])
            ax.set_xlabel('Magnetic Field Angle (°)')
            ax.set_ylabel(stat_labels[stat].split(' (')[0])
            ax.grid(True, alpha=0.3)
            
            # 設置x軸刻度
            ax.set_xlim(-10, 325)
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315])
    
    plt.tight_layout()
    plt.savefig(output_path / 'fitting_quality_vs_angle.png', dpi=300, bbox_inches='tight')
    print(f"✓ 擬合品質角度依賴性圖已保存: {output_path / 'fitting_quality_vs_angle.png'}")
    
    plt.close('all')

def main():
    """主函數"""
    # 實驗設計資訊 - 樣品 003-2 在 30mT 平行磁場下不同角度的 CPR 測量
    experiment_info = {
        'sample_id': '003-2',
        'condition': 'CPR@30mT',
        'description': 'Current-Phase Relation at 30mT parallel magnetic field',
        'angles': [0, 45, 58.7, 90, 135, 140.7, 180, 225, 270, 315],  # 磁場角度 (度)
        'file_ids': [386, 381, 418, 397, 394, 416, 396, 407, 380]     # 對應檔案ID
    }
    
    # 創建角度到檔案ID的映射
    angle_file_map = dict(zip(experiment_info['angles'], experiment_info['file_ids']))
    
    # CSV文件路徑
    csv_path = "/Users/albert-mac/Code/GitHub/CPR/output/full_analysis/images/analysis_summary.csv"
    
    print("🔬 Current-Phase Relation 參數角度依賴性分析")
    print("=" * 80)
    print(f"樣品編號: {experiment_info['sample_id']}")
    print(f"測量條件: {experiment_info['description']}")
    print(f"角度範圍: {min(experiment_info['angles'])}° - {max(experiment_info['angles'])}°")
    print(f"測量點數: {len(experiment_info['file_ids'])} 個角度")
    print(f"數據來源: {csv_path}")
    print("\n📐 角度-檔案對應:")
    for angle, file_id in angle_file_map.items():
        print(f"  {angle:6.1f}° → 檔案 {file_id}")
    print()
    
    # 讀取數據
    df = load_analysis_data(csv_path)
    if df is None:
        return 1
    
    # 提取指定檔案的數據
    target_df = extract_file_data(df, experiment_info['file_ids'])
    
    if target_df.empty:
        print("❌ 未找到任何匹配的檔案")
        return 1
    
    # 分析參數（傳入角度資訊）
    analyze_parameters_with_angles(target_df, angle_file_map)
    
    # 顯示詳細結果（傳入角度資訊）
    display_detailed_results_with_angles(target_df, angle_file_map)
    
    # 創建比較圖表（傳入實驗資訊）
    output_dir = "/Users/albert-mac/Code/GitHub/CPR/output/parameter_analysis"
    create_comparison_plots_with_angles(target_df, angle_file_map, experiment_info, output_dir)
    
    print(f"\n✅ 分析完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
