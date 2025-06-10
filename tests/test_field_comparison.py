#!/usr/bin/env python3
"""
比較 30mT 和 60mT 兩個磁場條件下的 CPR 實驗結果
樣品 003-2，磁場角度掃描對比分析
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

def extract_experiment_data(df, experiment_configs):
    """提取兩個實驗的數據"""
    experiment_data = {}
    
    for exp_name, config in experiment_configs.items():
        print(f"\n🔍 提取 {exp_name} 實驗數據...")
        file_ids = [str(fid) for fid in config['file_ids']]
        
        matches = []
        for file_id in file_ids:
            patterns = [f"{file_id}Ic", f"{file_id}Ic+", f"{file_id}Ic-"]
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
                print(f"  ❌ 未找到 {file_id}")
        
        if matches:
            result_df = pd.concat(matches, ignore_index=True)
            # 添加角度資訊
            file_to_angle = {str(fid): angle for angle, fid in zip(config['angles'], config['file_ids'])}
            angles = []
            for _, row in result_df.iterrows():
                file_id = str(row['dataid']).replace('Ic', '').replace('+', '').replace('-', '')
                angle = file_to_angle.get(file_id, None)
                angles.append(angle)
            
            result_df = result_df.copy()
            result_df['angle'] = angles
            result_df = result_df[result_df['angle'].notna()]
            result_df = result_df.sort_values('angle')
            experiment_data[exp_name] = result_df
        else:
            experiment_data[exp_name] = pd.DataFrame()
    
    return experiment_data

def compare_parameters(exp_data, exp_configs):
    """比較兩個實驗的參數"""
    print(f"\n📊 30mT vs 60mT 磁場條件參數比較")
    print("=" * 80)
    
    parameters = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    param_descriptions = {
        'I_c': '臨界電流 (A)',
        'phi_0': '相位偏移 (rad)', 
        'f': '特徵頻率 (Hz)',
        'T': '透明度',
        'r': '阻尼係數',
        'C': '電容 (F)'
    }
    
    for param in parameters:
        print(f"\n🔬 {param} ({param_descriptions[param]}):")
        print("-" * 60)
        
        for exp_name, df in exp_data.items():
            if not df.empty and param in df.columns:
                success_df = df[df['success'] == True]
                values = success_df[param].dropna()
                if len(values) > 0:
                    print(f"  {exp_name:8s}: "
                          f"平均={values.mean():.6e}, "
                          f"標準差={values.std():.6e}, "
                          f"範圍=[{values.min():.6e}, {values.max():.6e}]")
                else:
                    print(f"  {exp_name:8s}: 無有效數據")
    
    # 統計指標比較
    print(f"\n📈 擬合品質比較:")
    print("-" * 60)
    
    statistics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    stat_descriptions = {
        'r_squared': 'R²決定係數',
        'adj_r_squared': '調整R²',
        'rmse': '均方根誤差',
        'mae': '平均絕對誤差'
    }
    
    for stat in statistics:
        print(f"\n📊 {stat} ({stat_descriptions[stat]}):")
        for exp_name, df in exp_data.items():
            if not df.empty and stat in df.columns:
                success_df = df[df['success'] == True]
                values = success_df[stat].dropna()
                if len(values) > 0:
                    print(f"  {exp_name:8s}: "
                          f"平均={values.mean():.6f}, "
                          f"標準差={values.std():.6f}, "
                          f"範圍=[{values.min():.6f}, {values.max():.6f}]")

def create_comparison_plots(exp_data, exp_configs, output_dir):
    """創建30mT vs 60mT 比較圖表"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 設置matplotlib
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = {'30mT': 'blue', '60mT': 'red'}
    markers = {'30mT': 'o', '60mT': 's'}
    
    # 創建參數比較圖
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Josephson Junction Parameters: 30mT vs 60mT Parallel Magnetic Field\n'
                 'Sample: 003-2, Angular Dependence Comparison', fontsize=14)
    
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
        
        # 繪製兩個實驗的數據
        for exp_name, df in exp_data.items():
            if not df.empty and param in df.columns:
                success_df = df[df['success'] == True]
                values = success_df[param].dropna()
                angles = success_df[success_df[param].notna()]['angle']
                
                if len(values) > 0:
                    ax.plot(angles, values, 
                           marker=markers[exp_name], 
                           color=colors[exp_name],
                           linewidth=2, 
                           markersize=6,
                           label=f'{exp_name}',
                           linestyle='-' if exp_name == '30mT' else '--')
        
        ax.set_title(param_labels[param])
        ax.set_xlabel('Magnetic Field Angle (°)')
        ax.set_ylabel(param_labels[param].split(' (')[0])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 設置x軸刻度
        ax.set_xlim(-10, 325)
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315])
        
        # 使用科學記數法顯示y軸（如果需要）
        if param in ['I_c', 'f', 'r', 'C']:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path / 'parameters_comparison_30mT_vs_60mT.png', dpi=300, bbox_inches='tight')
    print(f"✓ 30mT vs 60mT 參數比較圖已保存: {output_path / 'parameters_comparison_30mT_vs_60mT.png'}")
    
    # 創建統計指標比較圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Fitting Quality Comparison: 30mT vs 60mT Parallel Magnetic Field\n'
                 'Sample: 003-2', fontsize=14)
    
    statistics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    stat_labels = {
        'r_squared': 'R² (Coefficient of Determination)',
        'adj_r_squared': 'Adjusted R²',
        'rmse': 'RMSE (Root Mean Square Error)',
        'mae': 'MAE (Mean Absolute Error)'
    }
    
    for i, stat in enumerate(statistics):
        ax = axes[i//2, i%2]
        
        for exp_name, df in exp_data.items():
            if not df.empty and stat in df.columns:
                success_df = df[df['success'] == True]
                values = success_df[stat].dropna()
                angles = success_df[success_df[stat].notna()]['angle']
                
                if len(values) > 0:
                    ax.plot(angles, values,
                           marker=markers[exp_name], 
                           color=colors[exp_name],
                           linewidth=2, 
                           markersize=6,
                           label=f'{exp_name}',
                           linestyle='-' if exp_name == '30mT' else '--')
        
        ax.set_title(stat_labels[stat])
        ax.set_xlabel('Magnetic Field Angle (°)')
        ax.set_ylabel(stat_labels[stat].split(' (')[0])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 設置x軸刻度
        ax.set_xlim(-10, 325)
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315])
    
    plt.tight_layout()
    plt.savefig(output_path / 'fitting_quality_comparison_30mT_vs_60mT.png', dpi=300, bbox_inches='tight')
    print(f"✓ 30mT vs 60mT 擬合品質比較圖已保存: {output_path / 'fitting_quality_comparison_30mT_vs_60mT.png'}")
    
    # 創建關鍵參數的直接對比圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Key Parameters Direct Comparison: 30mT vs 60mT\n'
                 'Sample: 003-2', fontsize=14)
    
    key_params = ['I_c', 'T', 'f', 'r_squared']
    key_labels = ['Critical Current I_c (A)', 'Transparency T', 'Frequency f (Hz)', 'R² Quality']
    
    for i, param in enumerate(key_params):
        ax = axes[i//2, i%2]
        
        # 收集兩個條件的數據
        data_30mT = []
        data_60mT = []
        angles_30mT = []
        angles_60mT = []
        
        for exp_name, df in exp_data.items():
            if not df.empty and param in df.columns:
                success_df = df[df['success'] == True]
                values = success_df[param].dropna()
                angles = success_df[success_df[param].notna()]['angle']
                
                if exp_name == '30mT':
                    data_30mT = values.tolist()
                    angles_30mT = angles.tolist()
                else:
                    data_60mT = values.tolist()
                    angles_60mT = angles.tolist()
        
        # 創建散點圖 - 確保數據長度匹配
        if data_30mT and data_60mT:
            # 取較短數據的長度
            min_len = min(len(data_30mT), len(data_60mT))
            data_30mT_matched = data_30mT[:min_len]
            data_60mT_matched = data_60mT[:min_len]
            
            if min_len > 0:
                ax.scatter(data_30mT_matched, data_60mT_matched, s=60, alpha=0.7, color='purple')
                
                # 添加 y=x 參考線
                min_val = min(min(data_30mT_matched), min(data_60mT_matched))
                max_val = max(max(data_30mT_matched), max(data_60mT_matched))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
                
                ax.set_xlabel(f'{key_labels[i]} @ 30mT')
                ax.set_ylabel(f'{key_labels[i]} @ 60mT')
                ax.set_title(f'{param} Correlation (n={min_len})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # 計算相關係數
                if len(data_30mT_matched) > 1:
                    corr = np.corrcoef(data_30mT_matched, data_60mT_matched)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No matching data', transform=ax.transAxes, ha='center')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
        
        # 使用科學記數法（如果需要）
        if param in ['I_c', 'f']:
            ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path / 'direct_comparison_30mT_vs_60mT.png', dpi=300, bbox_inches='tight')
    print(f"✓ 30mT vs 60mT 直接對比圖已保存: {output_path / 'direct_comparison_30mT_vs_60mT.png'}")
    
    plt.close('all')

def analyze_field_dependence(exp_data):
    """分析磁場強度依賴性"""
    print(f"\n🧲 磁場強度依賴性分析")
    print("=" * 80)
    
    if '30mT' not in exp_data or '60mT' not in exp_data:
        print("❌ 缺少實驗數據")
        return
    
    df_30 = exp_data['30mT']
    df_60 = exp_data['60mT']
    
    if df_30.empty or df_60.empty:
        print("❌ 實驗數據為空")
        return
    
    # 只分析成功的記錄
    success_30 = df_30[df_30['success'] == True]
    success_60 = df_60[df_60['success'] == True]
    
    print(f"📊 數據點統計:")
    print(f"  30mT: {len(success_30)} 個成功測量點")
    print(f"  60mT: {len(success_60)} 個成功測量點")
    
    # 參數變化分析
    parameters = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    
    print(f"\n📈 關鍵參數變化 (60mT 相對於 30mT):")
    print("-" * 60)
    
    for param in parameters:
        if param in success_30.columns and param in success_60.columns:
            values_30 = success_30[param].dropna()
            values_60 = success_60[param].dropna()
            
            if len(values_30) > 0 and len(values_60) > 0:
                mean_30 = values_30.mean()
                mean_60 = values_60.mean()
                change_percent = ((mean_60 - mean_30) / mean_30) * 100
                
                param_desc = {
                    'I_c': '臨界電流',
                    'phi_0': '相位偏移', 
                    'f': '特徵頻率',
                    'T': '透明度',
                    'r': '阻尼係數',
                    'C': '電容'
                }
                
                print(f"  {param:8s} ({param_desc[param]:8s}): "
                      f"30mT={mean_30:.6e}, 60mT={mean_60:.6e}, "
                      f"變化={change_percent:+.1f}%")

def main():
    """主函數"""
    # 定義兩個實驗的配置
    experiment_configs = {
        '30mT': {
            'sample_id': '003-2',
            'condition': 'CPR@30mT',
            'description': 'Current-Phase Relation at 30mT parallel magnetic field',
            'angles': [0, 45, 58.7, 90, 135, 140.7, 180, 225, 270, 315],
            'file_ids': [386, 381, 418, 397, 394, 416, 396, 407, 380]  # 缺少 315° 的數據點
        },
        '60mT': {
            'sample_id': '003-2',
            'condition': 'CPR@60mT',
            'description': 'Current-Phase Relation at 60mT parallel magnetic field',
            'angles': [0, 45, 58.7, 90, 135, 140.7, 180, 225, 270, 315],
            'file_ids': [317, 346, 435, 338, 337, 439, 336, 352, 335, 341]
        }
    }
    
    # CSV文件路徑
    csv_path = "/Users/albert-mac/Code/GitHub/CPR/output/full_analysis/images/analysis_summary.csv"
    
    print("🔬 CPR 磁場強度對比分析 (30mT vs 60mT)")
    print("=" * 80)
    print("樣品編號: 003-2")
    print("測量條件: Current-Phase Relation 角度掃描")
    print("磁場強度: 30mT vs 60mT 平行磁場")
    print(f"數據來源: {csv_path}")
    print()
    
    # 讀取數據
    df = load_analysis_data(csv_path)
    if df is None:
        return 1
    
    # 提取兩個實驗的數據
    exp_data = extract_experiment_data(df, experiment_configs)
    
    # 檢查是否成功提取數據
    if not exp_data or all(df.empty for df in exp_data.values()):
        print("❌ 未找到任何匹配的實驗數據")
        return 1
    
    # 比較參數
    compare_parameters(exp_data, experiment_configs)
    
    # 磁場依賴性分析
    analyze_field_dependence(exp_data)
    
    # 創建比較圖表
    output_dir = "/Users/albert-mac/Code/GitHub/CPR/output/parameter_analysis"
    create_comparison_plots(exp_data, experiment_configs, output_dir)
    
    print(f"\n✅ 30mT vs 60mT 對比分析完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
