#!/usr/bin/env python3
"""
分析特定文件的參數和統計數據
針對文件 ID: 386, 381, 418, 397, 394, 416, 396, 407, 380
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 目標文件 ID 列表
TARGET_FILES = [386, 381, 418, 397, 394, 416, 396, 407, 380]

# 文件路徑
ANALYSIS_SUMMARY_PATH = "/Users/albert-mac/Code/GitHub/CPR/output/full_analysis/images/analysis_summary.csv"
OUTPUT_DIR = Path("/Users/albert-mac/Code/GitHub/CPR/output/specific_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_analysis_data():
    """載入分析摘要數據"""
    print("正在載入分析摘要數據...")
    df = pd.read_csv(ANALYSIS_SUMMARY_PATH)
    print(f"總共載入 {len(df)} 條記錄")
    return df

def extract_target_data(df):
    """提取目標文件的數據"""
    print(f"正在提取目標文件數據: {TARGET_FILES}")
    
    # 創建匹配模式：dataid包含目標數字並以"Ic"結尾
    target_patterns = [f"{file_id}Ic" for file_id in TARGET_FILES]
    
    target_data = []
    for pattern in target_patterns:
        matches = df[df['dataid'].str.contains(pattern, na=False)]
        if not matches.empty:
            # 如果有多個匹配，取第一個
            target_data.append(matches.iloc[0])
            print(f"找到文件 {pattern}: {matches.iloc[0]['dataid']}")
        else:
            print(f"警告: 未找到文件 {pattern}")
    
    if target_data:
        target_df = pd.DataFrame(target_data)
        return target_df
    else:
        return pd.DataFrame()

def analyze_parameters(df):
    """分析物理參數"""
    print("\n=== 物理參數分析 ===")
    
    # 主要物理參數
    params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    
    analysis_results = {}
    
    for param in params:
        if param in df.columns:
            values = pd.to_numeric(df[param], errors='coerce')
            values = values.dropna()
            
            if len(values) > 0:
                analysis_results[param] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'count': len(values)
                }
                
                print(f"\n{param}:")
                print(f"  平均值: {values.mean():.2e}")
                print(f"  標準差: {values.std():.2e}")
                print(f"  範圍: [{values.min():.2e}, {values.max():.2e}]")
                print(f"  中位數: {values.median():.2e}")
    
    return analysis_results

def analyze_quality_metrics(df):
    """分析擬合品質指標"""
    print("\n=== 擬合品質指標分析 ===")
    
    quality_metrics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    
    quality_results = {}
    
    for metric in quality_metrics:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce')
            values = values.dropna()
            
            if len(values) > 0:
                quality_results[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'count': len(values)
                }
                
                print(f"\n{metric}:")
                print(f"  平均值: {values.mean():.6f}")
                print(f"  標準差: {values.std():.6f}")
                print(f"  範圍: [{values.min():.6f}, {values.max():.6f}]")
                print(f"  中位數: {values.median():.6f}")
    
    return quality_results

def create_parameter_comparison_plot(df):
    """創建參數比較圖"""
    print("\n正在創建參數比較圖...")
    
    # 準備數據
    numeric_df = df.copy()
    for col in ['I_c', 'phi_0', 'f', 'T', 'r', 'C']:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    # 創建子圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('特定文件物理參數比較', fontsize=16, fontweight='bold')
    
    params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    param_names = ['臨界電流 I_c (A)', '相位 φ_0 (rad)', '頻率 f (Hz)', 
                   '溫度 T (K)', '電阻 r (Ω)', '電容 C (F)']
    
    for i, (param, name) in enumerate(zip(params, param_names)):
        ax = axes[i//3, i%3]
        
        if param in numeric_df.columns:
            values = numeric_df[param].dropna()
            if len(values) > 0:
                # 創建條形圖
                x_labels = [f"File {row['dataid']}" for _, row in numeric_df.iterrows() 
                           if pd.notna(row[param])]
                
                bars = ax.bar(range(len(values)), values, 
                             color=plt.cm.viridis(np.linspace(0, 1, len(values))))
                ax.set_xlabel('文件ID')
                ax.set_ylabel(name)
                ax.set_title(f'{param} 分佈')
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels([f"{TARGET_FILES[j]}" for j in range(len(values))], 
                                  rotation=45)
                
                # 添加數值標籤
                for j, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2e}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'無 {param} 數據', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{param} (無數據)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_metrics_plot(df):
    """創建品質指標圖"""
    print("正在創建品質指標圖...")
    
    # 準備數據
    quality_cols = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    quality_names = ['R²', '調整R²', 'RMSE', 'MAE']
    
    numeric_df = df.copy()
    for col in quality_cols:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('特定文件擬合品質指標比較', fontsize=16, fontweight='bold')
    
    for i, (col, name) in enumerate(zip(quality_cols, quality_names)):
        ax = axes[i//2, i%2]
        
        if col in numeric_df.columns:
            values = numeric_df[col].dropna()
            if len(values) > 0:
                bars = ax.bar(range(len(values)), values,
                             color=plt.cm.plasma(np.linspace(0, 1, len(values))))
                ax.set_xlabel('文件ID')
                ax.set_ylabel(name)
                ax.set_title(f'{name} 比較')
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels([f"{TARGET_FILES[j]}" for j in range(len(values))],
                                  rotation=45)
                
                # 添加數值標籤
                for j, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, f'無 {col} 數據', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{name} (無數據)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_summary_table(df):
    """創建詳細摘要表格"""
    print("正在創建詳細摘要表格...")
    
    # 選擇重要列
    important_cols = ['dataid', 'success', 'I_c', 'phi_0', 'f', 'T', 'r', 'C',
                     'r_squared', 'adj_r_squared', 'rmse', 'mae']
    
    summary_df = df[important_cols].copy()
    
    # 保存為CSV
    summary_df.to_csv(OUTPUT_DIR / 'detailed_summary.csv', index=False)
    
    # 創建格式化的表格圖像
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 準備表格數據
    table_data = []
    headers = ['文件ID', '成功', 'I_c (A)', 'φ_0 (rad)', 'f (Hz)', 'T (K)', 
               'r (Ω)', 'C (F)', 'R²', '調整R²', 'RMSE', 'MAE']
    
    for _, row in summary_df.iterrows():
        row_data = [
            row['dataid'],
            '是' if row['success'] else '否',
            f"{float(row['I_c']):.2e}" if pd.notna(row['I_c']) else 'N/A',
            f"{float(row['phi_0']):.4f}" if pd.notna(row['phi_0']) else 'N/A',
            f"{float(row['f']):.2e}" if pd.notna(row['f']) else 'N/A',
            f"{float(row['T']):.2e}" if pd.notna(row['T']) else 'N/A',
            f"{float(row['r']):.2e}" if pd.notna(row['r']) else 'N/A',
            f"{float(row['C']):.2e}" if pd.notna(row['C']) else 'N/A',
            f"{float(row['r_squared']):.4f}" if pd.notna(row['r_squared']) else 'N/A',
            f"{float(row['adj_r_squared']):.4f}" if pd.notna(row['adj_r_squared']) else 'N/A',
            f"{float(row['rmse']):.2e}" if pd.notna(row['rmse']) else 'N/A',
            f"{float(row['mae']):.2e}" if pd.notna(row['mae']) else 'N/A'
        ]
        table_data.append(row_data)
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.08]*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 設定標題
    ax.set_title('特定文件詳細參數摘要表', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(OUTPUT_DIR / 'detailed_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis_results(param_results, quality_results, df):
    """保存分析結果到JSON文件"""
    print("正在保存分析結果...")
    
    results = {
        'target_files': TARGET_FILES,
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'total_files_found': len(df),
        'parameter_analysis': param_results,
        'quality_analysis': quality_results,
        'file_details': []
    }
    
    # 添加每個文件的詳細信息
    for _, row in df.iterrows():
        file_detail = {
            'dataid': row['dataid'],
            'success': bool(row['success']),
            'parameters': {},
            'quality_metrics': {}
        }
        
        # 參數
        for param in ['I_c', 'phi_0', 'f', 'T', 'r', 'C']:
            if param in row and pd.notna(row[param]):
                file_detail['parameters'][param] = float(row[param])
        
        # 品質指標
        for metric in ['r_squared', 'adj_r_squared', 'rmse', 'mae']:
            if metric in row and pd.notna(row[metric]):
                file_detail['quality_metrics'][metric] = float(row[metric])
        
        results['file_details'].append(file_detail)
    
    with open(OUTPUT_DIR / 'analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def print_summary_statistics(df):
    """打印摘要統計"""
    print("\n" + "="*60)
    print("特定文件分析摘要統計")
    print("="*60)
    
    print(f"分析的文件數量: {len(df)}")
    print(f"成功分析的文件: {df['success'].sum()}")
    print(f"失敗的文件: {(~df['success']).sum()}")
    
    # 最佳擬合品質的文件
    if 'r_squared' in df.columns:
        best_r2_idx = df['r_squared'].idxmax()
        best_file = df.loc[best_r2_idx]
        print(f"\n最佳R²擬合: {best_file['dataid']} (R² = {best_file['r_squared']:.6f})")
    
    # 參數範圍摘要
    print(f"\n參數範圍摘要:")
    for param in ['I_c', 'phi_0', 'f', 'T', 'r', 'C']:
        if param in df.columns:
            values = pd.to_numeric(df[param], errors='coerce').dropna()
            if len(values) > 0:
                print(f"  {param}: [{values.min():.2e}, {values.max():.2e}]")

def main():
    """主函數"""
    print("開始分析特定文件的參數和統計數據...")
    print(f"目標文件: {TARGET_FILES}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    
    # 載入數據
    df = load_analysis_data()
    
    # 提取目標數據
    target_df = extract_target_data(df)
    
    if target_df.empty:
        print("錯誤: 未找到任何目標文件數據")
        return
    
    print(f"\n成功提取 {len(target_df)} 個目標文件的數據")
    
    # 分析參數
    param_results = analyze_parameters(target_df)
    
    # 分析品質指標
    quality_results = analyze_quality_metrics(target_df)
    
    # 創建可視化
    create_parameter_comparison_plot(target_df)
    create_quality_metrics_plot(target_df)
    create_detailed_summary_table(target_df)
    
    # 保存結果
    save_analysis_results(param_results, quality_results, target_df)
    
    # 打印摘要統計
    print_summary_statistics(target_df)
    
    print(f"\n分析完成！結果已保存到: {OUTPUT_DIR}")
    print("生成的文件:")
    print("  - parameter_comparison.png: 參數比較圖")
    print("  - quality_metrics.png: 品質指標圖")
    print("  - detailed_summary_table.png: 詳細摘要表格")
    print("  - detailed_summary.csv: 詳細摘要CSV")
    print("  - analysis_results.json: 完整分析結果JSON")

if __name__ == "__main__":
    main()
