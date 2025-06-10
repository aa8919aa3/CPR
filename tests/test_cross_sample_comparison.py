#!/usr/bin/env python3
"""
跨樣本比較分析：003-2 vs 005-1
比較兩個樣本在相同磁場條件下的CPR特性差異
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from pathlib import Path
from datetime import datetime

# 設置matplotlib的字體以支援中文
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False

# 樣本配置
SAMPLE_003_2_30MT = {
    'sample_name': '003-2@30mT',
    'magnetic_field': 30,  # mT
    'angles': [0, 45, 90, 135, 180, 225, 270, 315, 360],  # degrees
    'file_ids': [72, 73, 74, 75, 76, 77, 78, 79, 80]
}

SAMPLE_003_2_60MT = {
    'sample_name': '003-2@60mT',
    'magnetic_field': 60,  # mT
    'angles': [0, 40, 80, 120, 160, 200, 240, 280, 320, 360],  # degrees
    'file_ids': [58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
}

SAMPLE_005_1_30MT = {
    'sample_name': '005-1@30mT',
    'magnetic_field': 30,  # mT
    'angles': [0, 45, 90, 135, 180, 225, 270, 315],  # degrees
    'file_ids': [156, 145, 94, 164, 168, 170, 172, 147]
}

SAMPLE_005_1_60MT = {
    'sample_name': '005-1@60mT',
    'magnetic_field': 60,  # mT
    'angles': [0, 45, 90, 135, 180, 225, 270, 315],  # degrees
    'file_ids': [197, 196, 178, 180, 186, 185, 188, 191]
}

def load_analysis_data(csv_path):
    """讀取分析結果CSV文件"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ 成功讀取 {len(df)} 行數據")
        return df
    except Exception as e:
        print(f"❌ 讀取CSV文件失敗: {e}")
        return None

def extract_sample_data(df, sample_config):
    """提取指定樣本的數據"""
    file_ids = sample_config['file_ids']
    angles = sample_config['angles']
    sample_name = sample_config['sample_name']
    
    print(f"\n🔍 搜尋 {sample_name} 數據...")
    
    # 搜尋匹配的記錄
    matches = []
    found_ids = []
    
    for i, file_id in enumerate(file_ids):
        angle = angles[i] if i < len(angles) else angles[0]  # 防止索引越界
        
        # 嘗試多種匹配模式
        patterns = [f"{file_id}Ic", f"{file_id}Ic+", f"{file_id}Ic-"]
        
        found = False
        for pattern in patterns:
            mask = df['dataid'].str.contains(f"^{pattern}$", na=False, regex=True)
            if mask.any():
                match_data = df[mask].iloc[0].copy()
                match_data['angle'] = angle
                match_data['file_id'] = file_id
                matches.append(match_data)
                found_ids.append(file_id)
                found = True
                break
        
        if not found:
            print(f"   ❌ 未找到 ID {file_id} (角度 {angle}°)")
    
    if matches:
        result_df = pd.DataFrame(matches)
        print(f"✓ 成功提取 {len(result_df)} 個文件的數據")
        return result_df
    else:
        print("❌ 未找到任何匹配的數據")
        return None

def calculate_sample_statistics(data, sample_name):
    """計算樣本統計數據"""
    params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C', 'r_squared']
    
    stats = {}
    for param in params:
        if param in data.columns:
            values = pd.to_numeric(data[param], errors='coerce').dropna()
            
            if len(values) > 0:
                stats[param] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'count': len(values)
                }
    
    return stats

def create_cross_sample_comparison_table(data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60, output_dir):
    """創建跨樣本比較表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 計算所有樣本的統計數據
    stats_003_2_30 = calculate_sample_statistics(data_003_2_30, '003-2@30mT')
    stats_003_2_60 = calculate_sample_statistics(data_003_2_60, '003-2@60mT')
    stats_005_1_30 = calculate_sample_statistics(data_005_1_30, '005-1@30mT')
    stats_005_1_60 = calculate_sample_statistics(data_005_1_60, '005-1@60mT')
    
    params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C', 'r_squared']
    
    comparison_data = []
    
    for param in params:
        if all(param in stats for stats in [stats_003_2_30, stats_003_2_60, stats_005_1_30, stats_005_1_60]):
            comparison_data.append({
                '參數': param,
                '003-2@30mT_平均值': stats_003_2_30[param]['mean'],
                '003-2@30mT_標準差': stats_003_2_30[param]['std'],
                '003-2@60mT_平均值': stats_003_2_60[param]['mean'],
                '003-2@60mT_標準差': stats_003_2_60[param]['std'],
                '005-1@30mT_平均值': stats_005_1_30[param]['mean'],
                '005-1@30mT_標準差': stats_005_1_30[param]['std'],
                '005-1@60mT_平均值': stats_005_1_60[param]['mean'],
                '005-1@60mT_標準差': stats_005_1_60[param]['std'],
                '003-2_磁場比值(60/30)': stats_003_2_60[param]['mean'] / stats_003_2_30[param]['mean'] if stats_003_2_30[param]['mean'] != 0 else np.nan,
                '005-1_磁場比值(60/30)': stats_005_1_60[param]['mean'] / stats_005_1_30[param]['mean'] if stats_005_1_30[param]['mean'] != 0 else np.nan,
                '30mT_樣本比值(005-1/003-2)': stats_005_1_30[param]['mean'] / stats_003_2_30[param]['mean'] if stats_003_2_30[param]['mean'] != 0 else np.nan,
                '60mT_樣本比值(005-1/003-2)': stats_005_1_60[param]['mean'] / stats_003_2_60[param]['mean'] if stats_003_2_60[param]['mean'] != 0 else np.nan,
            })
    
    # 創建DataFrame並保存
    comparison_df = pd.DataFrame(comparison_data)
    output_file = output_dir / 'cross_sample_comparison.csv'
    comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📊 跨樣本比較統計:")
    print(comparison_df.to_string(index=False))
    print(f"\n✓ 跨樣本比較表已保存: {output_file}")
    
    return comparison_df

def create_cross_sample_plots(data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60, output_dir):
    """創建跨樣本比較圖表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建大型比較圖
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('跨樣本CPR比較分析：003-2 vs 005-1', fontsize=18, fontweight='bold')
    
    params = [
        ('I_c', '臨界電流 I_c (A)', axes[0, 0]),
        ('phi_0', '相位偏移 φ₀ (rad)', axes[0, 1]),
        ('f', '頻率 f (Hz)', axes[0, 2]),
        ('r', '電阻 r (Ω)', axes[1, 0]),
        ('C', '電容 C (F)', axes[1, 1]),
        ('r_squared', 'R² 擬合質量', axes[1, 2]),
        ('I_c', '臨界電流對數尺度', axes[2, 0]),
        ('f', '頻率對數尺度', axes[2, 1]),
        ('C', '電容對數尺度', axes[2, 2])
    ]
    
    datasets = [
        (data_003_2_30, '003-2@30mT', 'blue', 'o'),
        (data_003_2_60, '003-2@60mT', 'red', 's'),
        (data_005_1_30, '005-1@30mT', 'green', '^'),
        (data_005_1_60, '005-1@60mT', 'orange', 'v')
    ]
    
    for i, (param, ylabel, ax) in enumerate(params):
        # 對於重複的參數，在最後一行使用對數尺度
        use_log = i >= 6
        
        for data, label, color, marker in datasets:
            if param in data.columns:
                angles = data['angle'].values
                values = pd.to_numeric(data[param], errors='coerce')
                
                # 繪製散點圖
                if use_log:
                    valid_mask = (values > 0) & ~np.isnan(values)
                    if np.any(valid_mask):
                        ax.scatter(angles[valid_mask], values[valid_mask], 
                                 s=80, alpha=0.7, c=color, 
                                 edgecolors='black', label=label, marker=marker)
                        ax.set_yscale('log')
                else:
                    ax.scatter(angles, values, s=80, alpha=0.7, c=color, 
                             edgecolors='black', label=label, marker=marker)
        
        ax.set_xlabel('角度 (度)')
        ax.set_ylabel(ylabel if not use_log else f'{ylabel} (對數尺度)')
        ax.set_title(f'{param} - 跨樣本比較{"（對數尺度）" if use_log else ""}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 370)
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    plt.tight_layout()
    
    # 保存圖表
    output_file = output_dir / 'cross_sample_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 跨樣本比較圖表已保存: {output_file}")
    plt.close()

def create_parameter_distribution_plots(data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60, output_dir):
    """創建參數分佈比較圖"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    params = ['I_c', 'r_squared', 'f', 'C']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('參數分佈比較', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    datasets = [
        (data_003_2_30, '003-2@30mT', 'blue'),
        (data_003_2_60, '003-2@60mT', 'red'),
        (data_005_1_30, '005-1@30mT', 'green'),
        (data_005_1_60, '005-1@60mT', 'orange')
    ]
    
    for i, param in enumerate(params):
        ax = axes_flat[i]
        
        for data, label, color in datasets:
            if param in data.columns:
                values = pd.to_numeric(data[param], errors='coerce').dropna()
                if len(values) > 0:
                    # 創建直方圖
                    ax.hist(values, bins=5, alpha=0.6, label=label, color=color, density=True)
        
        ax.set_xlabel(param)
        ax.set_ylabel('密度')
        ax.set_title(f'{param} 分佈')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存圖表
    output_file = output_dir / 'parameter_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 參數分佈圖表已保存: {output_file}")
    plt.close()

def generate_cross_sample_report(comparison_df, output_dir):
    """生成跨樣本分析報告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'cross_sample_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("跨樣本CPR分析報告：003-2 vs 005-1\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("一、實驗設計對比\n")
        f.write("-" * 30 + "\n")
        f.write("樣本 003-2:\n")
        f.write("  30mT: 9個角度點 (0°-360°)\n")
        f.write("  60mT: 10個角度點 (0°-360°)\n\n")
        f.write("樣本 005-1:\n")
        f.write("  30mT: 8個角度點 (0°-315°)\n")
        f.write("  60mT: 8個角度點 (0°-315°)\n\n")
        
        f.write("二、關鍵參數比較\n")
        f.write("-" * 30 + "\n")
        
        # 臨界電流分析
        i_c_row = comparison_df[comparison_df['參數'] == 'I_c'].iloc[0]
        f.write("臨界電流 (I_c):\n")
        f.write(f"  003-2@30mT: {i_c_row['003-2@30mT_平均值']:.3e} A\n")
        f.write(f"  003-2@60mT: {i_c_row['003-2@60mT_平均值']:.3e} A\n")
        f.write(f"  005-1@30mT: {i_c_row['005-1@30mT_平均值']:.3e} A\n")
        f.write(f"  005-1@60mT: {i_c_row['005-1@60mT_平均值']:.3e} A\n")
        f.write(f"  003-2 磁場效應: {i_c_row['003-2_磁場比值(60/30)']:.3f}\n")
        f.write(f"  005-1 磁場效應: {i_c_row['005-1_磁場比值(60/30)']:.3f}\n\n")
        
        # 擬合質量分析
        r2_row = comparison_df[comparison_df['參數'] == 'r_squared'].iloc[0]
        f.write("擬合質量 (R²):\n")
        f.write(f"  003-2@30mT: {r2_row['003-2@30mT_平均值']:.3f}\n")
        f.write(f"  003-2@60mT: {r2_row['003-2@60mT_平均值']:.3f}\n")
        f.write(f"  005-1@30mT: {r2_row['005-1@30mT_平均值']:.3f}\n")
        f.write(f"  005-1@60mT: {r2_row['005-1@60mT_平均值']:.3f}\n\n")
        
        f.write("三、主要發現\n")
        f.write("-" * 30 + "\n")
        
        # 分析磁場效應差異
        f.write("磁場效應對比:\n")
        for _, row in comparison_df.iterrows():
            param = row['參數']
            ratio_003_2 = row['003-2_磁場比值(60/30)']
            ratio_005_1 = row['005-1_磁場比值(60/30)']
            
            if not np.isnan(ratio_003_2) and not np.isnan(ratio_005_1):
                f.write(f"  {param}: 003-2={ratio_003_2:.3f}, 005-1={ratio_005_1:.3f}\n")
        
        f.write("\n樣本間差異:\n")
        f.write(f"  30mT條件下，005-1/003-2 臨界電流比值: {i_c_row['30mT_樣本比值(005-1/003-2)']:.3f}\n")
        f.write(f"  60mT條件下，005-1/003-2 臨界電流比值: {i_c_row['60mT_樣本比值(005-1/003-2)']:.3f}\n")
        
        f.write("\n四、結論\n")
        f.write("-" * 30 + "\n")
        
        if i_c_row['005-1_磁場比值(60/30)'] < i_c_row['003-2_磁場比值(60/30)']:
            f.write("• 005-1樣本對磁場變化更敏感，在60mT下臨界電流降低更明顯\n")
        else:
            f.write("• 003-2樣本對磁場變化更敏感\n")
        
        if r2_row['005-1@30mT_平均值'] > r2_row['003-2@30mT_平均值']:
            f.write("• 005-1樣本在30mT下擬合質量更佳\n")
        else:
            f.write("• 003-2樣本在30mT下擬合質量更佳\n")
        
        f.write("\n報告結束\n")
    
    print(f"✓ 跨樣本分析報告已保存: {report_file}")

def main():
    """主函數"""
    print("🚀 跨樣本CPR比較分析：003-2 vs 005-1")
    print("=" * 80)
    
    # 設置路徑
    project_root = Path(__file__).parent.parent
    analysis_csv = project_root / "output" / "full_analysis" / "images" / "analysis_summary.csv"
    output_dir = project_root / "output" / "cross_sample_analysis"
    
    # 檢查輸入文件
    if not analysis_csv.exists():
        print(f"❌ 分析結果文件不存在: {analysis_csv}")
        return 1
    
    # 讀取數據
    df = load_analysis_data(analysis_csv)
    if df is None:
        return 1
    
    # 提取所有樣本數據
    print("\n" + "="*80)
    data_003_2_30 = extract_sample_data(df, SAMPLE_003_2_30MT)
    data_003_2_60 = extract_sample_data(df, SAMPLE_003_2_60MT)
    data_005_1_30 = extract_sample_data(df, SAMPLE_005_1_30MT)
    data_005_1_60 = extract_sample_data(df, SAMPLE_005_1_60MT)
    
    # 檢查數據完整性
    if any(data is None for data in [data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60]):
        print("❌ 部分樣本數據提取失敗")
        return 1
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成跨樣本比較分析
    print("\n📊 生成跨樣本比較分析...")
    comparison_df = create_cross_sample_comparison_table(
        data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60, output_dir)
    
    # 創建比較圖表
    print("\n📈 生成比較圖表...")
    create_cross_sample_plots(data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60, output_dir)
    create_parameter_distribution_plots(data_003_2_30, data_003_2_60, data_005_1_30, data_005_1_60, output_dir)
    
    # 生成分析報告
    print("\n📄 生成跨樣本分析報告...")
    generate_cross_sample_report(comparison_df, output_dir)
    
    print(f"\n🎉 跨樣本分析完成！結果保存在: {output_dir}")
    print("\n生成的文件:")
    for file in output_dir.glob("*"):
        print(f"  📄 {file.name}")
    
    return 0

if __name__ == "__main__":
    exit(main())
