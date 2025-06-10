#!/usr/bin/env python3
"""
樣本 005-1 CPR 分析
分析樣本005-1在30mT和60mT磁場條件下的CPR（Current-Phase Relation）數據
基於 analysis_summary.csv 中已處理的數據進行統計分析和可視化
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

# 005-1樣本的文件ID對應
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
    print(f"   文件ID: {file_ids}")
    print(f"   角度: {angles}")
    
    # 搜尋匹配的記錄
    matches = []
    found_ids = []
    
    for i, file_id in enumerate(file_ids):
        angle = angles[i]
        
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
                print(f"   ✓ 找到 {pattern} (角度 {angle}°)")
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

def analyze_sample_parameters(data, sample_name):
    """分析樣本參數"""
    print(f"\n📊 {sample_name} 參數分析:")
    print("=" * 60)
    
    # 基本統計
    params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C', 'r_squared']
    
    stats = {}
    for param in params:
        if param in data.columns:
            values = pd.to_numeric(data[param], errors='coerce')
            valid_values = values.dropna()
            
            if len(valid_values) > 0:
                stats[param] = {
                    'mean': valid_values.mean(),
                    'std': valid_values.std(),
                    'min': valid_values.min(),
                    'max': valid_values.max(),
                    'median': valid_values.median(),
                    'count': len(valid_values)
                }
                
                print(f"\n{param}:")
                print(f"  平均值: {stats[param]['mean']:.6e}")
                print(f"  標準差: {stats[param]['std']:.6e}")
                print(f"  範圍: {stats[param]['min']:.6e} ~ {stats[param]['max']:.6e}")
                print(f"  中位數: {stats[param]['median']:.6e}")
                print(f"  有效數據點: {stats[param]['count']}")
    
    return stats

def create_angular_plots(data, sample_name, output_dir):
    """創建角度相關的圖表"""
    # 確保輸出目錄存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建多子圖布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{sample_name} 參數角度依賴性分析', fontsize=16, fontweight='bold')
    
    # 準備數據
    angles = data['angle'].values
    
    # 參數列表和對應的軸
    params = [
        ('I_c', '臨界電流 I_c (A)', axes[0, 0]),
        ('phi_0', '相位偏移 φ₀ (rad)', axes[0, 1]),
        ('f', '頻率 f (Hz)', axes[0, 2]),
        ('r', '電阻 r (Ω)', axes[1, 0]),
        ('C', '電容 C (F)', axes[1, 1]),
        ('r_squared', 'R² 擬合質量', axes[1, 2])
    ]
    
    # 繪製每個參數
    for param, ylabel, ax in params:
        if param in data.columns:
            values = pd.to_numeric(data[param], errors='coerce')
            
            # 繪製散點圖
            ax.scatter(angles, values, s=80, alpha=0.7, c='blue', edgecolors='black')
            
            # 嘗試擬合曲線（如果數據點足夠）
            if len(values.dropna()) >= 3:
                try:
                    # 使用sin/cos基函數進行擬合（考慮到角度週期性）
                    angles_rad = np.deg2rad(angles)
                    A = np.column_stack([
                        np.ones(len(angles_rad)),
                        np.sin(angles_rad),
                        np.cos(angles_rad),
                        np.sin(2*angles_rad),
                        np.cos(2*angles_rad)
                    ])
                    
                    # 移除NaN值
                    valid_mask = ~np.isnan(values)
                    if np.sum(valid_mask) >= 3:
                        A_valid = A[valid_mask]
                        values_valid = values[valid_mask]
                        
                        # 最小二乘法擬合
                        coeffs, residuals, rank, s = np.linalg.lstsq(A_valid, values_valid, rcond=None)
                        
                        # 繪製擬合曲線
                        angles_smooth = np.linspace(0, 360, 100)
                        angles_smooth_rad = np.deg2rad(angles_smooth)
                        A_smooth = np.column_stack([
                            np.ones(len(angles_smooth_rad)),
                            np.sin(angles_smooth_rad),
                            np.cos(angles_smooth_rad),
                            np.sin(2*angles_smooth_rad),
                            np.cos(2*angles_smooth_rad)
                        ])
                        fitted_values = A_smooth @ coeffs
                        
                        ax.plot(angles_smooth, fitted_values, 'r-', alpha=0.7, linewidth=2, label='擬合曲線')
                        
                except Exception as e:
                    print(f"⚠️ {param} 擬合失敗: {e}")
            
            ax.set_xlabel('角度 (度)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{param} vs 角度')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-10, 370)
            
            # 設置x軸刻度
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    plt.tight_layout()
    
    # 保存圖表
    output_file = output_dir / f'{sample_name.replace("@", "_")}_angular_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 角度分析圖表已保存: {output_file}")
    plt.close()

def create_comparison_table(data_30mt, data_60mt, output_dir):
    """創建30mT和60mT條件的比較表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 準備比較數據
    params = ['I_c', 'phi_0', 'f', 'T', 'r', 'C', 'r_squared']
    
    comparison_data = []
    
    for param in params:
        if param in data_30mt.columns and param in data_60mt.columns:
            # 30mT數據
            values_30 = pd.to_numeric(data_30mt[param], errors='coerce').dropna()
            # 60mT數據
            values_60 = pd.to_numeric(data_60mt[param], errors='coerce').dropna()
            
            if len(values_30) > 0 and len(values_60) > 0:
                comparison_data.append({
                    '參數': param,
                    '30mT_平均值': values_30.mean(),
                    '30mT_標準差': values_30.std(),
                    '60mT_平均值': values_60.mean(),
                    '60mT_標準差': values_60.std(),
                    '比值_60mT/30mT': values_60.mean() / values_30.mean() if values_30.mean() != 0 else np.nan,
                    '30mT_數據點': len(values_30),
                    '60mT_數據點': len(values_60)
                })
    
    # 創建DataFrame並保存
    comparison_df = pd.DataFrame(comparison_data)
    output_file = output_dir / 'sample_005_1_field_comparison.csv'
    comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n📊 磁場比較統計:")
    print(comparison_df.to_string(index=False))
    print(f"\n✓ 比較表已保存: {output_file}")
    
    return comparison_df

def analyze_transparency_parameter(data_30mt, data_60mt, output_dir):
    """專門分析穿透率T參數"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔬 穿透率T參數專項分析:")
    print("=" * 60)
    
    # 提取穿透率數據
    T_30mt = pd.to_numeric(data_30mt['T'], errors='coerce').dropna()
    T_60mt = pd.to_numeric(data_60mt['T'], errors='coerce').dropna()
    angles_30 = data_30mt.loc[T_30mt.index, 'angle'].values
    angles_60 = data_60mt.loc[T_60mt.index, 'angle'].values
    
    # 統計分析
    print(f"\n30mT條件下穿透率T:")
    print(f"  平均值: {T_30mt.mean():.4f} ({T_30mt.mean()*100:.2f}%)")
    print(f"  標準差: {T_30mt.std():.4f}")
    print(f"  範圍: {T_30mt.min():.4f} ~ {T_30mt.max():.4f}")
    print(f"  變化幅度: {(T_30mt.max() - T_30mt.min()):.4f}")
    
    print(f"\n60mT條件下穿透率T:")
    print(f"  平均值: {T_60mt.mean():.4f} ({T_60mt.mean()*100:.2f}%)")
    print(f"  標準差: {T_60mt.std():.4f}")
    print(f"  範圍: {T_60mt.min():.4f} ~ {T_60mt.max():.4f}")
    print(f"  變化幅度: {(T_60mt.max() - T_60mt.min()):.4f}")
    
    # 計算變化率
    T_change_percent = ((T_60mt.mean() - T_30mt.mean()) / T_30mt.mean()) * 100
    print(f"\n磁場效應:")
    print(f"  60mT相對30mT變化: {T_change_percent:+.2f}%")
    print(f"  比值 (60mT/30mT): {T_60mt.mean()/T_30mt.mean():.3f}")
    
    # 角度依賴性分析
    print(f"\n角度依賴性分析:")
    if len(T_30mt) >= 3:
        # 計算角度相關性
        correlation_30 = np.corrcoef(angles_30, T_30mt)[0,1] if len(angles_30) == len(T_30mt) else np.nan
        print(f"  30mT: T與角度相關係數 = {correlation_30:.3f}")
    
    if len(T_60mt) >= 3:
        correlation_60 = np.corrcoef(angles_60, T_60mt)[0,1] if len(angles_60) == len(T_60mt) else np.nan
        print(f"  60mT: T與角度相關係數 = {correlation_60:.3f}")
    
    # 創建穿透率專門的可視化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('樣本 005-1 穿透率T參數深度分析', fontsize=16, fontweight='bold')
    
    # 1. 角度依賴性比較
    ax1.scatter(angles_30, T_30mt*100, s=100, alpha=0.7, c='blue', 
               edgecolors='black', label='30mT', marker='o')
    ax1.scatter(angles_60, T_60mt*100, s=100, alpha=0.7, c='red', 
               edgecolors='black', label='60mT', marker='s')
    ax1.set_xlabel('角度 (度)')
    ax1.set_ylabel('穿透率T (%)')
    ax1.set_title('穿透率T的角度依賴性')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 370)
    ax1.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    # 2. 統計分佈比較
    ax2.hist(T_30mt*100, bins=8, alpha=0.6, color='blue', label='30mT', density=True)
    ax2.hist(T_60mt*100, bins=8, alpha=0.6, color='red', label='60mT', density=True)
    ax2.axvline(T_30mt.mean()*100, color='blue', linestyle='--', linewidth=2, label=f'30mT平均 ({T_30mt.mean()*100:.1f}%)')
    ax2.axvline(T_60mt.mean()*100, color='red', linestyle='--', linewidth=2, label=f'60mT平均 ({T_60mt.mean()*100:.1f}%)')
    ax2.set_xlabel('穿透率T (%)')
    ax2.set_ylabel('機率密度')
    ax2.set_title('穿透率T分佈比較')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 磁場效應分析
    if len(T_30mt) == len(T_60mt):  # 如果數據點對應
        T_ratio = T_60mt.values / T_30mt.values
        ax3.scatter(angles_30, T_ratio, s=100, alpha=0.7, c='purple', edgecolors='black')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='無變化線')
        ax3.axhline(y=T_ratio.mean(), color='purple', linestyle='-', linewidth=2, 
                   label=f'平均比值 ({T_ratio.mean():.3f})')
        ax3.set_xlabel('角度 (度)')
        ax3.set_ylabel('T比值 (60mT/30mT)')
        ax3.set_title('穿透率磁場效應')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-10, 370)
        ax3.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    else:
        ax3.text(0.5, 0.5, '數據點不匹配\n無法計算比值', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('穿透率磁場效應 (數據不匹配)')
    
    # 4. 穿透率與臨界電流關係
    I_c_30 = pd.to_numeric(data_30mt['I_c'], errors='coerce').dropna()
    I_c_60 = pd.to_numeric(data_60mt['I_c'], errors='coerce').dropna()
    
    if len(T_30mt) == len(I_c_30):
        ax4.scatter(T_30mt*100, I_c_30*1e6, s=100, alpha=0.7, c='blue', 
                   edgecolors='black', label='30mT')
    if len(T_60mt) == len(I_c_60):
        ax4.scatter(T_60mt*100, I_c_60*1e6, s=100, alpha=0.7, c='red', 
                   edgecolors='black', label='60mT')
    
    ax4.set_xlabel('穿透率T (%)')
    ax4.set_ylabel('臨界電流 I_c (μA)')
    ax4.set_title('穿透率T與臨界電流關係')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存穿透率分析圖
    output_file = output_dir / 'sample_005_1_transparency_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 穿透率分析圖表已保存: {output_file}")
    plt.close()
    
    # 保存穿透率分析數據
    transparency_data = {
        '磁場條件': ['30mT', '60mT'],
        '平均穿透率': [T_30mt.mean(), T_60mt.mean()],
        '標準差': [T_30mt.std(), T_60mt.std()],
        '最小值': [T_30mt.min(), T_60mt.min()],
        '最大值': [T_30mt.max(), T_60mt.max()],
        '數據點數': [len(T_30mt), len(T_60mt)]
    }
    
    transparency_df = pd.DataFrame(transparency_data)
    csv_file = output_dir / 'sample_005_1_transparency_comparison.csv'
    transparency_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ 穿透率比較數據已保存: {csv_file}")
    
    return {
        'T_30mt_mean': T_30mt.mean(),
        'T_60mt_mean': T_60mt.mean(),
        'T_change_percent': T_change_percent,
        'T_30mt_std': T_30mt.std(),
        'T_60mt_std': T_60mt.std()
    }

def create_field_comparison_plot(data_30mt, data_60mt, output_dir):
    """創建磁場比較圖表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建比較圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('樣本 005-1: 30mT vs 60mT 磁場比較', fontsize=16, fontweight='bold')
    
    params = [
        ('I_c', '臨界電流 I_c (A)', axes[0, 0]),
        ('T', '穿透率 T', axes[0, 1]),  # 將T參數提前到更顯著位置
        ('phi_0', '相位偏移 φ₀ (rad)', axes[0, 2]),
        ('f', '頻率 f (Hz)', axes[1, 0]),
        ('r', '電阻 r (Ω)', axes[1, 1]),
        ('r_squared', 'R² 擬合質量', axes[1, 2])
    ]
    
    for param, ylabel, ax in params:
        if param in data_30mt.columns and param in data_60mt.columns:
            angles_30 = data_30mt['angle'].values
            values_30 = pd.to_numeric(data_30mt[param], errors='coerce')
            
            angles_60 = data_60mt['angle'].values
            values_60 = pd.to_numeric(data_60mt[param], errors='coerce')
            
            # 對穿透率T進行特殊處理 - 轉換為百分比
            if param == 'T':
                values_30 = values_30 * 100
                values_60 = values_60 * 100
                ylabel = '穿透率 T (%)'
            
            # 繪製兩個磁場條件的數據
            ax.scatter(angles_30, values_30, s=80, alpha=0.7, c='blue', 
                      edgecolors='black', label='30mT', marker='o')
            ax.scatter(angles_60, values_60, s=80, alpha=0.7, c='red', 
                      edgecolors='black', label='60mT', marker='s')
            
            # 對穿透率添加額外的分析線
            if param == 'T':
                # 添加平均線
                ax.axhline(y=np.nanmean(values_30), color='blue', linestyle='--', alpha=0.5, 
                          label=f'30mT平均 ({np.nanmean(values_30):.1f}%)')
                ax.axhline(y=np.nanmean(values_60), color='red', linestyle='--', alpha=0.5,
                          label=f'60mT平均 ({np.nanmean(values_60):.1f}%)')
            
            ax.set_xlabel('角度 (度)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{param} - 磁場比較')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-10, 370)
            ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    plt.tight_layout()
    
    # 保存圖表
    output_file = output_dir / 'sample_005_1_field_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 磁場比較圖表已保存: {output_file}")
    plt.close()

def generate_summary_report(stats_30mt, stats_60mt, comparison_df, transparency_results, output_dir):
    """生成總結報告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'sample_005_1_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("樣本 005-1 CPR 分析報告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("一、實驗條件\n")
        f.write("-" * 20 + "\n")
        f.write("樣本編號: 005-1\n")
        f.write("磁場條件: 30mT, 60mT\n")
        f.write("測量角度: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°\n")
        f.write("數據來源: analysis_summary.csv\n\n")
        
        f.write("二、30mT 磁場條件分析\n")
        f.write("-" * 20 + "\n")
        for param, stat in stats_30mt.items():
            f.write(f"{param}:\n")
            if param == 'T':
                f.write(f"  平均值: {stat['mean']:.4f} ({stat['mean']*100:.2f}%)\n")
                f.write(f"  標準差: {stat['std']:.4f}\n")
                f.write(f"  範圍: {stat['min']:.4f} ~ {stat['max']:.4f}\n")
            else:
                f.write(f"  平均值: {stat['mean']:.6e}\n")
                f.write(f"  標準差: {stat['std']:.6e}\n")
                f.write(f"  範圍: {stat['min']:.6e} ~ {stat['max']:.6e}\n")
            f.write(f"  中位數: {stat['median']:.6e}\n")
            f.write(f"  有效數據點: {stat['count']}\n\n")
        
        f.write("三、60mT 磁場條件分析\n")
        f.write("-" * 20 + "\n")
        for param, stat in stats_60mt.items():
            f.write(f"{param}:\n")
            if param == 'T':
                f.write(f"  平均值: {stat['mean']:.4f} ({stat['mean']*100:.2f}%)\n")
                f.write(f"  標準差: {stat['std']:.4f}\n")
                f.write(f"  範圍: {stat['min']:.4f} ~ {stat['max']:.4f}\n")
            else:
                f.write(f"  平均值: {stat['mean']:.6e}\n")
                f.write(f"  標準差: {stat['std']:.6e}\n")
                f.write(f"  範圍: {stat['min']:.6e} ~ {stat['max']:.6e}\n")
            f.write(f"  中位數: {stat['median']:.6e}\n")
            f.write(f"  數據點數: {stat['count']}\n\n")
        
        f.write("四、穿透率T專項分析\n")
        f.write("-" * 20 + "\n")
        f.write(f"30mT條件:\n")
        f.write(f"  平均穿透率: {transparency_results['T_30mt_mean']:.4f} ({transparency_results['T_30mt_mean']*100:.2f}%)\n")
        f.write(f"  標準差: {transparency_results['T_30mt_std']:.4f}\n\n")
        f.write(f"60mT條件:\n")
        f.write(f"  平均穿透率: {transparency_results['T_60mt_mean']:.4f} ({transparency_results['T_60mt_mean']*100:.2f}%)\n")
        f.write(f"  標準差: {transparency_results['T_60mt_std']:.4f}\n\n")
        f.write(f"磁場效應:\n")
        f.write(f"  相對變化: {transparency_results['T_change_percent']:+.2f}%\n")
        f.write(f"  比值 (60mT/30mT): {transparency_results['T_60mt_mean']/transparency_results['T_30mt_mean']:.3f}\n\n")
        
        f.write("五、磁場比較分析\n")
        f.write("-" * 20 + "\n")
        for _, row in comparison_df.iterrows():
            param = row['參數']
            ratio = row['比值_60mT/30mT']
            f.write(f"{param}: 60mT/30mT = {ratio:.3f}\n")
        
        f.write("\n六、主要發現\n")
        f.write("-" * 20 + "\n")
        
        # 分析臨界電流變化
        if 'I_c' in comparison_df['參數'].values:
            i_c_ratio = comparison_df[comparison_df['參數'] == 'I_c']['比值_60mT/30mT'].iloc[0]
            if i_c_ratio > 1.1:
                f.write(f"• 臨界電流在60mT下增加了 {(i_c_ratio-1)*100:.1f}%\n")
            elif i_c_ratio < 0.9:
                f.write(f"• 臨界電流在60mT下減少了 {(1-i_c_ratio)*100:.1f}%\n")
            else:
                f.write("• 臨界電流在不同磁場下變化較小\n")
        
        # 分析穿透率變化
        T_change = transparency_results['T_change_percent']
        if abs(T_change) > 5:
            direction = "增加" if T_change > 0 else "減少"
            f.write(f"• 穿透率在60mT下{direction}了 {abs(T_change):.1f}%\n")
        else:
            f.write("• 穿透率在不同磁場下變化較小\n")
        
        # 分析擬合質量
        if 'r_squared' in comparison_df['參數'].values:
            r2_30 = stats_30mt.get('r_squared', {}).get('mean', 0)
            r2_60 = stats_60mt.get('r_squared', {}).get('mean', 0)
            f.write(f"• 30mT條件下平均R² = {r2_30:.3f}\n")
            f.write(f"• 60mT條件下平均R² = {r2_60:.3f}\n")
        
        # 物理意義解釋
        f.write("\n七、物理意義解釋\n")
        f.write("-" * 20 + "\n")
        f.write("穿透率T反映了約瑟夫森結的透明度，影響超電流的傳輸效率：\n")
        f.write(f"• T = 0: 完全不透明（隧道結）\n")
        f.write(f"• T = 1: 完全透明（彈道傳輸）\n")
        f.write(f"• 實測值: 30mT時平均{transparency_results['T_30mt_mean']*100:.1f}%, 60mT時平均{transparency_results['T_60mt_mean']*100:.1f}%\n")
        
        if abs(T_change) > 1:
            f.write(f"• 磁場從30mT增加到60mT，穿透率變化{T_change:+.1f}%，\n")
            f.write(f"  這可能反映了磁通量子對結特性的調製效應。\n")
        
        f.write("\n報告結束\n")
    
    print(f"✓ 分析報告已保存: {report_file}")

def main():
    """主函數"""
    print("🚀 樣本 005-1 CPR 分析")
    print("=" * 60)
    
    # 設置路徑
    project_root = Path(__file__).parent.parent
    analysis_csv = project_root / "output" / "full_analysis" / "images" / "analysis_summary.csv"
    output_dir = project_root / "output" / "sample_005_1_analysis"
    
    # 檢查輸入文件
    if not analysis_csv.exists():
        print(f"❌ 分析結果文件不存在: {analysis_csv}")
        return 1
    
    # 讀取數據
    df = load_analysis_data(analysis_csv)
    if df is None:
        return 1
    
    # 提取30mT數據
    print("\n" + "="*60)
    data_30mt = extract_sample_data(df, SAMPLE_005_1_30MT)
    if data_30mt is None:
        print("❌ 30mT數據提取失敗")
        return 1
    
    # 提取60mT數據
    print("\n" + "="*60)
    data_60mt = extract_sample_data(df, SAMPLE_005_1_60MT)
    if data_60mt is None:
        print("❌ 60mT數據提取失敗")
        return 1
    
    # 分析參數
    stats_30mt = analyze_sample_parameters(data_30mt, SAMPLE_005_1_30MT['sample_name'])
    stats_60mt = analyze_sample_parameters(data_60mt, SAMPLE_005_1_60MT['sample_name'])
    
    # 穿透率T專項分析
    print("\n" + "="*60)
    transparency_results = analyze_transparency_parameter(data_30mt, data_60mt, output_dir)
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建圖表
    print("\n📈 生成圖表...")
    create_angular_plots(data_30mt, SAMPLE_005_1_30MT['sample_name'], output_dir)
    create_angular_plots(data_60mt, SAMPLE_005_1_60MT['sample_name'], output_dir)
    create_field_comparison_plot(data_30mt, data_60mt, output_dir)
    
    # 創建比較表
    print("\n📊 生成比較分析...")
    comparison_df = create_comparison_table(data_30mt, data_60mt, output_dir)
    
    # 生成報告
    print("\n📄 生成分析報告...")
    generate_summary_report(stats_30mt, stats_60mt, comparison_df, transparency_results, output_dir)
    
    print(f"\n🎉 分析完成！結果保存在: {output_dir}")
    print("\n生成的文件:")
    for file in sorted(output_dir.glob("*")):
        if file.name.endswith('.png'):
            print(f"  🖼️  {file.name}")
        elif file.name.endswith('.csv'):
            print(f"  📊 {file.name}")
        elif file.name.endswith('.txt'):
            print(f"  📄 {file.name}")
        else:
            print(f"  📁 {file.name}")
    
    # 特別標註穿透率分析文件
    transparency_files = [
        'sample_005_1_transparency_analysis.png',
        'sample_005_1_transparency_comparison.csv'
    ]
    
    print(f"\n🔬 穿透率T專項分析文件:")
    for filename in transparency_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (未生成)")
    
    return 0

if __name__ == "__main__":
    exit(main())
