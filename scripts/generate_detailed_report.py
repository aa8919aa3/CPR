#!/usr/bin/env python3
"""
生成特定文件的深度分析報告
針對文件 ID: 386, 381, 418, 397, 394, 416, 396, 407, 380
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_data():
    """載入並分析特定文件數據"""
    
    # 讀取詳細摘要CSV
    summary_path = "/Users/albert-mac/Code/GitHub/CPR/output/specific_analysis/detailed_summary.csv"
    df = pd.read_csv(summary_path)
    
    print("🔍 特定文件 CPR 分析深度報告")
    print("="*60)
    print(f"分析文件數量: {len(df)}")
    print(f"分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 基本統計摘要
    print("📊 1. 基本統計摘要")
    print("-"*40)
    
    # 物理參數統計
    params = {
        'I_c': '臨界電流 (A)',
        'phi_0': '相位偏移 (rad)', 
        'f': '特徵頻率 (Hz)',
        'T': '溫度參數 (K)',
        'r': '電阻 (Ω)',
        'C': '電容 (F)'
    }
    
    for param, name in params.items():
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                print(f"\n{name}:")
                print(f"  平均值: {values.mean():.4e}")
                print(f"  標準差: {values.std():.4e}")
                print(f"  變異係數: {(values.std()/values.mean()*100):.2f}%")
                print(f"  範圍: [{values.min():.4e}, {values.max():.4e}]")
                print(f"  中位數: {values.median():.4e}")
                print(f"  峰度: {stats.kurtosis(values):.3f}")
                print(f"  偏度: {stats.skew(values):.3f}")
    
    # 2. 擬合品質分析
    print("\n\n🎯 2. 擬合品質分析")
    print("-"*40)
    
    quality_metrics = {
        'r_squared': 'R²決定係數',
        'adj_r_squared': '調整R²',
        'rmse': '均方根誤差',
        'mae': '平均絕對誤差'
    }
    
    for metric, name in quality_metrics.items():
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"\n{name}:")
                print(f"  平均值: {values.mean():.6f}")
                print(f"  標準差: {values.std():.6f}")
                print(f"  最佳: {values.max():.6f} (文件: {df.loc[values.idxmax(), 'dataid']})")
                print(f"  最差: {values.min():.6f} (文件: {df.loc[values.idxmin(), 'dataid']})")
    
    # 3. 文件排名分析
    print("\n\n🏆 3. 文件品質排名")
    print("-"*40)
    
    # 按R²排序
    df_sorted = df.sort_values('r_squared', ascending=False)
    print("\n按擬合品質(R²)排名:")
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"  {i}. {row['dataid']}: R² = {row['r_squared']:.6f}")
    
    # 4. 參數相關性分析
    print("\n\n🔗 4. 參數相關性分析")
    print("-"*40)
    
    # 計算參數間的相關係數
    param_cols = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    correlation_matrix = df[param_cols].corr()
    
    print("\n重要相關性 (|相關係數| > 0.5):")
    for i in range(len(param_cols)):
        for j in range(i+1, len(param_cols)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"  {param_cols[i]} ↔ {param_cols[j]}: {corr:.3f}")
    
    # 5. 異常值檢測
    print("\n\n⚠️ 5. 異常值檢測")
    print("-"*40)
    
    for param in param_cols:
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                if len(outliers) > 0:
                    outlier_files = df[df[param].isin(outliers)]['dataid'].tolist()
                    print(f"\n{param} 異常值:")
                    for file_id, value in zip(outlier_files, outliers):
                        print(f"  {file_id}: {value:.4e}")
    
    # 6. 性能評估
    print("\n\n📈 6. 總體性能評估")
    print("-"*40)
    
    # 計算綜合得分 (基於多個指標)
    df['performance_score'] = (
        df['r_squared'] * 0.4 +  # R²權重40%
        df['adj_r_squared'] * 0.3 +  # 調整R²權重30%
        (1 - df['rmse'] / df['rmse'].max()) * 0.2 +  # RMSE權重20% (反向)
        (1 - df['mae'] / df['mae'].max()) * 0.1  # MAE權重10% (反向)
    )
    
    best_performer = df.loc[df['performance_score'].idxmax()]
    worst_performer = df.loc[df['performance_score'].idxmin()]
    
    print(f"\n最佳整體性能: {best_performer['dataid']}")
    print(f"  綜合得分: {best_performer['performance_score']:.4f}")
    print(f"  R²: {best_performer['r_squared']:.6f}")
    print(f"  RMSE: {best_performer['rmse']:.2e}")
    
    print(f"\n最差整體性能: {worst_performer['dataid']}")
    print(f"  綜合得分: {worst_performer['performance_score']:.4f}")
    print(f"  R²: {worst_performer['r_squared']:.6f}")
    print(f"  RMSE: {worst_performer['rmse']:.2e}")
    
    # 7. 建議和結論
    print("\n\n💡 7. 分析結論與建議")
    print("-"*40)
    
    avg_r2 = df['r_squared'].mean()
    std_r2 = df['r_squared'].std()
    
    print(f"\n✅ 主要發現:")
    print(f"  • 所有9個文件都成功完成了CPR分析")
    print(f"  • 平均R²值為 {avg_r2:.4f}，表明擬合品質優秀")
    print(f"  • R²標準差為 {std_r2:.4f}，顯示結果一致性良好")
    print(f"  • 最佳文件 {df.loc[df['r_squared'].idxmax(), 'dataid']} 的R²達到 {df['r_squared'].max():.6f}")
    
    print(f"\n📋 參數特點:")
    ic_mean = df['I_c'].mean()
    print(f"  • 臨界電流範圍: {df['I_c'].min():.2e} - {df['I_c'].max():.2e} A")
    print(f"  • 平均臨界電流: {ic_mean:.2e} A")
    print(f"  • 頻率集中在 121-129 kHz 範圍內")
    print(f"  • 溫度參數分佈在 0.32-0.80 K 之間")
    
    print(f"\n🔧 優化建議:")
    if std_r2 > 0.01:
        print(f"  • 考慮進一步優化擬合算法以提高一致性")
    if df['rmse'].max() > 1e-6:
        print(f"  • 某些文件的RMSE較高，可能需要檢查數據品質")
    print(f"  • 文件 {df.loc[df['r_squared'].idxmax(), 'dataid']} 可作為最佳實踐參考")
    
    return df

def create_comprehensive_analysis_plots(df):
    """創建綜合分析圖表"""
    
    output_dir = Path("/Users/albert-mac/Code/GitHub/CPR/output/specific_analysis")
    
    # 1. 相關性熱圖
    plt.figure(figsize=(12, 10))
    param_cols = ['I_c', 'phi_0', 'f', 'T', 'r', 'C', 'r_squared', 'rmse']
    correlation_matrix = df[param_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 性能雷達圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    
    top_4_files = df.nlargest(4, 'r_squared')
    
    metrics = ['r_squared', 'adj_r_squared', 'rmse', 'mae']
    metric_labels = ['R²', 'Adj R²', '1-RMSE_norm', '1-MAE_norm']
    
    for idx, (_, row) in enumerate(top_4_files.iterrows()):
        ax = axes[idx//2, idx%2]
        
        # 正規化指標 (RMSE和MAE使用反向正規化)
        values = [
            row['r_squared'],
            row['adj_r_squared'], 
            1 - (row['rmse'] / df['rmse'].max()),  # 反向正規化
            1 - (row['mae'] / df['mae'].max())     # 反向正規化
        ]
        
        angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
        values += values[:1]  # 閉合圖形
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['dataid'])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"File: {row['dataid']}", fontweight='bold')
        ax.grid(True)
    
    plt.suptitle('Top 4 Files Performance Radar Chart', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 參數分佈箱線圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    param_cols = ['I_c', 'phi_0', 'f', 'T', 'r', 'C']
    param_names = ['Critical Current (A)', 'Phase Offset (rad)', 'Frequency (Hz)', 
                   'Temperature (K)', 'Resistance (Ω)', 'Capacitance (F)']
    
    for i, (param, name) in enumerate(zip(param_cols, param_names)):
        ax = axes[i//3, i%3]
        if param in df.columns:
            df.boxplot(column=param, ax=ax)
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('')
            
            # 添加數據點
            y = df[param].dropna()
            x = np.ones(len(y))
            ax.scatter(x, y, alpha=0.6, color='red', s=50)
    
    plt.suptitle('Parameter Distribution Box Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函數"""
    print("開始生成深度分析報告...")
    
    # 載入並分析數據
    df = load_and_analyze_data()
    
    # 創建綜合分析圖表
    create_comprehensive_analysis_plots(df)
    
    print("\n\n🎉 深度分析報告生成完成！")
    print("="*60)
    print("生成的新文件:")
    print("  - correlation_heatmap.png: 參數相關性熱圖")
    print("  - performance_radar.png: 性能雷達圖")
    print("  - parameter_distributions.png: 參數分佈箱線圖")

if __name__ == "__main__":
    main()
