#!/usr/bin/env python3
"""
CPR Results Analysis Example
This script demonstrates how to analyze CPR processing results, including statistical analysis and visualization
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add project path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / 'src'))

def load_analysis_results(results_dir):
    """Load analysis results"""
    results_dir = Path(results_dir)
    
    # Load statistical summary
    stats_file = results_dir / 'reports' / 'analysis_stats.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        print(f"❌ Statistics file does not exist: {stats_file}")
        return None, None
    
    # Load detailed results
    detailed_file = results_dir / 'reports' / 'detailed_results.csv'
    if detailed_file.exists():
        detailed_df = pd.read_csv(detailed_file)
    else:
        print(f"❌ Detailed results file does not exist: {detailed_file}")
        return stats, None
    
    return stats, detailed_df

def create_analysis_visualizations(stats, detailed_df, output_dir):
    """Create analysis visualization charts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Processing results summary pie chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CPR Analysis Results Overview', fontsize=16, fontweight='bold')
    
    # Pie chart - Processing results distribution
    labels = ['Successfully Processed', 'Skipped (Data Quality)', 'Processing Failed']
    sizes = [stats['successful'], stats['skipped'], stats['failed']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Processing Results Distribution')
    
    # 2. I_c distribution histogram
    if detailed_df is not None and not detailed_df.empty:
        successful_df = detailed_df[detailed_df['success'] == True]
        if not successful_df.empty and 'I_c' in successful_df.columns:
            # Use logarithmic scale to display I_c distribution
            i_c_values = successful_df['I_c'].dropna()
            log_i_c = np.log10(i_c_values + 1e-12)  # Avoid log(0)
            
            axes[0,1].hist(log_i_c, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_title('Critical Current (I_c) Distribution (Log Scale)')
            axes[0,1].set_xlabel('log₁₀(I_c)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
    
    # 3. R² distribution histogram
    if detailed_df is not None and not detailed_df.empty:
        successful_df = detailed_df[detailed_df['success'] == True]
        if not successful_df.empty and 'r_squared' in successful_df.columns:
            r_squared_values = successful_df['r_squared'].dropna()
            
            axes[1,0].hist(r_squared_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1,0].set_title('Fitting Quality (R²) Distribution')
            axes[1,0].set_xlabel('R²')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add statistical line
            mean_r2 = r_squared_values.mean()
            axes[1,0].axvline(mean_r2, color='red', linestyle='--', 
                             label=f'Mean: {mean_r2:.3f}')
            axes[1,0].legend()
    
    # 4. Failure reason analysis
    if 'failure_reasons' in stats and stats['failure_reasons']:
        reasons = list(stats['failure_reasons'].keys())
        counts = list(stats['failure_reasons'].values())
        
        # Simplify reason names
        simplified_reasons = []
        for reason in reasons:
            if 'Insufficient data points' in reason:
                simplified_reasons.append('Insufficient Data')
            elif 'Data quality insufficient' in reason:
                simplified_reasons.append('Poor Data Quality')
            elif 'Curve fitting failed' in reason:
                simplified_reasons.append('Fitting Failed')
            else:
                simplified_reasons.append(reason[:20] + '...' if len(reason) > 20 else reason)
        
        axes[1,1].barh(simplified_reasons, counts, color='salmon')
        axes[1,1].set_title('Failure Reason Analysis')
        axes[1,1].set_xlabel('Number of Files')
        
        # Add value labels
        for i, v in enumerate(counts):
            axes[1,1].text(v + 0.1, i, str(v), va='center')
    
    plt.tight_layout()
    
    # Save chart
    summary_plot_path = output_dir / 'analysis_summary_visualization.png'
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Summary chart saved: {summary_plot_path}")
    plt.close()
    
    # 5. Create detailed statistical charts
    if detailed_df is not None and not detailed_df.empty:
        create_detailed_statistics_plot(detailed_df, output_dir)

def create_detailed_statistics_plot(detailed_df, output_dir):
    """Create detailed statistical analysis charts"""
    successful_df = detailed_df[detailed_df['success'] == True]
    
    if successful_df.empty:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CPR Detailed Statistical Analysis', fontsize=16, fontweight='bold')
    
    # 1. I_c vs R² scatter plot
    if 'I_c' in successful_df.columns and 'r_squared' in successful_df.columns:
        i_c_values = successful_df['I_c'].dropna()
        r_squared_values = successful_df['r_squared'].dropna()
        
        # Ensure both sequences have the same length
        min_len = min(len(i_c_values), len(r_squared_values))
        if min_len > 0:
            axes[0,0].scatter(np.log10(i_c_values[:min_len] + 1e-12), 
                            r_squared_values[:min_len], 
                            alpha=0.6, s=30)
            axes[0,0].set_xlabel('log₁₀(I_c)')
            axes[0,0].set_ylabel('R²')
            axes[0,0].set_title('Critical Current vs Fitting Quality')
            axes[0,0].grid(True, alpha=0.3)
    
    # 2. I_c box plot (if categorization info available)
    if 'I_c' in successful_df.columns:
        i_c_values = successful_df['I_c'].dropna()
        log_i_c = np.log10(i_c_values + 1e-12)
        
        axes[0,1].boxplot(log_i_c, labels=['All Data'])
        axes[0,1].set_title('Critical Current Distribution (Box Plot)')
        axes[0,1].set_ylabel('log₁₀(I_c)')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. R² box plot
    if 'r_squared' in successful_df.columns:
        r_squared_values = successful_df['r_squared'].dropna()
        
        axes[0,2].boxplot(r_squared_values, labels=['All Data'])
        axes[0,2].set_title('Fitting Quality Distribution (Box Plot)')
        axes[0,2].set_ylabel('R²')
        axes[0,2].grid(True, alpha=0.3)
    
    # 4. Processing time analysis (if available)
    if 'processing_time' in successful_df.columns:
        time_values = successful_df['processing_time'].dropna()
        axes[1,0].hist(time_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1,0].set_title('Processing Time Distribution')
        axes[1,0].set_xlabel('Processing Time (seconds)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'No Processing Time Data', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Processing Time Distribution')
    
    # 5. Success rate trend (sorted by filename)
    axes[1,1].plot(range(len(successful_df)), [1]*len(successful_df), 'g-', alpha=0.7, linewidth=2)
    axes[1,1].set_title('Successful File Processing Trend')
    axes[1,1].set_xlabel('File Index')
    axes[1,1].set_ylabel('Success Indicator')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Data quality assessment
    if 'r_squared' in successful_df.columns:
        r_squared_values = successful_df['r_squared'].dropna()
        
        # Classify quality based on R² values
        excellent = sum(r_squared_values >= 0.9)
        good = sum((r_squared_values >= 0.7) & (r_squared_values < 0.9))
        fair = sum((r_squared_values >= 0.5) & (r_squared_values < 0.7))
        poor = sum(r_squared_values < 0.5)
        
        quality_labels = ['Excellent\n(R²≥0.9)', 'Good\n(0.7≤R²<0.9)', 'Fair\n(0.5≤R²<0.7)', 'Poor\n(R²<0.5)']
        quality_counts = [excellent, good, fair, poor]
        quality_colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
        
        bars = axes[1,2].bar(quality_labels, quality_counts, color=quality_colors, alpha=0.8)
        axes[1,2].set_title('Fitting Quality Assessment')
        axes[1,2].set_ylabel('Number of Files')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                              f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save detailed statistics chart
    detailed_plot_path = output_dir / 'detailed_statistics.png'
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed statistics chart saved: {detailed_plot_path}")
    plt.close()

def generate_summary_report(stats, detailed_df, output_file):
    """Generate text summary report"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# CPR Analysis Results Summary Report\n\n")
        f.write(f"Generated time: {stats.get('timestamp', 'N/A')}\n\n")
        
        f.write("## Processing Overview\n\n")
        f.write(f"- **Total files**: {stats['total_files']}\n")
        f.write(f"- **Successfully processed**: {stats['successful']} ({stats['success_rate']:.1f}%)\n")
        f.write(f"- **Skipped (data quality)**: {stats['skipped']} ({stats['skip_rate']:.1f}%)\n")
        f.write(f"- **Processing failed**: {stats['failed']}\n")
        f.write(f"- **Total processing time**: {stats['processing_time']:.2f} seconds\n")
        f.write(f"- **Average processing time**: {stats['average_time_per_file']:.3f} seconds/file\n\n")
        
        if 'i_c_statistics' in stats:
            i_c_stats = stats['i_c_statistics']
            f.write("## Critical Current (I_c) Statistics\n\n")
            f.write(f"- **Mean**: {i_c_stats['mean']:.3e} A\n")
            f.write(f"- **Standard deviation**: {i_c_stats['std']:.3e} A\n")
            f.write(f"- **Minimum**: {i_c_stats['min']:.3e} A\n")
            f.write(f"- **Maximum**: {i_c_stats['max']:.3e} A\n")
            f.write(f"- **Median**: {i_c_stats['median']:.3e} A\n\n")
        
        if 'r_squared_statistics' in stats:
            r2_stats = stats['r_squared_statistics']
            f.write("## Fitting Quality (R²) Statistics\n\n")
            f.write(f"- **Mean**: {r2_stats['mean']:.4f}\n")
            f.write(f"- **Standard deviation**: {r2_stats['std']:.4f}\n")
            f.write(f"- **Minimum**: {r2_stats['min']:.4f}\n")
            f.write(f"- **Maximum**: {r2_stats['max']:.4f}\n")
            f.write(f"- **Median**: {r2_stats['median']:.4f}\n\n")
        
        if 'skip_reasons' in stats and stats['skip_reasons']:
            f.write("## Skip Reason Analysis\n\n")
            for reason, count in stats['skip_reasons'].items():
                f.write(f"- **{reason}**: {count} files\n")
            f.write("\n")
        
        if 'failure_reasons' in stats and stats['failure_reasons']:
            f.write("## Failure Reason Analysis\n\n")
            for reason, count in stats['failure_reasons'].items():
                f.write(f"- **{reason}**: {count} files\n")
            f.write("\n")
        
        f.write("## Recommendations\n\n")
        success_rate = stats['success_rate']
        if success_rate >= 90:
            f.write("🎉 Processing results are excellent! Most files were successfully processed.\n\n")
        elif success_rate >= 80:
            f.write("✅ Processing results are good, with only a few files requiring attention to data quality.\n\n")
        elif success_rate >= 60:
            f.write("⚠️ Processing results are average, recommend checking data quality of failed files.\n\n")
        else:
            f.write("❌ Processing results are unsatisfactory, need to check data quality and processing parameters.\n\n")
        
        if stats['failed'] > 0:
            f.write("For failed files, recommendations:\n")
            f.write("1. Check if the number of data points is sufficient (recommend ≥20 points)\n")
            f.write("2. Check data quality and variation range\n")
            f.write("3. Consider adjusting preprocessing parameters\n\n")

def main():
    """Main function - Results analysis example"""
    print("="*60)
    print("CPR Results Analysis Example")
    print("="*60)
    
    # Set paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'output' / 'full_analysis'
    output_dir = project_root / 'output' / 'analysis_visualization'
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"❌ Results directory does not exist: {results_dir}")
        print("Please first run scripts/analyze_all_csv.py to generate analysis results")
        return
    
    print(f"📁 Loading results: {results_dir}")
    
    # Load analysis results
    stats, detailed_df = load_analysis_results(results_dir)
    
    if stats is None:
        print("❌ Unable to load analysis results")
        return
    
    print(f"✅ Successfully loaded statistical data")
    if detailed_df is not None:
        print(f"✅ Successfully loaded detailed results ({len(detailed_df)} records)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {output_dir}")
    
    # Generate visualizations
    print("\n🎨 Generating visualization charts...")
    create_analysis_visualizations(stats, detailed_df, output_dir)
    
    # Generate summary report
    print("\n📄 Generating summary report...")
    report_file = output_dir / 'analysis_summary_report.md'
    generate_summary_report(stats, detailed_df, report_file)
    print(f"✅ Summary report saved: {report_file}")
    
    print("\n" + "="*60)
    print("Results analysis completed!")
    print("="*60)
    print(f"\n📊 Processing overview:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Successfully processed: {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"   Skipped (data quality): {stats['skipped']}")
    print(f"   Processing failed: {stats['failed']}")
    
    print(f"\n📁 Output files:")
    print(f"   Overview chart: {output_dir}/analysis_summary_visualization.png")
    print(f"   Detailed statistics: {output_dir}/detailed_statistics.png")
    print(f"   Summary report: {output_dir}/analysis_summary_report.md")

if __name__ == "__main__":
    main()