"""
Advanced visualization module with publication-quality plots
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multithreading
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class PublicationPlotter:
    """Publication-quality plotting with consistent styling"""
    
    def __init__(self, config):
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.setup_matplotlib()
        
    def setup_matplotlib(self):
        """Configure matplotlib for publication quality"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    def create_fitted_curve_plot(self, x_data: np.ndarray, y_data: np.ndarray, 
                               fitted_data: np.ndarray, parameters: dict, 
                               statistics: dict, dataid: str, normalized: bool = True) -> plt.Figure:
        """Create fitted curve plot with parameter information"""
        fig, ax = plt.subplots(figsize=self.config.get('FIGURE_SIZE', (12, 8)))
        
        # Sort data for proper line connections
        sort_indices = np.argsort(x_data)
        x_sorted = x_data[sort_indices]
        y_sorted = y_data[sort_indices]
        fitted_sorted = fitted_data[sort_indices]
        
        # Plot original data
        ax.plot(x_sorted, y_sorted, '--', color=self.colors[0], linewidth=1, 
                alpha=0.7, label=f'{dataid} (connected)')
        ax.scatter(x_data, y_data, color=self.colors[0], s=20, alpha=0.8, zorder=5)
        
        # Plot fitted curve
        ax.plot(x_sorted, fitted_sorted, color=self.colors[1], linewidth=2, 
                label='Josephson Model Fit')
        
        # Plot linear trend
        linear_trend = parameters['r'] * x_data + parameters['C']
        linear_sorted = linear_trend[sort_indices]
        ax.plot(x_sorted, linear_sorted, '--', color=self.colors[2], linewidth=2, 
                alpha=0.8, label='Linear Trend (rx+C)')
        
        # Labels and title
        scale_text = "Normalized" if normalized else "Original"
        ax.set_xlabel(f'{scale_text} External Magnetic Flux (Φ_ext)')
        ax.set_ylabel(f'{scale_text} Supercurrent (I_s)')
        ax.set_title(f'Josephson Junction Analysis - {dataid}')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add parameter text box
        param_text = self._format_parameters(parameters)
        stats_text = self._format_statistics(statistics)
        
        ax.text(1.02, 0.65, 'Fitted Parameters:', transform=ax.transAxes, 
                fontweight='bold', fontsize=11)
        ax.text(1.02, 0.40, param_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.text(1.02, 0.25, 'Statistical Metrics:', transform=ax.transAxes, 
                fontweight='bold', fontsize=11)
        ax.text(1.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_residuals_analysis_plot(self, x_data: np.ndarray, y_data: np.ndarray, 
                                     fitted_data: np.ndarray, dataid: str) -> plt.Figure:
        """Create comprehensive residuals analysis plot"""
        residuals = y_data - fitted_data
        
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Main residuals plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(x_data, residuals, color=self.colors[2], s=15, alpha=0.7)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Add trend line
        z = np.polyfit(x_data, residuals, 1)
        p = np.poly1d(z)
        ax1.plot(x_data, p(x_data), color=self.colors[1], linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('External Magnetic Flux')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'Residual Analysis - {dataid}')
        ax1.grid(True, alpha=0.3)
        
        # Residuals vs fitted values
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(fitted_data, residuals, alpha=0.6, s=10, color=self.colors[3])
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('Fitted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Fitted')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Normal Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax4 = fig.add_subplot(gs[1, 2])
        n, bins, patches = ax4.hist(residuals, bins=25, alpha=0.7, density=True, 
                                   edgecolor='black', color=self.colors[4])
        
        # Overlay normal distribution
        x_normal = np.linspace(residuals.min(), residuals.max(), 100)
        y_normal = stats.norm.pdf(x_normal, np.mean(residuals), np.std(residuals))
        ax4.plot(x_normal, y_normal, 'r-', linewidth=2, label='Normal Fit')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title('Residual Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Autocorrelation of residuals
        ax5 = fig.add_subplot(gs[2, 0])
        lags = range(-min(50, len(residuals)//4), min(50, len(residuals)//4))
        autocorr = [np.corrcoef(residuals[max(0, lag):len(residuals)+min(0, lag)], 
                               residuals[max(0, -lag):len(residuals)+min(0, -lag)])[0,1] 
                   for lag in lags]
        ax5.plot(lags, autocorr, marker='o', markersize=3)
        ax5.axhline(y=0, color='red', linestyle='--')
        ax5.set_xlabel('Lag')
        ax5.set_ylabel('Autocorrelation')
        ax5.set_title('Residual Autocorrelation')
        ax5.grid(True, alpha=0.3)
        
        # Scale-location plot
        ax6 = fig.add_subplot(gs[2, 1])
        sqrt_residuals = np.sqrt(np.abs(residuals))
        ax6.scatter(fitted_data, sqrt_residuals, alpha=0.6, s=10, color=self.colors[5])
        ax6.set_xlabel('Fitted Values')
        ax6.set_ylabel('√|Residuals|')
        ax6.set_title('Scale-Location Plot')
        ax6.grid(True, alpha=0.3)
        
        # Residual statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_text = f"""Residual Statistics:
Mean: {np.mean(residuals):.2e}
Std Dev: {np.std(residuals):.2e}
Skewness: {stats.skew(residuals):.3f}
Kurtosis: {stats.kurtosis(residuals):.3f}
Jarque-Bera p-value: {stats.jarque_bera(residuals)[1]:.3f}
Durbin-Watson: {self._durbin_watson(residuals):.3f}"""
        
        ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_phase_folded_plot(self, phase_data: dict, dataid: str, 
                               frequency: float) -> plt.Figure:
        """Create enhanced phase-folded plot with drift analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Main phase-folded plot
        cycle_data = phase_data['cycle_data']
        total_cycles = phase_data['total_cycles']
        
        # Plot individual cycles
        for cycle in range(min(total_cycles, 10)):  # Limit to 10 cycles for clarity
            if cycle in cycle_data:
                color = self.colors[cycle % len(self.colors)]
                data = cycle_data[cycle]
                
                ax1.plot(data['phase'], data['values'], 'o-', color=color, 
                        label=f'Cycle {cycle + 1}', markersize=4, linewidth=1.5, alpha=0.8)
                
                # Mark peak
                ax1.scatter([data['peak_phase']], [data['peak_value']], 
                           color=color, s=100, marker='*', edgecolors='black', 
                           linewidth=1, zorder=5)
        
        # Plot average profile
        bin_centers = phase_data['bin_centers']
        mean_values = phase_data['mean_binned_values']
        valid_mask = ~np.isnan(mean_values)
        
        if np.any(valid_mask):
            ax1.plot(bin_centers[valid_mask], mean_values[valid_mask], 
                    'k--', linewidth=3, label='Average Profile', 
                    marker='s', markersize=6)
        
        ax1.set_xlabel(f'Phase (Period = {1/frequency:.6f})')
        ax1.set_ylabel('Normalized Supercurrent')
        ax1.set_title(f'Phase-Folded Analysis - {dataid} ({total_cycles} cycles)')
        ax1.set_xlim(0, 1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Phase drift analysis
        if 'drift_stats' in phase_data:
            drift_stats = phase_data['drift_stats']
            peak_phases = phase_data['cycle_peak_phases']
            
            if len(peak_phases) > 1:
                cycles = np.arange(len(peak_phases))
                ax2.plot(cycles, peak_phases, 'bo-', markersize=6, linewidth=2)
                
                # Add trend line
                if len(peak_phases) > 2:
                    z = np.polyfit(cycles, peak_phases, 1)
                    p = np.poly1d(z)
                    ax2.plot(cycles, p(cycles), 'r--', linewidth=2, alpha=0.8, 
                            label=f'Trend: {z[0]:.4f} phase/cycle')
                
                ax2.set_xlabel('Cycle Number')
                ax2.set_ylabel('Peak Phase Position')
                ax2.set_title('Phase Drift Analysis')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Add statistics
                stats_text = f"""Drift Statistics:
Mean drift: {drift_stats['mean_drift']:.6f}
Std deviation: {drift_stats['std_drift']:.6f}
Max drift: {drift_stats['max_drift']:.6f}"""
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_cycle_colored_plot(self, x_data: np.ndarray, y_data: np.ndarray, 
                                phase_data: dict, dataid: str, frequency: float) -> plt.Figure:
        """Create plot with cycles colored differently"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        cycle_number = phase_data['cycle_number']
        total_cycles = phase_data['total_cycles']
        
        # Plot each cycle with different color
        for cycle in range(total_cycles):
            mask = cycle_number == cycle
            if np.any(mask):
                color = self.colors[cycle % len(self.colors)]
                ax.scatter(x_data[mask], y_data[mask], color=color, 
                          label=f'Cycle {cycle + 1}', s=25, alpha=0.8)
        
        # Add cycle boundaries
        for cycle in range(1, total_cycles):
            boundary = cycle / frequency
            if boundary <= np.max(x_data):
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Normalized External Magnetic Flux (Φ_ext)')
        ax.set_ylabel('Normalized Supercurrent (I_s)')
        ax.set_title(f'Data Segmented by Cycles - {dataid} ({total_cycles} cycles)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, filepath: str, formats: List[str] = None):
        """Save plot in multiple formats with error handling"""
        if formats is None:
            formats = self.config.get('PLOT_FORMATS', ['png'])
        
        filepath = Path(filepath)
        
        for fmt in formats:
            try:
                output_path = filepath.with_suffix(f'.{fmt}')
                fig.savefig(output_path, dpi=self.config.dpi, 
                           bbox_inches='tight', format=fmt, 
                           facecolor='white', edgecolor='none')
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Failed to save {output_path}: {e}")
        
        plt.close(fig)  # Clean up memory
    
    def _format_parameters(self, parameters: dict) -> str:
        """Format parameters for display"""
        return f"""I_c: {parameters['I_c']:.3e}
φ_0: {parameters['phi_0']:.3f}
f: {parameters['f']:.3e}
T: {parameters['T']:.1%}
r: {parameters['r']:.3e}
C: {parameters['C']:.3e}"""
    
    def _format_statistics(self, statistics: dict) -> str:
        """Format statistics for display"""
        return f"""R²: {statistics['r_squared']:.4f}
Adj. R²: {statistics['adj_r_squared']:.4f}
RMSE: {statistics['rmse']:.4f}
MAE: {statistics['mae']:.4f}
SSE: {statistics['ss_res']:.2e}"""
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic"""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)