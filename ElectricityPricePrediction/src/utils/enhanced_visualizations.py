"""
Enhanced Visualization Module (A–Z Refactor)
Generates high-resolution (300 DPI) plots for model analysis.
Uses consistent 'ggplot' style and robust Pathlib handling.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Enforce consistent plotting style
plt.style.use('ggplot')

def create_comprehensive_comparison(predictions, metrics_df, output_dir):
    """Create separate individual analysis plots at 300 DPI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_len = min(len(predictions), 7*24)
    data = predictions.iloc[-plot_len:]
    exclude = ['Actual', 'LSTM_uncertainty', 'XGBoost_uncertainty', 'P05', 'P95', 'P50', 
               'LSTM_P05', 'LSTM_P95', 'XGBoost_P05', 'XGBoost_P95']
    
    # === 1. Forecast Comparison (SEPARATE) ===
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Actual'], label='Actual', color='black', linewidth=2, alpha=0.8)
    for col in [c for c in data.columns if c not in exclude]:
        plt.plot(data.index, data[col], label=col, linewidth=1.2, alpha=0.7)
        # Add Uncertainty Bands if P05/P95 columns exist
        p05_col = f"{col}_P05" if f"{col}_P05" in data.columns else "P05" if col == "LSTM" else None
        p95_col = f"{col}_P95" if f"{col}_P95" in data.columns else "P95" if col == "LSTM" else None
        
        if p05_col and p95_col and p05_col in data.columns and p95_col in data.columns:
            plt.fill_between(data.index, data[p05_col], data[p95_col], alpha=0.1, label=f'{col} 90% CI')
            
    plt.title(f'Forecast Comparison (Last {plot_len//24} Days)', fontweight='bold')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "forecast_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 2. RMSE Comparison (SEPARATE) ===
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='RMSE', y='Model', palette='viridis')
    plt.title('Model Performance (RMSE)', fontweight='bold')
    for i, v in enumerate(metrics_df['RMSE']):
        plt.text(v + 0.5, i, f'{v:.2f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 3. Error Distribution (SEPARATE) ===
    plt.figure(figsize=(10, 6))
    for col in [c for c in predictions.columns if c not in exclude]:
        sns.kdeplot(predictions[col] - predictions['Actual'], label=col, fill=True, alpha=0.3)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Error Distribution (Residuals)', fontweight='bold')
    plt.xlabel('Error (Predicted - Actual)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 4. Intraday Profile (SEPARATE) ===
    if 'Actual' in predictions.columns:
        plt.figure(figsize=(10, 6))
        hourly_avg = predictions['Actual'].groupby(predictions.index.hour).mean()
        plt.plot(hourly_avg.index, hourly_avg.values, marker='o', color='teal', linewidth=2)
        plt.title('Average Intra-day Price Profile', fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Avg Price')
        plt.xticks(range(0, 24, 4))
        plt.tight_layout()
        plt.savefig(output_dir / "intraday_profile.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also keep the combined version for backward compat
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Electricity Price Forecasting: Model Comparison', fontsize=18, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['Actual'], label='Actual', color='black', linewidth=2, alpha=0.8)
    for col in [c for c in data.columns if c not in exclude]:
        ax1.plot(data.index, data[col], label=col, linewidth=1.2, alpha=0.7)
    ax1.set_title(f'Recent Forecasts (Last {plot_len//24} Days)', fontweight='bold')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.legend()
    
    ax2 = axes[0, 1]
    sns.barplot(data=metrics_df, x='RMSE', y='Model', ax=ax2, palette='viridis')
    ax2.set_title('Model Performance (RMSE)', fontweight='bold')
    for i, v in enumerate(metrics_df['RMSE']):
        ax2.text(v + 0.5, i, f'{v:.2f}', va='center', fontweight='bold')
    
    ax3 = axes[1, 0]
    for col in [c for c in predictions.columns if c not in exclude]:
        sns.kdeplot(predictions[col] - predictions['Actual'], ax=ax3, label=col, fill=True, alpha=0.3)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_title('Error Distribution (Residuals)', fontweight='bold')
    ax3.set_xlabel('Error (Predicted - Actual)')
    ax3.legend()
    
    ax4 = axes[1, 1]
    if 'Actual' in predictions.columns:
        hourly_avg = predictions['Actual'].groupby(predictions.index.hour).mean()
        ax4.plot(hourly_avg.index, hourly_avg.values, marker='o', color='teal', linewidth=2)
        ax4.set_title('Average Intra-day Price Profile', fontweight='bold')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Avg Price')
        ax4.set_xticks(range(0, 24, 4))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_forecasting_insights(predictions, output_dir):
    """Detailed insights into model errors by time of day/week."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Error by Hour
    preds_copy = predictions.copy()
    preds_copy['hour'] = preds_copy.index.hour
    exclude = ['Actual', 'LSTM_uncertainty', 'P05', 'P95', 'P50', 'hour', 'dow']
    for col in [c for c in preds_copy.columns if c not in exclude]:
        mae_hour = (preds_copy[col] - preds_copy['Actual']).abs().groupby(preds_copy['hour']).mean()
        ax1.plot(mae_hour.index, mae_hour.values, marker='s', label=f'{col} MAE')
    
    ax1.set_title('Mean Absolute Error by Hour of Day', fontweight='bold')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('MAE (EUR/MWh)')
    ax1.set_xticks(range(24))
    ax1.legend()

    # Error by Day of Week
    preds_copy['dow'] = preds_copy.index.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for col in [c for c in preds_copy.columns if c not in exclude]:
        mae_dow = (preds_copy[col] - preds_copy['Actual']).abs().groupby(preds_copy['dow']).mean()
        ax2.plot(mae_dow.index, mae_dow.values, marker='D', label=f'{col} MAE')
    
    ax2.set_title('Mean Absolute Error by Day of Week', fontweight='bold')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(day_names)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "forecasting_insights.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_impact_analysis(model, feature_names, output_dir):
    """Plot feature importance at 300 DPI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    # Take top 20
    indices = np.argsort(importances)[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Feature Importance Analysis (XGBoost)', fontweight='bold')
    plt.xlabel('Relative Importance')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_probabilistic_summary(predictions, output_dir):
    """
    Dedicated visualization for Probabilistic Analysis.
    Focuses on Uncertainty Width and Interval Calibration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify models with probabilistic data
    models = []
    if 'LSTM_P05' in predictions.columns and 'LSTM_P95' in predictions.columns:
        models.append('LSTM')
    if 'XGBoost_P05' in predictions.columns and 'XGBoost_P95' in predictions.columns:
        models.append('XGBoost')
        
    if not models:
        return

    # Prepare Data for plotting
    analysis_data = []
    for model in models:
        p05 = predictions[f'{model}_P05']
        p95 = predictions[f'{model}_P95']
        width = p95 - p05
        
        # Calibration: % of points inside interval
        inside = ((predictions['Actual'] >= p05) & (predictions['Actual'] <= p95)).mean() * 100
        
        analysis_data.append({
            'Model': model,
            'Avg Width': width.mean(),
            'Calibration (%)': inside,
            'Widths': width
        })

    # === PLOTTING ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Advanced Probabilistic Forecast Analysis', fontsize=16, fontweight='bold')

    # 1. Uncertainty Width (Precision)
    width_df = pd.DataFrame({m['Model']: m['Widths'] for m in analysis_data})
    sns.boxplot(data=width_df, ax=axes[0], palette='pastel')
    axes[0].set_title('Uncertainty Interval Width (Lower is more precise)', fontweight='bold')
    axes[0].set_ylabel('Width (P95 - P05) in Price Units')

    # 2. Calibration (Reliability)
    cal_df = pd.DataFrame([{'Model': m['Model'], 'Calibration': m['Calibration (%)']} for m in analysis_data])
    sns.barplot(data=cal_df, x='Model', y='Calibration', ax=axes[1], palette='viridis')
    axes[1].axhline(90, color='red', linestyle='--', label='Target (90%)')
    axes[1].set_title('Interval Calibration (Target = 90%)', fontweight='bold')
    axes[1].set_ylabel('Actual Coverage (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "probabilistic_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Uncertainty over Time (Last 3 days)
    plt.figure(figsize=(12, 5))
    plot_len = min(len(predictions), 3*24)
    data_tail = predictions.iloc[-plot_len:]
    
    for model in models:
        w = data_tail[f'{model}_P95'] - data_tail[f'{model}_P05']
        plt.plot(data_tail.index, w, label=f'{model} Uncertainty Width', linewidth=2)
    
    plt.title(f'Uncertainty Dynamics (Last {plot_len//24} Days)', fontweight='bold')
    plt.ylabel('Interval Width')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_dynamics.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_probabilistic_fan_chart(history_df, future_df, output_dir):
    """
    Creates a 'Fan Chart' showing historical Actuals connecting to 
    future Probabilistic Forecasts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Take last 48h of history
    hist_len = min(len(history_df), 48)
    history = history_df.iloc[-hist_len:]
    
    # Identify models in future_df
    models = []
    if 'LSTM' in future_df.columns: models.append('LSTM')
    if 'XGBoost' in future_df.columns: models.append('XGBoost')
    
    if not models:
        return

    plt.figure(figsize=(14, 7))
    
    # 1. Plot History
    plt.plot(history.index, history['Actual'], label='Historical Actual', color='black', linewidth=2.5)
    
    # Get last point of history to connect
    last_hist_time = history.index[-1]
    last_hist_val = history['Actual'].iloc[-1]
    
    colors = {'LSTM': '#1f77b4', 'XGBoost': '#ff7f0e'} # Blue, Orange
    
    for model in models:
        # Prepend last history point for visual continuity
        m_times = [last_hist_time] + list(future_df.index)
        m_preds = [last_hist_val] + list(future_df[model])
        
        plt.plot(m_times, m_preds, label=f'{model} Forecast', color=colors[model], linewidth=2, linestyle='--')
        
        # Uncertainty Bands
        p05_col = f"{model}_P05"
        p95_col = f"{model}_P95"
        
        if p05_col in future_df.columns and p95_col in future_df.columns:
            m_p05 = [last_hist_val] + list(future_df[p05_col])
            m_p95 = [last_hist_val] + list(future_df[p95_col])
            plt.fill_between(m_times, m_p05, m_p95, color=colors[model], alpha=0.15, label=f'{model} 90% CI')

    plt.axvline(last_hist_time, color='gray', linestyle=':', alpha=0.5)
    plt.text(last_hist_time, plt.ylim()[0], ' Forecast Start ', rotation=90, verticalalignment='bottom', alpha=0.7)

    plt.title('Out-of-Sample Probabilistic Fan Chart (Next 24h)', fontsize=16, fontweight='bold')
    plt.ylabel('Price (EUR/MWh)')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "probabilistic_fan_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_sensitivity(predictions, output_dir):
    """Plots how interval coverage changes with different scaling factors."""
    models = [c for c in predictions.columns if f"{c}_P05" in predictions.columns and f"{c}_P95" in predictions.columns]
    if not models or 'Actual' not in predictions.columns:
        return

    plt.figure(figsize=(10, 6))
    scalings = np.linspace(0.5, 3.5, 30)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        coverages = []
        actual = predictions['Actual']
        pred = predictions[model]
        
        # Current distance from mean to percentiles
        hw_low = pred - predictions[f"{model}_P05"]
        hw_high = predictions[f"{model}_P95"] - pred
        
        for s in scalings:
            # Calculate coverage if we scaled the CURRENT intervals by 's'
            p05 = pred - hw_low * s
            p95 = pred + hw_high * s
            inside = ((actual >= p05) & (actual <= p95)).mean() * 100
            coverages.append(inside)
            
        plt.plot(scalings, coverages, marker='o', label=f'{model} Coverage', linewidth=2, markersize=4)

    plt.axhline(y=90, color='r', linestyle='--', label='90% Target')
    plt.axvline(x=1.0, color='gray', linestyle=':', label='Current Setting')
    
    plt.title("Calibration Sensitivity: Coverage vs. Scaling Factor", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Scaling Multiplier (1.0 = Current)", fontsize=12)
    plt.ylabel("Actual Coverage (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_dir / "calibration_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_spaghetti_plot(history_df, future_preds, output_dir, n_samples=30):
    """
    Simulates and plots multiple 'possible futures' (scenarios) based on the 
    forecasted mean and uncertainty for each model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hist_len = min(len(history_df), 48)
    history = history_df.iloc[-hist_len:]
    last_hist_time = history.index[-1]
    last_hist_val = history['Actual'].iloc[-1]
    
    models = [c for c in ['LSTM', 'XGBoost'] if c in future_preds.columns and f"{c}_uncertainty" in future_preds.columns]
    if not models:
        return
        
    plt.figure(figsize=(14, 7))
    plt.plot(history.index, history['Actual'], label='Historical Actual', color='black', linewidth=2.5, zorder=10)
    
    colors = {'LSTM': 'blue', 'XGBoost': 'orange'}
    np.random.seed(42) # For reproducible spaghetti
    
    for model in models:
        m_times = [last_hist_time] + list(future_preds.index)
        mean_preds = future_preds[model].values
        stds = future_preds[f"{model}_uncertainty"].values
        
        # Plot simulated paths
        for i in range(n_samples):
            # To make paths look somewhat continuous (not just white noise at every hour),
            # we draw a base random offset for the path and add some hour-to-hour noise
            path_bias = np.random.normal(0, stds.mean() * 0.5) 
            path_noise = np.random.normal(0, stds)
            
            # Blend mean, bias, and noise
            path_vals = mean_preds + (path_bias + path_noise) * 0.8
            m_path = [last_hist_val] + list(path_vals)
            
            label = f"{model} Scenarios" if i == 0 else ""
            plt.plot(m_times, m_path, color=colors[model], alpha=0.15, linewidth=1, label=label)
            
        # Plot mean as solid line
        m_mean = [last_hist_val] + list(mean_preds)
        plt.plot(m_times, m_mean, color=colors[model], linewidth=2.5, linestyle='--', label=f"{model} Mean Forecast")
        
    plt.axvline(last_hist_time, color='gray', linestyle=':', alpha=0.5)
    plt.title('Spaghetti Plot: Simulated Price Scenarios (Next 24h)', fontsize=16, fontweight='bold')
    plt.ylabel('Price (EUR/MWh)')
    plt.xlabel('Time')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "spaghetti_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_density_plot(future_preds, output_dir):
    """
    Plots the Probability Density Function (PDF) for the peak price hour 
    in the future 24h forecast.
    """
    import scipy.stats as stats
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [c for c in ['LSTM', 'XGBoost'] if c in future_preds.columns and f"{c}_uncertainty" in future_preds.columns]
    if not models:
        return
        
    # Find the hour with the highest expected price (using first available model's peak)
    peak_hour = future_preds[models[0]].idxmax()
    
    plt.figure(figsize=(10, 6))
    colors = {'LSTM': 'blue', 'XGBoost': 'orange'}
    
    for model in models:
        mean_val = future_preds.loc[peak_hour, model]
        std_val = future_preds.loc[peak_hour, f"{model}_uncertainty"]
        
        # Generate x values covering 4 standard deviations
        x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 500)
        y = stats.norm.pdf(x, mean_val, std_val)
        
        plt.plot(x, y, color=colors[model], linewidth=2.5, label=f"{model} Distribution")
        plt.fill_between(x, y, alpha=0.2, color=colors[model])
        plt.axvline(mean_val, color=colors[model], linestyle='--', alpha=0.8, label=f"{model} Mean: {mean_val:.1f}")
        
        # Add Text Overlay for actual values
        if f"{model}_P05" in future_preds.columns and f"{model}_P95" in future_preds.columns:
            p05_val = future_preds.loc[peak_hour, f"{model}_P05"]
            p95_val = future_preds.loc[peak_hour, f"{model}_P95"]
            
            # Draw vertical lines for intervals
            plt.axvline(p05_val, color=colors[model], linestyle=':', alpha=0.6)
            plt.axvline(p95_val, color=colors[model], linestyle=':', alpha=0.6)
            
            # Add text box with metrics
            text_str = f"{model} Peak Hour Metrics:\n"
            text_str += f"Mean: {mean_val:.2f} €\n"
            text_str += f"P05 (Low): {p05_val:.2f} €\n"
            text_str += f"P95 (High): {p95_val:.2f} €"
            
            # Position depending on model to avoid overlap
            y_pos = 0.95 if model == 'LSTM' else 0.80
            x_pos = 0.05
            
            plt.gca().text(x_pos, y_pos, text_str, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=colors[model]))
        
    plt.title(f'Probability Density Function at Peak Hour\n({peak_hour.strftime("%Y-%m-%d %H:%00")})', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Price (EUR/MWh)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "density_plot_peak_hour.png", dpi=300, bbox_inches='tight')
    plt.close()
