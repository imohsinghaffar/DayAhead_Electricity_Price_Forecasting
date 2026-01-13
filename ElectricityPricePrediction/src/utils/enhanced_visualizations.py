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
    exclude = ['Actual', 'LSTM_uncertainty', 'P05', 'P95', 'P50']
    
    # === 1. Forecast Comparison (SEPARATE) ===
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Actual'], label='Actual', color='black', linewidth=2, alpha=0.8)
    for col in [c for c in data.columns if c not in exclude]:
        plt.plot(data.index, data[col], label=col, linewidth=1.2, alpha=0.7)
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
