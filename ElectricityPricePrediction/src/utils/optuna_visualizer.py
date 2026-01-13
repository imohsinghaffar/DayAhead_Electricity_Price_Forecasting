"""
Optuna Visualization Module (A–Z Refactor)
Generates high-resolution (300 DPI) Optuna plots with consistent ggplot style.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Enforce style
plt.style.use('ggplot')

def visualize_optuna_history(json_path, output_dir):
    """Main function to generate all Optuna visualizations at 300 DPI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(json_path)
    
    if not json_path.exists():
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    trials = data.get('all_trials', data.get('trials', []))
    if not trials:
        return

    # Flatten for DataFrame
    records = []
    for t in trials:
        rec = {'number': t['trial_number'], 'value': t['rmse']}
        for k, v in t.get('params', {}).items():
            rec[k] = v
        records.append(rec)
    df = pd.DataFrame(records)
    param_cols = [c for c in df.columns if c not in ['number', 'value']]

    # 1. Progression Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['number'], df['value'], 'o-', alpha=0.6, label='Trial RMSE')
    plt.plot(df['number'], df['value'].cummin(), 'r-', linewidth=2, label='Best So Far')
    plt.title('Optimization Progress', fontweight='bold')
    plt.xlabel('Trial Number')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(output_dir / 'trial_progression.png', dpi=300)
    plt.close()

    # 2. Importance (Correlation-based)
    corrs = {c: abs(df[c].corr(df['value'])) for c in param_cols if df[c].dtype != object}
    if corrs:
        plt.figure(figsize=(10, 6))
        ser = pd.Series(corrs).sort_values()
        ser.plot(kind='barh', color='teal')
        plt.title('Hyperparameter Importance (Correlation)', fontweight='bold')
        plt.xlabel('Absolute Correlation with RMSE')
        plt.savefig(output_dir / 'hyperparameter_importance.png', dpi=300)
        plt.close()

    # 3. Slice Plot (Parameter vs RMSE)
    n_params = len(param_cols)
    if n_params > 0:
        rows = int(np.ceil(n_params/2))
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        axes = axes.flatten()
        for i, col in enumerate(param_cols):
            sns.scatterplot(data=df, x=col, y='value', ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Slice: {col}', fontweight='bold')
            axes[i].set_ylabel('RMSE')
        
        # Hide extra axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_slices.png', dpi=300)
        plt.close()

    # 4. Parallel Coordinate (Normalized)
    if n_params > 1:
        plt.figure(figsize=(12, 6))
        from pandas.plotting import parallel_coordinates
        # Normalize for visualization
        df_norm = df.copy()
        for col in param_cols + ['value']:
            if df_norm[col].dtype != object:
                c_min, c_max = df_norm[col].min(), df_norm[col].max()
                if c_max > c_min:
                    df_norm[col] = (df_norm[col] - c_min) / (c_max - c_min)
        
        # Color by RMSE (quantile)
        try:
            df_norm['quality'] = pd.qcut(df['value'], 4, labels=['Best', 'Good', 'Fair', 'Poor'])
            parallel_coordinates(df_norm[param_cols + ['quality']], 'quality', colormap='viridis', alpha=0.5)
            plt.title('Hyperparameter Interactions (Parallel Coordinates)', fontweight='bold')
            plt.savefig(output_dir / 'parallel_coordinates.png', dpi=300)
        except:
            pass
        plt.close()

    # 5. Summary Text Box
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    meta = data.get('study_metadata', {})
    best_params = meta.get('best_params', {})
    
    info = f"OPTUNA SUMMARY\n{'='*20}\n"
    info += f"Best Trial: #{meta.get('best_trial_number', 'N/A')}\n"
    info += f"Best RMSE: {meta.get('best_rmse', 0):.4f}\n\n"
    info += "BEST PARAMETERS SETTINGS:\n"
    for k, v in best_params.items():
        info += f"- {k}: {v}\n"
        
    ax.text(0.1, 0.9, info, family='monospace', verticalalignment='top', fontsize=12)
    plt.savefig(output_dir / 'optimization_timeline.png', dpi=300)
    plt.close()
