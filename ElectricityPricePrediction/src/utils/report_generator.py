import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    """Generates HTML and JSON reports for the forecasting run."""
    
    def __init__(self, run_id, output_dir):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }

    def add_model_info(self, model_name, metrics, n_points):
        self.metadata["models"][model_name] = {
            "metrics": metrics,
            "n_points": n_points
        }

    def generate(self, metrics_df, optuna_history=None):
        # 1. Save JSON Report
        report_json = self.output_dir / "report.json"
        with open(report_json, 'w') as f:
            json.dump(self.metadata, f, indent=4)
            
        # 2. Build HTML Body
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .plot {{ max-width: 48%; border: 1px solid #ccc; }}
                .summary {{ background: #e7f3fe; padding: 15px; border-left: 6px solid #2196F3; margin: 20px 0; }}
            </style>
            <title>Forecast Report - {self.run_id}</title>
        </head>
        <body>
            <h1>Forecast Run Report: {self.run_id}</h1>
            <div class="summary">
                <strong>Run Timestamp:</strong> {self.metadata['timestamp']}<br>
                <strong>Status:</strong> Completed Successfully
            </div>

            <h2>Model Metrics</h2>
            {metrics_df.to_html(classes='table', index=False)}

            <h2>Forecast Analysis</h2>
            <div class="plot-container">
                <div>
                    <h3>Forecasting</h3>
                    <img src="Plots/Forecasting/comprehensive_analysis.png" class="plot" style="max-width: 100%;">
                </div>
                <div>
                    <h3>Error Analysis</h3>
                    <img src="Plots/Error_Analysis/forecasting_insights.png" class="plot" style="max-width: 100%;">
                </div>
                <div>
                    <h3>Features</h3>
                    <img src="Plots/Features/feature_importance.png" class="plot" style="max-width: 100%;">
                </div>
            </div>
            
            <h2>Sequential Model Logs</h2>
            <p>Persistent logs for each model are saved in <code>Analysis/Model_History/&lt;ModelName&gt;/{self.metadata.get('run_number', 'N')}.json</code></p>
        """
        
        if optuna_history:
            meta = optuna_history.get("study_metadata", {})
            imp = meta.get("improvement", {})
            html_content += f"""
            <h2>Optuna Tuning Summary</h2>
            <p><strong>Baseline RMSE:</strong> {meta.get('baseline_rmse', 'N/A')}</p>
            <p><strong>Best RMSE:</strong> {meta.get('best_rmse', 'N/A')}</p>
            <p style="color: green;"><strong>Improvement:</strong> {imp.get('rmse_pct', 0):.2f}%</p>
            <div class="plot-container">
                <img src="Optuna/XGBoost/Plots/trial_progression.png" class="plot">
                <img src="Optuna/XGBoost/Plots/hyperparameter_importance.png" class="plot">
                <img src="Optuna/XGBoost/Plots/optimization_timeline.png" class="plot">
            </div>
            """

        html_content += """
            <p style="margin-top: 50px; color: #888;">&copy; Antigravity Forecasting System 2026</p>
        </body>
        </html>
        """
        
        report_html = self.output_dir / "report.html"
        with open(report_html, 'w') as f:
            f.write(html_content)
        
        return report_html, report_json
