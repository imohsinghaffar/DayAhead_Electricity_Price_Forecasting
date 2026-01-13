#!/usr/bin/env python3
"""
Interactive Visualizations Module (Plotly)
Generates interactive HTML-based plots for model comparison.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path


def create_interactive_forecast(predictions, output_dir):
    """Creates an interactive multi-model forecast comparison plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to last 7 days for clarity
    plot_len = min(len(predictions), 7*24)
    data = predictions.iloc[-plot_len:].reset_index()
    data.rename(columns={'index': 'Date'}, inplace=True)
    
    fig = go.Figure()
    
    # Add Actual
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Actual'],
        name='Actual', line=dict(color='black', width=2)
    ))
    
    # Add all model predictions
    exclude = ['Actual', 'LSTM_uncertainty', 'P05', 'P95', 'P50', 'Date']
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate([c for c in data.columns if c not in exclude]):
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data[col],
            name=col, line=dict(width=1.5),
            opacity=0.8
        ))
    
    fig.update_layout(
        title='Interactive Forecast Comparison (Last 7 Days)',
        xaxis_title='Date',
        yaxis_title='Price (EUR/MWh)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        template='plotly_white',
        xaxis=dict(rangeslider=dict(visible=True), type='date')
    )
    
    # Save interactive HTML
    fig.write_html(output_dir / "interactive_forecast.html")
    return output_dir / "interactive_forecast.html"


def create_interactive_error_analysis(predictions, output_dir):
    """Creates interactive error distribution and hourly MAE plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=['Error Distribution', 'MAE by Hour'])
    
    exclude = ['Actual', 'LSTM_uncertainty', 'P05', 'P95', 'P50']
    
    for col in [c for c in predictions.columns if c not in exclude]:
        errors = predictions[col] - predictions['Actual']
        fig.add_trace(go.Histogram(x=errors, name=col, opacity=0.6), row=1, col=1)
    
    # MAE by Hour
    preds_copy = predictions.copy()
    preds_copy['hour'] = preds_copy.index.hour
    for col in [c for c in predictions.columns if c not in exclude]:
        mae_hour = (preds_copy[col] - preds_copy['Actual']).abs().groupby(preds_copy['hour']).mean()
        fig.add_trace(go.Scatter(x=mae_hour.index, y=mae_hour.values, name=f'{col} MAE', mode='lines+markers'), row=2, col=1)
    
    fig.update_layout(
        title='Interactive Error Analysis',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.write_html(output_dir / "interactive_error_analysis.html")
    return output_dir / "interactive_error_analysis.html"
