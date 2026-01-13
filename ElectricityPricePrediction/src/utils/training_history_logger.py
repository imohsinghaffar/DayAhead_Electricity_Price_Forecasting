"""
Training History Logger
Custom Keras callback to log and visualize LSTM training progress.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import tensorflow as tf


class TrainingHistoryLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log training history and generate visualizations.
    
    Args:
        log_dir: Directory to save logs and plots
        model_name: Name of the model being trained
        run_id: Optional run identifier (auto-generated if not provided)
    """
    
    def __init__(self, log_dir="training_logs", model_name="LSTM", run_id=None):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training metrics storage
        self.history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'lr': []
        }
        
        # Metadata
        self.metadata = {
            'model_name': model_name,
            'run_id': self.run_id,
            'start_time': None,
            'end_time': None,
            'total_epochs': 0,
            'stopped_early': False,
            'best_epoch': 0
        }
        
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.metadata['start_time'] = datetime.now().isoformat()
        print(f"\n{'='*60}")
        print(f"Training Started: {self.model_name} | Run ID: {self.run_id}")
        print(f"{'='*60}\n")
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Store metrics
        self.history['epoch'].append(epoch + 1)
        self.history['loss'].append(logs.get('loss', np.nan))
        self.history['val_loss'].append(logs.get('val_loss', np.nan))
        self.history['mae'].append(logs.get('mae', np.nan))
        self.history['val_mae'].append(logs.get('val_mae', np.nan))
        
        # Get learning rate - TensorFlow 2.x compatible
        try:
            # Method 1: Try optimizer.learning_rate (TF 2.x)
            if hasattr(self.model.optimizer, 'learning_rate'):
                lr_value = self.model.optimizer.learning_rate
                if callable(lr_value):
                    # It's a schedule, get current step
                    lr = float(lr_value(self.model.optimizer.iterations))
                else:
                    lr = float(tf.keras.backend.get_value(lr_value))
            # Method 2: Try optimizer.lr (older TF)
            elif hasattr(self.model.optimizer, 'lr'):
                lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            else:
                lr = 0.0
        except Exception as e:
            # Fallback: get from logs if available
            lr = logs.get('lr', logs.get('learning_rate', 0.0))
        self.history['lr'].append(float(lr) if lr else 0.0)
        
        # Print progress
        print(f"Epoch {epoch + 1:3d} | "
              f"Loss: {logs.get('loss', 0):.4f} | "
              f"Val Loss: {logs.get('val_loss', 0):.4f} | "
              f"MAE: {logs.get('mae', 0):.4f} | "
              f"Val MAE: {logs.get('val_mae', 0):.4f} | "
              f"LR: {lr:.2e}")
        
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_epochs'] = len(self.history['epoch'])
        
        # Determine if early stopping occurred
        if hasattr(self.model, 'stop_training') and self.model.stop_training:
            self.metadata['stopped_early'] = True
        
        # Find best epoch (lowest val_loss)
        try:
            val_losses = [v for v in self.history['val_loss'] if not np.isnan(v)]
            if val_losses:
                best_idx = np.argmin(val_losses)
                self.metadata['best_epoch'] = self.history['epoch'][best_idx]
        except:
            self.metadata['best_epoch'] = self.metadata['total_epochs']
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Total Epochs: {self.metadata['total_epochs']}")
        print(f"Best Epoch: {self.metadata['best_epoch']}")
        print(f"Early Stopping: {self.metadata['stopped_early']}")
        print(f"{'='*60}\n")
        
        # Save history
        self._save_history()
        
        # Generate plots
        self._generate_plots()
        
    def _save_history(self):
        """Save training history to JSON."""
        output = {
            'metadata': self.metadata,
            'history': self.history
        }
        
        json_path = self.log_dir / f"{self.model_name.lower()}_training_history_{self.run_id}.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        print(f"✓ Saved training history: {json_path}")
        
    def _generate_plots(self):
        """Generate comprehensive training visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.model_name} Training History | Run: {self.run_id}', 
                     fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        best_epoch = self.metadata['best_epoch']
        
        # Panel 1: Loss Curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['loss'], 'b-', linewidth=2, label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        ax1.axvline(best_epoch, color='green', linestyle='--', linewidth=2, 
                   label=f'Best Epoch ({best_epoch})')
        ax1.set_title('Model Loss', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Huber)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: MAE Curves
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['mae'], 'b-', linewidth=2, label='Training MAE')
        ax2.plot(epochs, self.history['val_mae'], 'r-', linewidth=2, label='Validation MAE')
        ax2.axvline(best_epoch, color='green', linestyle='--', linewidth=2,
                   label=f'Best Epoch ({best_epoch})')
        ax2.set_title('Mean Absolute Error', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Learning Rate Schedule
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['lr'], 'g-', linewidth=2, marker='o', markersize=4)
        ax3.set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Training Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        final_train_loss = self.history['loss'][-1] if self.history['loss'] else np.nan
        final_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else np.nan
        best_val_loss = min([v for v in self.history['val_loss'] if not np.isnan(v)], default=np.nan)
        
        summary_text = f"""
        TRAINING SUMMARY
        {'─' * 40}
        Total Epochs:        {self.metadata['total_epochs']}
        Best Epoch:          {best_epoch}
        Early Stopping:      {self.metadata['stopped_early']}
        
        FINAL METRICS
        {'─' * 40}
        Training Loss:       {final_train_loss:.4f}
        Validation Loss:     {final_val_loss:.4f}
        Best Val Loss:       {best_val_loss:.4f}
        
        Training MAE:        {self.history['mae'][-1]:.4f}
        Validation MAE:      {self.history['val_mae'][-1]:.4f}
        
        IMPROVEMENT
        {'─' * 40}
        Loss Reduction:      {((self.history['loss'][0] - final_train_loss) / self.history['loss'][0] * 100):.1f}%
        Val Loss Reduction:  {((self.history['val_loss'][0] - best_val_loss) / self.history['val_loss'][0] * 100):.1f}%
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / f"{self.model_name.lower()}_training_curves_{self.run_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves: {plot_path}")
        plt.close()
        
    def get_history_dict(self):
        """Return training history as dictionary."""
        return {
            'metadata': self.metadata,
            'history': self.history
        }
