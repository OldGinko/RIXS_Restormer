import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch


class TrainingHistoryLogger:
    """Logger for training history with automatic plot generation.
    
    Similar to NAFNet's training history tracking, this logger:
    1. Tracks loss values during training
    2. Tracks validation metrics (PSNR, SSIM, MAE, MSE)
    3. Generates training history plots
    4. Saves progress to JSON files
    """
    
    def __init__(self, log_dir, experiment_name, target_mae=0.001, target_mse=0.000001):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.target_mae = target_mae
        self.target_mse = target_mse
        
        # Storage for metrics
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.epochs = []
        self.iterations = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # File paths
        self.history_file = os.path.join(log_dir, f'{experiment_name}_training_history.json')
        self.plot_file = os.path.join(log_dir, f'{experiment_name}_training_history.png')
        
        print(f"Training history will be saved to: {self.history_file}")
        print(f"Training plots will be saved to: {self.plot_file}")
        print(f"Target MAE: {target_mae}, Target MSE: {target_mse}")
    
    def log_training_step(self, epoch, iteration, losses_dict):
        """Log training step metrics.
        
        Args:
            epoch (int): Current epoch
            iteration (int): Current iteration
            losses_dict (dict): Dictionary containing loss values
                e.g., {'l_pix': 0.1, 'l_total': 0.1, 'mae': 0.001, 'mse': 0.000001}
        """
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        
        if iteration not in self.iterations:
            self.iterations.append(iteration)
        
        # Store all loss values
        for key, value in losses_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.train_history[key].append(value)
        
        # Store epoch and iteration for plotting
        self.train_history['epoch'].append(epoch)
        self.train_history['iteration'].append(iteration)
    
    def log_validation_step(self, epoch, iteration, metrics_dict):
        """Log validation step metrics.
        
        Args:
            epoch (int): Current epoch
            iteration (int): Current iteration  
            metrics_dict (dict): Dictionary containing validation metrics
                e.g., {'psnr': 45.2, 'ssim': 0.987, 'mae': 0.001, 'mse': 0.000001}
        """
        # Store all validation metrics
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.val_history[key].append(value)
        
        # Store epoch and iteration for plotting
        self.val_history['epoch'].append(epoch)
        self.val_history['iteration'].append(iteration)
        
        # Check target achievement
        if 'mae' in metrics_dict and 'mse' in metrics_dict:
            mae_achieved = metrics_dict['mae'] <= self.target_mae
            mse_achieved = metrics_dict['mse'] <= self.target_mse
            
            if mae_achieved and mse_achieved:
                print(f"🎯 TARGET ACHIEVED! Epoch {epoch}, Iter {iteration}")
                print(f"   MAE: {metrics_dict['mae']:.6f} <= {self.target_mae}")
                print(f"   MSE: {metrics_dict['mse']:.6f} <= {self.target_mse}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_data = {
            'experiment_name': self.experiment_name,
            'target_mae': self.target_mae,
            'target_mse': self.target_mse,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'total_epochs': len(self.epochs),
            'total_iterations': len(self.iterations)
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved to: {self.history_file}")
    
    def generate_training_plots(self, save_plot=True, show_plot=False):
        """Generate training history plots like NAFNet.
        
        Creates a plot similar to the one you showed with:
        - Training loss over epochs
        - Target lines for MAE/MSE
        - Clean, professional styling
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History: {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss over Epochs
        if 'l_pix' in self.train_history or 'l_total' in self.train_history:
            ax1 = axes[0, 0]
            
            # Use total loss if available, otherwise pixel loss
            loss_key = 'l_total' if 'l_total' in self.train_history else 'l_pix'
            epochs = self.train_history['epoch']
            losses = self.train_history[loss_key]
            
            ax1.plot(epochs, losses, 'b-', linewidth=2, label=f'{loss_key}')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Validation Loss', fontsize=12)
            ax1.set_title('Training History', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Style similar to your attached image
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
        
        # Plot 2: Validation PSNR over Epochs
        if 'psnr' in self.val_history:
            ax2 = axes[0, 1]
            epochs = self.val_history['epoch']
            psnr_values = self.val_history['psnr']
            
            ax2.plot(epochs, psnr_values, 'g-', linewidth=2, label='PSNR')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('PSNR (dB)', fontsize=12)
            ax2.set_title('PSNR Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: MAE Evolution with Target Line
        if 'mae' in self.val_history:
            ax3 = axes[1, 0]
            epochs = self.val_history['epoch']
            mae_values = self.val_history['mae']
            
            ax3.plot(epochs, mae_values, 'r-', linewidth=2, label='MAE')
            ax3.axhline(y=self.target_mae, color='r', linestyle='--', alpha=0.7, 
                       label=f'Target MAE = {self.target_mae}')
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('MAE', fontsize=12)
            ax3.set_title('MAE Evolution (Alpha Target)', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_yscale('log')  # Log scale for better visibility
        
        # Plot 4: MSE Evolution with Target Line
        if 'mse' in self.val_history:
            ax4 = axes[1, 1]
            epochs = self.val_history['epoch']
            mse_values = self.val_history['mse']
            
            ax4.plot(epochs, mse_values, 'm-', linewidth=2, label='MSE')
            ax4.axhline(y=self.target_mse, color='m', linestyle='--', alpha=0.7,
                       label=f'Target MSE = {self.target_mse}')
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('MSE', fontsize=12)
            ax4.set_title('MSE Evolution (Alpha Target)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_yscale('log')  # Log scale for better visibility
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plot_file, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to: {self.plot_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def generate_single_loss_plot(self, save_plot=True, show_plot=False):
        """Generate a single training loss plot like your attached image."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Use total loss if available, otherwise pixel loss
        loss_key = 'l_total' if 'l_total' in self.train_history else 'l_pix'
        epochs = self.train_history['epoch']
        losses = self.train_history[loss_key]
        
        # Plot with the same style as your image
        ax.plot(epochs, losses, color='#1f77b4', linewidth=2, label=f'k=(7,7)')
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('val_loss', fontsize=14)
        ax.set_title(f'Training History: k=(7,7)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)
        
        # Clean styling like your image
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            single_plot_file = os.path.join(self.log_dir, f'{self.experiment_name}_loss_history.png')
            plt.savefig(single_plot_file, dpi=300, bbox_inches='tight')
            print(f"Loss history plot saved to: {single_plot_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def print_progress_summary(self):
        """Print current training progress summary."""
        if not self.val_history:
            print("No validation history available yet.")
            return
        
        latest_epoch = self.val_history['epoch'][-1] if self.val_history['epoch'] else 0
        
        print(f"\n📊 Training Progress Summary - {self.experiment_name}")
        print(f"Current Epoch: {latest_epoch}")
        
        if 'psnr' in self.val_history:
            latest_psnr = self.val_history['psnr'][-1]
            print(f"Latest PSNR: {latest_psnr:.2f} dB")
        
        if 'mae' in self.val_history:
            latest_mae = self.val_history['mae'][-1]
            mae_progress = (self.target_mae / latest_mae) * 100 if latest_mae > 0 else 0
            print(f"Latest MAE: {latest_mae:.6f} (Target: {self.target_mae}, Progress: {mae_progress:.1f}%)")
        
        if 'mse' in self.val_history:
            latest_mse = self.val_history['mse'][-1]
            mse_progress = (self.target_mse / latest_mse) * 100 if latest_mse > 0 else 0
            print(f"Latest MSE: {latest_mse:.8f} (Target: {self.target_mse}, Progress: {mse_progress:.1f}%)")
        
        print("-" * 50)
