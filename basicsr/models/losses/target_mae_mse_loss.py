import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.losses.loss_util import weighted_loss


class TargetMAEMSELoss(nn.Module):
    """Combined L1 loss with target MAE/MSE monitoring and adaptive weighting.
    
    Similar to NAFNet's alpha parameter approach, this allows setting target
    MAE/MSE values that the training should aim for.
    
    Args:
        l1_weight (float): Weight for L1 loss. Default: 1.0.
        target_mae (float): Target MAE value to achieve (like alpha in NAFNet). Default: 0.001.
        target_mse (float): Target MSE value to achieve. Default: 0.000001.
        mae_weight (float): Weight for MAE component. Default: 0.1.
        mse_weight (float): Weight for MSE component. Default: 0.1.
        adaptive_weight (bool): Whether to adaptively adjust weights based on target achievement. Default: True.
        reduction (str): Reduction method. Default: 'mean'.
    """

    def __init__(self, 
                 l1_weight=1.0, 
                 target_mae=0.001, 
                 target_mse=0.000001,
                 mae_weight=0.1, 
                 mse_weight=0.1,
                 adaptive_weight=True,
                 reduction='mean'):
        super(TargetMAEMSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.l1_weight = l1_weight
        self.target_mae = target_mae
        self.target_mse = target_mse
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.adaptive_weight = adaptive_weight
        self.reduction = reduction
        
        # Track training progress
        self.iteration_count = 0
        self.current_mae = float('inf')
        self.current_mse = float('inf')

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): Predicted images of shape (N, C, H, W).
            target (Tensor): Target images of shape (N, C, H, W).
            weight (Tensor, optional): Element-wise weights.
        """
        self.iteration_count += 1
        
        # Calculate L1 loss (primary loss)
        l1_loss = F.l1_loss(pred, target, reduction='none')
        
        # Calculate MAE and MSE for monitoring and adaptive weighting
        mae = torch.mean(torch.abs(pred - target))
        mse = torch.mean((pred - target) ** 2)
        
        # Update current values for monitoring
        self.current_mae = mae.item()
        self.current_mse = mse.item()
        
        # Adaptive weighting based on target achievement
        if self.adaptive_weight:
            # Increase MAE weight if above target, decrease if below
            mae_adaptive = self.mae_weight * max(1.0, mae.item() / self.target_mae)
            mse_adaptive = self.mse_weight * max(1.0, mse.item() / self.target_mse)
        else:
            mae_adaptive = self.mae_weight
            mse_adaptive = self.mse_weight
        
        # Combine losses
        total_loss = (self.l1_weight * l1_loss + 
                     mae_adaptive * mae + 
                     mse_adaptive * mse)
        
        # Apply element-wise weight if provided
        if weight is not None:
            assert weight.dim() == total_loss.dim()
            assert weight.size(1) == 1 or weight.size(1) == total_loss.size(1)
            total_loss = total_loss * weight
        
        # Apply reduction
        if self.reduction == 'none':
            return total_loss
        elif self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()

    def get_progress_info(self):
        """Get current progress towards targets (for logging)."""
        mae_progress = min(1.0, self.target_mae / max(self.current_mae, 1e-8))
        mse_progress = min(1.0, self.target_mse / max(self.current_mse, 1e-8))
        
        return {
            'iteration': self.iteration_count,
            'current_mae': self.current_mae,
            'current_mse': self.current_mse,
            'target_mae': self.target_mae,
            'target_mse': self.target_mse,
            'mae_progress': mae_progress,  # 1.0 = target achieved
            'mse_progress': mse_progress   # 1.0 = target achieved
        }


class SimpleTargetMSELoss(nn.Module):
    """Simple MSE loss with target monitoring (like NAFNet alpha).
    
    Args:
        target_mse (float): Target MSE value (alpha equivalent). Default: 0.000001.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    
    def __init__(self, target_mse=0.000001, loss_weight=1.0, reduction='mean'):
        super(SimpleTargetMSELoss, self).__init__()
        self.target_mse = target_mse
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.current_mse = float('inf')
        
    def forward(self, pred, target, weight=None):
        mse_loss = F.mse_loss(pred, target, reduction='none')
        self.current_mse = mse_loss.mean().item()
        
        if weight is not None:
            mse_loss = mse_loss * weight
            
        if self.reduction == 'mean':
            return self.loss_weight * mse_loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * mse_loss.sum()
        else:
            return self.loss_weight * mse_loss
    
    def target_achieved(self):
        """Check if target MSE is achieved."""
        return self.current_mse <= self.target_mse


class SimpleTargetMAELoss(nn.Module):
    """Simple MAE loss with target monitoring (like NAFNet alpha).
    
    Args:
        target_mae (float): Target MAE value (alpha equivalent). Default: 0.001.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    
    def __init__(self, target_mae=0.001, loss_weight=1.0, reduction='mean'):
        super(SimpleTargetMAELoss, self).__init__()
        self.target_mae = target_mae
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.current_mae = float('inf')
        
    def forward(self, pred, target, weight=None):
        mae_loss = F.l1_loss(pred, target, reduction='none')
        self.current_mae = mae_loss.mean().item()
        
        if weight is not None:
            mae_loss = mae_loss * weight
            
        if self.reduction == 'mean':
            return self.loss_weight * mae_loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * mae_loss.sum()
        else:
            return self.loss_weight * mae_loss
    
    def target_achieved(self):
        """Check if target MAE is achieved."""
        return self.current_mae <= self.target_mae
