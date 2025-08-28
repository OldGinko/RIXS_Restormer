import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.losses.loss_util import weighted_loss


class AlphaMAEMSELoss(nn.Module):
    """Alpha-weighted MAE/MSE loss exactly like NAFNet.
    
    The alpha parameter controls the ratio between MAE and MSE:
    Loss = alpha * MAE + (1 - alpha) * MSE
    
    Args:
        alpha (float): Weight for MAE vs MSE. Range [0, 1]. Default: 0.5.
            - alpha = 0.0: Pure MSE loss
            - alpha = 1.0: Pure MAE loss  
            - alpha = 0.5: Equal weighting
        loss_weight (float): Overall loss weight. Default: 1.0.
        reduction (str): Reduction method. Default: 'mean'.
    """

    def __init__(self, alpha=0.5, loss_weight=1.0, reduction='mean'):
        super(AlphaMAEMSELoss, self).__init__()
        if not 0 <= alpha <= 1:
            raise ValueError(f'Alpha must be between 0 and 1, got {alpha}')
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')

        self.alpha = alpha
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        # Track current values for monitoring
        self.current_mae = float('inf')
        self.current_mse = float('inf')
        self.current_loss = float('inf')

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): Predicted images of shape (N, C, H, W).
            target (Tensor): Target images of shape (N, C, H, W).
            weight (Tensor, optional): Element-wise weights.
        """
        # Calculate MAE (L1 loss)
        mae = F.l1_loss(pred, target, reduction='none')
        
        # Calculate MSE (L2 loss)
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Alpha-weighted combination
        combined_loss = self.alpha * mae + (1 - self.alpha) * mse
        
        # Update current values for monitoring
        self.current_mae = mae.mean().item()
        self.current_mse = mse.mean().item()
        self.current_loss = combined_loss.mean().item()
        
        # Apply element-wise weight if provided
        if weight is not None:
            assert weight.dim() == combined_loss.dim()
            assert weight.size(1) == 1 or weight.size(1) == combined_loss.size(1)
            combined_loss = combined_loss * weight
        
        # Apply reduction
        if self.reduction == 'none':
            return self.loss_weight * combined_loss
        elif self.reduction == 'mean':
            return self.loss_weight * combined_loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * combined_loss.sum()

    def get_alpha_info(self):
        """Get current alpha weighting information (for logging)."""
        mae_contribution = self.alpha * self.current_mae
        mse_contribution = (1 - self.alpha) * self.current_mse
        
        return {
            'alpha': self.alpha,
            'current_mae': self.current_mae,
            'current_mse': self.current_mse,
            'current_loss': self.current_loss,
            'mae_contribution': mae_contribution,
            'mse_contribution': mse_contribution,
            'mae_weight': self.alpha,
            'mse_weight': 1 - self.alpha
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'alpha={self.alpha}, '
                f'loss_weight={self.loss_weight}, '
                f'reduction={self.reduction})')


class AdaptiveAlphaMAEMSELoss(nn.Module):
    """Adaptive alpha MAE/MSE loss that changes alpha during training.
    
    This can start with one alpha value and gradually transition to another,
    allowing the model to focus on different aspects during training.
    
    Args:
        start_alpha (float): Initial alpha value. Default: 1.0 (pure MAE).
        end_alpha (float): Final alpha value. Default: 0.0 (pure MSE).
        transition_epochs (int): Number of epochs to transition. Default: 50.
        loss_weight (float): Overall loss weight. Default: 1.0.
        reduction (str): Reduction method. Default: 'mean'.
    """

    def __init__(self, start_alpha=1.0, end_alpha=0.0, transition_epochs=50, 
                 loss_weight=1.0, reduction='mean'):
        super(AdaptiveAlphaMAEMSELoss, self).__init__()
        
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.transition_epochs = transition_epochs
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        self.current_epoch = 0
        self.current_alpha = start_alpha
        
        # Track current values
        self.current_mae = float('inf')
        self.current_mse = float('inf')
        self.current_loss = float('inf')

    def update_epoch(self, epoch):
        """Update the current epoch to adjust alpha."""
        self.current_epoch = epoch
        
        if epoch >= self.transition_epochs:
            self.current_alpha = self.end_alpha
        else:
            # Linear interpolation between start and end alpha
            progress = epoch / self.transition_epochs
            self.current_alpha = (1 - progress) * self.start_alpha + progress * self.end_alpha

    def forward(self, pred, target, weight=None):
        """Forward pass with current alpha."""
        # Calculate MAE and MSE
        mae = F.l1_loss(pred, target, reduction='none')
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Alpha-weighted combination with current alpha
        combined_loss = self.current_alpha * mae + (1 - self.current_alpha) * mse
        
        # Update current values
        self.current_mae = mae.mean().item()
        self.current_mse = mse.mean().item()
        self.current_loss = combined_loss.mean().item()
        
        # Apply element-wise weight if provided
        if weight is not None:
            combined_loss = combined_loss * weight
        
        # Apply reduction
        if self.reduction == 'none':
            return self.loss_weight * combined_loss
        elif self.reduction == 'mean':
            return self.loss_weight * combined_loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * combined_loss.sum()

    def get_alpha_info(self):
        """Get current alpha information."""
        return {
            'current_alpha': self.current_alpha,
            'start_alpha': self.start_alpha,
            'end_alpha': self.end_alpha,
            'current_epoch': self.current_epoch,
            'transition_progress': min(1.0, self.current_epoch / self.transition_epochs),
            'current_mae': self.current_mae,
            'current_mse': self.current_mse,
            'current_loss': self.current_loss,
            'mae_weight': self.current_alpha,
            'mse_weight': 1 - self.current_alpha
        }
