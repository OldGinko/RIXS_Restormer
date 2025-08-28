from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss)
from .target_mae_mse_loss import (TargetMAEMSELoss, SimpleTargetMSELoss, SimpleTargetMAELoss)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss',
    'TargetMAEMSELoss', 'SimpleTargetMSELoss', 'SimpleTargetMAELoss',
]
