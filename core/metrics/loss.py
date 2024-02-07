import torch.nn as nn
from core.metrics.metric import MeanDiceScore

class MeanDiceLoss(nn.Module):
    """Calculates the mean dice loss.

    Args:
        softmax (bool, optional): Whether to apply softmax to the inputs. Defaults to True.
        weights (list, optional): Weights for each class. Defaults to None.
        epsilon (float, optional): Small value added to the denominator for numerical stability. Defaults to 1.e-5.
    """

    def __init__(self, softmax=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.dice = MeanDiceScore(softmax, weights, epsilon)

    def forward(self, inputs, targets):
        """Forward pass of the mean dice loss.

        Args:
            inputs (torch.Tensor): Predicted segmentation masks.
            targets (torch.Tensor): Ground truth segmentation masks.

        Returns:
            torch.Tensor: Mean dice loss.
        """

        dice_score = self.dice(inputs, targets)

        return 1 - dice_score
    
class CombinedLoss(nn.Module):
    """
    Combined loss function that combines Dice loss and Cross Entropy loss.

    Args:
        dice_weight (float): Weight for the Dice loss. Default is 9.0.
        ce_weight (float): Weight for the Cross Entropy loss. Default is 1.0.
        softmax (bool): Flag indicating whether to apply softmax to the inputs. Default is True.
        weights (Tensor): Optional tensor of weights to apply to the Cross Entropy loss. Default is None.
        epsilon (float): Small value added to the denominator for numerical stability. Default is 1.e-5.
    """

    def __init__(self, dice_weight=9.0, ce_weight=1.0, softmax=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.dice = MeanDiceScore(softmax, weights, epsilon)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        """
        Forward pass of the combined loss function.

        Args:
            inputs (Tensor): Input tensor.
            targets (Tensor): Target tensor.

        Returns:
            Tensor: Weighted loss value.
        """
        dice_score = self.dice(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)

        weighted_loss = self.dice_weight * (1 - dice_score) + self.ce_weight * ce_loss

        return weighted_loss
