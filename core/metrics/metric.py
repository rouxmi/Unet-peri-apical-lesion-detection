import torch
import torch.nn as nn

class MeanDiceScore(nn.Module):
    """Calculates the mean dice score.

    Args:
        softmax (bool): Whether to apply softmax to the inputs. Default is True.
        weights (torch.Tensor): Weights to be applied to each class. Default is None.
        epsilon (float): Small value added to the denominator to avoid division by zero. Default is 1.e-5.
    """

    def __init__(self, softmax=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.softmax = softmax
        self.weights = weights
        self.eps = epsilon

    def forward(self, inputs, targets):
        """
        Calculates the mean dice score.

        Args:
            inputs (torch.Tensor): The predicted segmentation masks.
            targets (torch.Tensor): The ground truth segmentation masks.

        Returns:
            torch.Tensor: The mean dice score.
        """

        if self.softmax:
            inputs = nn.Softmax(dim=1)(inputs)

        if self.weights == None:
            self.weights = torch.ones(inputs.shape[1])
        w = self.weights[None, :, None, None]
        w = w.to(inputs.device)

        num = 2 * torch.sum(inputs * targets * w, dim=(1, 2, 3))
        den = torch.sum((inputs + targets) * w, dim=(1, 2, 3)) + self.eps

        return torch.mean(num/den)