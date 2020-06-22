import warnings
import torch
import torch.nn as nn


# Define the Linear loss function:
def linear_loss(input, target, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(),
                                                                      input.size()),
                      stacklevel=2)
    ret = (1 - input * target) / 2
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


# Define the Linear loss Module:
class LinearLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        self.reduction = reduction
        super(LinearLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return linear_loss(input, target, reduction=self.reduction)