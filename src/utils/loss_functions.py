import torch
import torch.nn as nn

# Define sMAPE loss function
class sMAPELoss(nn.Module):
    def __init__(self):
        super(sMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-7  # To avoid division by zero
        diff = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_true) + torch.abs(y_pred) + epsilon
        smape = 2.0 * torch.mean(diff / denominator) * 100
        return smape