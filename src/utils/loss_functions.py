import torch
import torch.nn as nn

class sMAPELoss(nn.Module):
    def __init__(self):
        super(sMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-7 
        diff = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_true) + torch.abs(y_pred) + epsilon
        smape = 2.0 * torch.mean(diff / denominator) * 100
        return smape


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse_loss)
        return rmse


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-7
        diff = torch.abs((y_true - y_pred) / (y_true + epsilon))
        mape = torch.mean(diff) * 100
        return mape

