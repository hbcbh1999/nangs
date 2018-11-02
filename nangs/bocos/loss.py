import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, preds, outputs):
        return (preds - outputs).pow(2).mean()