from torch import nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # For Bounding Box Prediction
        self.bce_loss = nn.BCELoss()  # For Class Prediction

    def forward(self, x, y):
        return 