import torch.nn as nn

from src.YOLOLayer import YOLOLayer


class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.yolo_layer = YOLOLayer(anchors=[(2, 4), (6, 8)], num_classes=1)

    def forward(self, x):
        x = x.yolo_layer(x)
        return x
