import torch.nn as nn

from src.YOLODummy import YOLODummy
from src.YOLOLayer import YOLOLayer
from src.YOLOModule import YOLOModule


# All code should be compatible with the supported types mentioned here:
# https://pytorch.org/docs/master/jit_unsupported.html


class YOLOv3(YOLOModule):
    def __init__(self):
        super(YOLOv3, self).__init__()
        # self.yolo_layer = YOLOLayer(anchors=[(2, 4), (6, 8)], num_classes=1)
        self.yolo_layer = YOLODummy(anchors=[(2, 4), (6, 8)], num_classes=1)

    def forward(self, x):
        x = self.yolo_layer(x)
        return x
