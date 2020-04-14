import torch
import torch.nn as nn
from typing import Dict, List, Tuple


# All code should be compatible with the supported types mentioned here:
# https://pytorch.org/docs/master/jit_unsupported.html

class YOLODummy(nn.Module):
    def __init__(self, anchors, num_classes, image_size=416):
        """
        YOLO Layer - consistent with the darknet code at: https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c
                     and with the PyTorch-YOLOv3 code at: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

        :param anchors: A list of tuples - the masked anchors with the format: [(x_i,y_i),...]
        :param num_classes: Maximum number of classes that will be predicted from one grid cell
        :param image_size: The expected image size in pixels, input_size == image width == image height
        """

        super(YOLODummy, self).__init__()
        self.anchors = anchors  # Useful video explaining this: https://youtu.be/RTlwl2bv0Tg (thanks Andrew!)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.threshold = 0.5
        self.mse_loss = nn.MSELoss()  # For Bounding Box Prediction
        self.bce_loss = nn.BCELoss()  # For Class Prediction
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}
        self.image_size = image_size
        self.grid_size = image_size  # The grid should cover the image exactly
        self.stride = 1.0
        self.grid_x = torch.empty((len(anchors), 2), dtype=torch.float32)
        self.grid_y = torch.empty((len(anchors), 2), dtype=torch.float32)
        self.scaled_anchors = torch.empty((len(anchors), 2), dtype=torch.float32)
        self.anchor_w = torch.empty((len(anchors), 2), dtype=torch.float32)
        self.anchor_h = torch.empty((len(anchors), 2), dtype=torch.float32)

    def forward(self, x):
        """
        The forward function has the following steps: 1. Extract predictions from previous layer.
                                                      2. Create prediction bounding boxes.
                                                      3. Compare to ground truth bounding boxes, update metrics and loss

        :param x: The input, with shape: (batch_size, channels_size, image_size, image_size)
        :param y: The ground truth, with shape: (batch_size, predicted_boxes, predicted_confidence, predicted_class)
        :return: The YOLO output - (batch_size, predicted_boxes, predicted_confidence, predicted_class), layer_loss
        """

        return torch.zeros((4, 2, 416, 416)), 42.0