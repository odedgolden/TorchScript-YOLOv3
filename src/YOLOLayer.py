import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.YOLOModule import YOLOModule


# All code should be compatible with the supported types mentioned here:
# https://pytorch.org/docs/master/jit_unsupported.html


class YOLOLayer(YOLOModule):
    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, image_size=416):
        """
        YOLO Layer - consistent with the darknet code at: https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c
                     and with the PyTorch-YOLOv3 code at: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

        :param anchors: A list of tuples - the masked anchors with the format: [(x_i,y_i),...]
        :param num_classes: Maximum number of classes that will be predicted from one grid cell
        :param image_size: The expected image size in pixels, input_size == image width == image height
        """

        super(YOLOLayer, self).__init__()
        self.anchors = anchors  # Useful video explaining this: https://youtu.be/RTlwl2bv0Tg (thanks Andrew!)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.threshold = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}
        self.image_size = image_size
        self.grid_size = image_size  # The grid should cover the image exactly
        self.stride = 1.0
        self.grid_x = 0
        self.grid_y = 0
        self.anchor_w = 0
        self.anchor_h = 0

    def forward(self, x, y=torch.tensor([42], dtype=torch.float32)):
        """
        The forward function has the following steps: 1. Extract predictions from previous layer.
                                                      2. Create prediction bounding boxes.
                                                      3. Compare to ground truth bounding boxes, update metrics and loss

        :param x: The input, with shape: (batch_size, channels_size, image_size, image_size)
        :param y: The ground truth, with shape: (batch_size, predicted_boxes, predicted_confidence, predicted_class)
        :return: The YOLO output - (batch_size, predicted_boxes, predicted_confidence, predicted_class), layer_loss
        """

        batch_size = x.size(0)
        grid_size = x.size(2)  # that is - the image size

        # In order to get the predictions we only reorganizing the input tensor, replacing the class prediction to be
        # the last dimension
        prediction = (
            # Reshape tensor after convolution layers
            x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)  # Class prediction should be the last dimension
                .contiguous()  # Make the actual changes inplace
        )

        # Extract predictions for bounding boxes, confidence and class
        # We use sigmoid in order to make sure the values are between (0,1)
        x_0 = torch.sigmoid(prediction[..., 0])  # Center x
        y_0 = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            grid_x, grid_y, anchor_w, anchor_h = self.update_grid(self.image_size, grid_size, self.anchors, x.is_cuda)
            # self.grid_x = grid_x
            # self.grid_y = grid_y
            # self.anchor_w = anchor_w
            # self.anchor_h = anchor_h

        # Add offset and scale with anchors
        pred_boxes = torch.tensor(prediction[..., :4].shape, dtype=torch.float32)
        if x.is_cuda:
            pred_boxes.cuda()
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # For output we only concatenating
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4) * self.stride,
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )

        return output

