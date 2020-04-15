import torch
from torch import nn
from typing import Dict, List, Tuple


# All code should be compatible with the supported types mentioned here:
# https://pytorch.org/docs/master/jit_unsupported.html


class YOLOModule(nn.Module):

    @staticmethod
    def update_grid(image_size: int, new_grid_size: int, anchors: List[Tuple[int, int]], cuda: bool):
        """


        :param anchors:
        :param image_size:
        :param new_grid_size: New grid size derived from the actual input size - i.e. new_grid_size=416
        :param cuda: Is Cuda available
        :return:
        """
        grid_size = int(new_grid_size.item())  # Update grid size
        stride = image_size / grid_size  # Calculate stride

        # Calculate offsets for each grid
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])

        s_anchors: List[List[float, float]] = []
        for a_w, a_h in anchors:
            s_anchors.append([a_w / stride, a_h / stride])
        scaled_anchors = torch.tensor(s_anchors)

        if cuda:
            scaled_anchors.cuda()
        anchor_w = scaled_anchors[:, 0:1].view((1, len(anchors), 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, len(anchors), 1, 1))

        return grid_x, grid_y, anchor_w, anchor_h

    @staticmethod
    def to_cpu(tensor):
        """

        :param tensor: Tensor with a single number
        :return: The number in the tensor
        """
        return tensor.detach().cpu().item()

    @staticmethod
    def box_iou(box1_tensor, box2_tensor):
        """
        Calculating Intersection over Union of the two bounding boxes.
        Useful video explaining the idea: https://youtu.be/ANIzQ5G-XPE (thanks again Andrew)
        :param box1_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :param box2_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :return: Tensor of intersection boxes
        """
        intersection = YOLOModule.box_intersection(box1_tensor, box2_tensor)
        union = YOLOModule.box_union(box1_tensor, box2_tensor)
        return intersection / union


    @staticmethod
    def box_union(box1_tensor, box2_tensor):
        """

        :param box1_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :param box2_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :return: Union tensor of the two boxes tensors
        """

        intersection = YOLOModule.box_union(box1_tensor, box2_tensor)
        union = box1_tensor[:, 0] * box1_tensor[:, 3] + box2_tensor[:, 0] * box2_tensor[:, 3] - intersection
        return union


    @staticmethod
    def box_intersection(box1_tensor, box2_tensor):
        """

        :param box1_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :param box2_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :return: Intersection tensor of the two boxes tensors
        """
        w = YOLOModule.one_dim_overlap(box1_tensor[:, 0], box1_tensor[:, 2], box1_tensor[:, 1], box1_tensor[:, 3])
        h = YOLOModule.one_dim_overlap(box2_tensor[:, 0], box2_tensor[:, 2], box2_tensor[:, 1], box2_tensor[:, 3])
        if w < 0 or h < 0:
            area = 0
        else:
            area = w * h
        return area


    @staticmethod
    def one_dim_overlap(x1_tensor, w1_tensor, x2_tensor, w2_tensor):
        """

        :param x1_tensor:
        :param w1_tensor:
        :param x2_tensor:
        :param w2_tensor:
        :return:
        """
        right = torch.min(x1_tensor + w1_tensor / 2, x2_tensor + w2_tensor / 2)
        left = torch.max(x1_tensor - w1_tensor / 2, x2_tensor - w2_tensor / 2)
        return right - left
