import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image

import numpy as np


class YOLOv3Dataset(nn.Module):
    def __init__(self, images_list_path, image_size=416):
        super(YOLOv3Dataset, self).__init__()
        with open(images_list_path, "r") as f:
            self.images_paths = f.readlines()
        self.labels_paths = [path.replace("images", "labels")
                                 .replace(".png", ".txt")
                                 .replace(".jpg", ".txt")
                                 .replace(".jpeg", ".txt")
                             for path in self.images_paths]
        self.image_size = image_size

    def __getitem__(self, item):
        image_path = self.images_paths[item % len(self.images_paths)]

        image_tensor = transforms.ToTensor()(Image.open(image_path).convert("RGB"))

        if len(image_tensor.shape) != 3:
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.expand((3, image_tensor.shape[1:]))

        _, height, width = image_tensor.shape

        image_tensor, pad = YOLOv3Dataset.pad_to_square(image_tensor)

        label_tensor = torch.FloatTensor()

        return image_path, image_tensor, label_tensor

    @staticmethod
    def pad_to_square(image_tensor):
        """
        Padding the image tensor in order to get a square
        :param image_tensor: A rectangle shape image tensor
        :return: A padded square tensor of the image
        """
        channels, height, width = image_tensor.shape
        dim_diff = np.abs(height - width)

        # Padding values for the needed dimension:
        pad1 = dim_diff // 2
        pad2 = dim_diff - pad1

        # Determine padding
        pad = (0, 0, pad1, pad2) if height <= width else (pad1, pad2, 0, 0)

        # Add padding
        padded_image_tensor = F.pad(image_tensor, pad, "constant", value=0)

        return padded_image_tensor, pad
