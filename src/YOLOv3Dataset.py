import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import os
import random

from PIL import Image

import numpy as np


class YOLOv3Dataset(Dataset):

    def __init__(self, images_list_path="../data/images_list.txt", image_size=416, should_augment=True, transform=None):
        super(YOLOv3Dataset, self).__init__()
        with open(images_list_path, "r") as f:
            self.images_paths = f.read().splitlines()
        self.labels_paths = [path.replace("images", "labels")
                                 .replace(".png", ".txt")
                                 .replace(".jpg", ".txt")
                                 .replace(".jpeg", ".txt")
                             for path in self.images_paths]
        self.image_size = image_size
        self.should_augment = should_augment
        self.transform = transform

    def __getitem__(self, item):

        image_path = self.images_paths[item % len(self.images_paths)]
        image_tensor = transforms.ToTensor()(Image.open(image_path).convert("RGB"))

        if len(image_tensor.shape) != 3:
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.expand((3, image_tensor.shape[1:]))

        _, height, width = image_tensor.shape

        image_tensor, pad = YOLOv3Dataset.pad_to_square(image_tensor)
        _, padded_h, padded_w = image_tensor.shape

        label_path = self.labels_paths[item % len(self.labels_paths)].rstrip()
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # Extract bounding boxes coordinates, and convert to desired format.
            # We leave boxes[:,0] as is, since that's the class annotation
            # We change the scale from 1*1 to w*d, since we want to apply to the tensor
            # Note: The bounding boxes are given in the format:
            #       class, left_x, up_y, right_x, bottom_y
            #       However, we need to return them in the format:
            #       class, center_x, center_y, width, height
            #       So this is why convert. #whyweconvert
            x1 = width * (boxes[:, 1] - boxes[:, 3] / 2) + pad[0]
            y1 = height * (boxes[:, 2] - boxes[:, 4] / 2) + pad[1]
            x2 = width * (boxes[:, 1] + boxes[:, 3] / 2) + pad[2]
            y2 = height * (boxes[:, 2] + boxes[:, 4] / 2) + pad[3]

            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= width / padded_w
            boxes[:, 4] *= height / padded_h
            label_tensor = torch.zeros(len(boxes), 6)
            label_tensor[:, 1:] = boxes

        else:
            print("File is missing: "+label_path)
            return
        if self.should_augment and self.transform:
            image_tensor, label_tensor = self.transform(image_tensor), self.transform(label_tensor)

        return image_path, image_tensor, label_tensor

    def collate_fn(self, batch):
        """
        Merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset

        :param batch:
        :return:
        """
        paths, imgs, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([transforms.Resize(img, self.img_size) for img in imgs])
        self.batch_count += 1

        return paths, imgs, targets

    def __len__(self):
        return len(self.images_paths)

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
