import torch
from torch.utils.data import DataLoader

from src.YOLOv3 import YOLOv3
from src.YOLOv3Dataset import YOLOv3Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLOv3()
# model.load_state_dict(torch.load("weights.pt"))

train_dataset = YOLOv3Dataset()
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)