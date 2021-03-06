from src.YOLOv3 import YOLOv3
from src.YOLOv3Dataset import YOLOv3Dataset

import torch
from torch.utils.data import DataLoader


dataset = YOLOv3Dataset()

image_path, image_tensor, label_tensor = dataset[0]
print(image_path)
print(image_tensor.shape)
print(label_tensor.shape)
print(image_tensor)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

model = YOLOv3()
model.eval()

for batch_i, (_, imgs, targets) in enumerate(dataloader):
    print(imgs.shape)
    print(targets.shape)
    output = model(imgs, targets)
script_model = torch.jit.script(model)

if isinstance(script_model, torch.jit.ScriptModule):
    print("Model was scripted successfully")
else:
    print("Something is not quite right")

