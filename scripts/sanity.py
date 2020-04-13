from src.YOLOv3 import YOLOv3
from src.YOLOv3Dataset import YOLOv3Dataset

import torch



dataset = YOLOv3Dataset()

image_path, image_tensor, label_tensor = dataset[0]
print(image_path)
print(image_tensor.shape)
print(label_tensor.shape)
print(image_tensor)

model = YOLOv3()
model.eval()
script_model = torch.jit.script(model)

if isinstance(script_model, torch.jit.ScriptModule):
    print("Model was scripted successfully")
else:
    print("Something is not quite right")

