from src.YOLOv3 import YOLOv3

import torch

model = YOLOv3()
model.eval()
script_model = torch.jit.script(model)

if isinstance(script_model, torch.jit.ScriptModule):
    print("Model was scripted successfully")
else:
    print("Something is not quite right")
