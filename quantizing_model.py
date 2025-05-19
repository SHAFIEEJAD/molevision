import torch
from torchvision.models import mobilenet_v3_large
import torch.nn as nn

model = mobilenet_v3_large()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("latest_model_epoch20.pth", map_location=torch.device("cpu")))
#model.load_state_dict(torch.load("latest_model_epoch20.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(model, dummy_input, "melanoma_model.onnx", input_names=["input"], output_names=["output"], opset_version=11)
