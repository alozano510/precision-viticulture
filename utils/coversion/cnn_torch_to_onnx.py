"""
Loads a ResNet models using PyTorch with custom weights and exports it as Rockchip Neural Network (RKNN)
IMPORTANT: this program only works on x86 Linux systems.
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()

model_path = args.model_path

while not model_path:
    print(f"Specify the models path for PyTorch Neural Network to convert")
    model_path = input("Enter models path: ").strip()

# Set GPU for processing
device = torch.device("cpu")
print(f"Using device: {device}\n")

# Load ResNet models
model = models.resnet50(weights=None)

# Change final layer to be a binary classifier
num_classes = 2  # healthy / unhealthy
class_names = ['no saludable', 'saludable']
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Load weights from fine-tuning
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
print("Model weights loaded\n")
model.to(device)
model.eval()

# Export PyTorch models to open standard format ONNX
dummy_input = torch.randn(1, 3, 224, 224)
model_name = os.path.splitext(os.path.basename(model_path))[0] + ".onnx"
torch.onnx.export(
    model,
    dummy_input,
    model_name,
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)

print("Torch Model exported to ONNX")