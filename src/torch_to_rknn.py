"""
Loads a ResNet model using PyTorch with custom weights and exports it as Rockchip Neural Network (RKNN)
IMPORTANT: this program only works on x86 Linux systems.
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models
from rknn.api import RKNN

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()

if args.model_path:
    model_path = args.model_path
else:
    model_path = "model\\binary_classifier_v1.pt"

# Set GPU for processing
device = torch.device("cpu")
print(f"Using device: {device}\n")

# Load ResNet model
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

# Export PyTorch model to open standard format ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "vine_binary_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)

print("Torch Model exported to ONNX")

# Convert ONNX to RKNN
rknn = RKNN()

rknn.config(
    mean_values=[[0.485 * 255, 0.456 * 255, 0.406 * 255]],
    std_values=[[0.229 * 255, 0.224 * 255, 0.225 * 255]],
    target_platform='rk3588'
)

rknn.load_onnx(model="vine_binary_classifier.onnx")
rknn.build(do_quantization=False)
rknn.export_rknn("vine_binary_classifier.rknn")

print("Exported ONNX model to RKNN")
rknn.release()