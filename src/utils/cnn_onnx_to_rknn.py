"""
Converts and exports Neural Network with ONNX framework as a Rockchip Neural Network (RKNN)
IMPORTANT: this program only works on x86 Linux systems.
"""
import os
import argparse
from rknn.api import RKNN

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()

model = args.model_path

while not model:
    print(f"Specify the model path for the ONNX to convert")
    model = input("Enter model path: ").strip()

# Convert ONNX to RKNN
rknn = RKNN()

rknn.config(
    mean_values=[[0.485 * 255, 0.456 * 255, 0.406 * 255]],
    std_values=[[0.229 * 255, 0.224 * 255, 0.225 * 255]],
    target_platform='rk3588'
)

rknn.load_onnx(model=model)
rknn.build(do_quantization=False)
model_name = os.path.splitext(os.path.basename(model))[0]
rknn.export_rknn(model_name + ".rknn")

print("Exported ONNX model to RKNN")
rknn.release()