"""
Converts and exports a YOLO model in ONNX framework as a Rockchip Neural Network (RKNN)
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
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
    output_optimize=True
)

rknn.load_onnx(model=model)
rknn.build(do_quantization=False)
model_name = os.path.splitext(os.path.basename(model))[0]
rknn.export_rknn(model_name + ".rknn")

print("Exported ONNX model to RKNN")
rknn.release()