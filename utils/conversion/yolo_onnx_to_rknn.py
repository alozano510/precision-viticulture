import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()

model_path = args.model_path

while not model_path:
    print(f"Specify the model path to convert")
    model_path = input("Enter model path: ").strip()

model = YOLO(model_path)
model.export(format="rknn", name="rk3588")
