from ultralytics import YOLO
model = YOLO("utils/coversion/best.pt")
model.export(format="rknn", name="rk3588")
