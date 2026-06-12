from ultralytics import YOLO
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()

    # Load pretrained models
    model = YOLO("/runs/detect/train-9/weights/yolo_grapevine_canopy_leaves.pt")

    # Train models | If available, GPU is used by default
    #data_dir = args.path
    #models.train(data=data_dir, epochs=50, device=0, save=True, imgsz=640)

    # Test object detection with fine-tuned models
    results = model("D:\\aloza\\Documents\\Escuela\\Feb-Jun26\\yolo_test_leaves_2.jpeg")

    for result in results:
        result.show()

    # Export the models to RKNN format
    # models.export(format="onnx")