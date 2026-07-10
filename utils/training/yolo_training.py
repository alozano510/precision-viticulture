from ultralytics import YOLO
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str)
    parser.add_argument('-d', '--training_data', type=str)
    parser.add_argument('-e', '--epochs', type=str)
    args = parser.parse_args()

    # Load pretrained model
    model = YOLO(args.model_path)

    # Train models | If available, GPU is used by default
    model.train(data=args.training_data, epochs=args.epochs, device=0, save=True, imgsz=640)

    # Test object detection with fine-tuned models
    results = model("D:\\aloza\\Documents\\Escuela\\Feb-Jun26\\yolo_test_leaves_2.jpeg")

    for result in results:
        result.show()

    # Export the models to RKNN format
    model.export(format="onnx")