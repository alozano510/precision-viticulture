import cv2
import os
from PIL import Image
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from vine_health_classifier import VineHealthClassifier

class VineHealthClassifierTorch(VineHealthClassifier):
    """
    PyTorch-based subclass of VineHealthClassifier for development on Windows.
    Replaces RKNN models loading and inference with PyTorch (ResNet50) and Ultralytics YOLO.
    Camera rotation is not applied on _image_capture() since the camera is upright on the dev machine.
    """

    def _load_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}\n")
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_binary_classifier_v1.pt')
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("PyTorch classifier loaded")

        # Load Ultralytics YOLO detector
        self.yolo_model_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'runs', 'detect', 'train-7', 'weights', 'yolo_grapevine_canopy_leaves.pt'
        )
        self.detector = YOLO(self.yolo_model_path)
        self.yolo_classes = self.detector.names
        print("Ultralytics YOLO loaded")

    def _image_capture(self):
        has_frame, frame = self.source.read()
        if not has_frame:
            print("Could not read frame")
            return

        return frame

    def _preprocess_torch_tensor(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        input_tensor = transform(pil_image).unsqueeze(0).to(self.device)

        return input_tensor

    def leaf_classification(self, frame):
        t0 = time.perf_counter()
        input_tensor = self._preprocess_torch_tensor(frame)
        t1 = time.perf_counter()

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)
            label = self.class_names[pred.item()]
            confidence = probs[pred.item()].item() * 100

        t2 = time.perf_counter()

        runtime = {
            'preprocessing_time': t1 - t0,
            'inference_time': t2 - t1,
            'postprocessing_time': 0.0,
            'total_time': t2 - t0,
        }

        # No NPU on dev machine; RAM only
        memory = {
            'ram_mb': self._get_process_mem_mb(),
            'npu_mb': 0,
        }

        return label, confidence, runtime, memory

    def leaf_detection(self, frame, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """Override: run Ultralytics YOLO inference instead of RKNN."""
        t0 = time.perf_counter()
        output = self.detector(frame, verbose=False, conf=conf_threshold, iou=iou_threshold)
        t1 = time.perf_counter()

        results = []
        for box in output[0].boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            results.append((cls_id, score, x1, y1, x2, y2))

        t2 = time.perf_counter()

        runtime = {
            'preprocessing_time': 0.0,
            'inference_time': t1 - t0,
            'postprocessing_time': t2 - t1,
            'total_time': t2 - t0,
        }

        memory = {
            'ram_mb': self._get_process_mem_mb(),
            'npu_mb': 0,
        }

        return results, runtime, memory