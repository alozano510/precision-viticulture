import threading
import cv2
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

class VineHealthClassifierTorch:
    def __init__(self, camera: int):
        self.class_names = ['no saludable', 'saludable']

        # Camera settings
        self.camera = camera
        self.source = cv2.VideoCapture(self.camera)

        if not self.source.isOpened():
            raise ValueError(
                f"Error: Could not open camera from source {self.camera} \n Available cameras: {self.list_available_cameras()}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}\n")
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'binary_classifier_v1.pt')
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def _image_capture(self):
        has_frame, frame = self.source.read()
        if not has_frame:
            print("Could not read frame")
            return

        return frame

    def _draw_prediction(self, frame, label, confidence):
        color = (0, 255, 0) if label == 'saludable' else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2)

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

    def run_leaf_detection(self):
        self._running = True
        # Load pretrained model
        model = YOLO("D:\\PycharmProjects\\precision-viticulture\\runs\\detect\\train-9\\weights\\best.pt")

        while self._running:
            frame = self._image_capture()

            output = model(frame, verbose=False)
            annotated_frame = output[0].plot()

            # pass frame display to main processing thread
            with self._frame_lock:
                self._latest_frame = annotated_frame

        print("Leaf detection stopped")

    def run_analysis(self):
        self._running = True

        while self._running:
            frame = self._image_capture()

            input_data = self._preprocess_torch_tensor(frame)

            with torch.no_grad():
                output = self.model(input_data)
                probs = torch.softmax(output, dim=1)[0]
                _, pred = torch.max(output, 1)
                label = self.class_names[pred.item()]
                confidence = probs[pred.item()].item() * 100

            annotated_frame = self._draw_prediction(frame, label, confidence)

            # pass frame display to main processing thread
            with self._frame_lock:
                self._latest_frame = annotated_frame

        print("Plant analysis stopped")

    def get_latest_frame(self):
        """Retrieves the latest annotated frame. Used to display it on the main thread."""
        with self._frame_lock:
            return self._latest_frame

    def stop(self):
        self._running = False
        self.source.release()

    @staticmethod
    def list_available_cameras(max_index=10):
        """Returns a list of all available cameras. If no camera is available, returns a string"""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        if not available:
            return "No available cameras"
        return available