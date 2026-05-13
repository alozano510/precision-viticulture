import cv2
import numpy as np
import argparse
import os
from rknnlite.api import RKNNLite
from scipy.special import softmax

class VineHealthClassifier:
    def __init__(self, camera):
        self.class_names = ['unhealthy', 'healthy']
        
        # Camera settings
        self.camera = camera
        self.source = cv2.VideoCapture(self.camera)

        if not self.source.isOpened():
            raise ValueError(
                f"Error: Could not open camera from source {self.camera} \n Available cameras: {self.list_available_cameras()}")

        self.win_name = "Clasificador de salud de viñedos"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        # Load model
        self.model = RKNNLite()
        self.model_path = os.path.join(os.path.dirname(__file__), 'vine_health_classifier.rknn')
        self.model.load_rknn(self.model_path)
        self.model.init_runtime()
        print('RKNN model loaded on NPU')

    def image_capture(self):
        has_frame, frame = self.source.read()
        if not has_frame:
            print("Could not read frame")
            return

        return frame

    def draw_prediction(self, frame, prediction, confidence):
        color = (0, 255, 0) if prediction == 'saludable' else (0, 0, 255)
        cv2.putText(frame, f"{prediction} ({confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2)

        cv2.imshow(self.win_name, frame)

    @staticmethod
    def preprocess_rknn(frame):
        # Resize and format for RKNN
        img = cv2.resize(frame, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Center crop
        h, w = img.shape[:2]
        top = (h - 224) // 2
        left = (w - 224) // 2
        img = img[top:top + 224, left:left + 224]
        img = np.expand_dims(img, axis=0)

        return img

    def run_analysis(self):
        frame = self.image_capture()

        input_data = self.preprocess_rknn(frame)
        outputs = self.model.inference(inputs=[input_data])
        pred = int(np.argmax(outputs[0].flatten()))
        confidence = float(softmax(outputs[0].flatten())[pred] * 100)
        label = self.class_names[pred]

        self.draw_prediction(frame, pred, confidence)
        cv2.waitKey(1)

        print(f'Prediction: {label} {confidence:.1f}%')
        
        return label, confidence

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