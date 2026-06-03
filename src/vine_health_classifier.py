import threading
import cv2
import time
import numpy as np
import os
from scipy.special import softmax

class VineHealthClassifier:
    def __init__(self, camera: int):
        self.class_names = ['no saludable', 'saludable']
        
        # Camera settings
        self.camera = camera
        self.source = cv2.VideoCapture(self.camera)

        # Cap camera buffer for lighter data transfer
        self.source.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.source.set(cv2.CAP_PROP_FPS, 30)
        self.source.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.source.isOpened():
            raise ValueError(
                f"Error: Could not open camera from source {self.camera} \n Available cameras: {self.list_available_cameras()}")

        # Load classifier model
        from rknnlite.api import RKNNLite
        self.model = RKNNLite()
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'vine_health_classifier.rknn')
        self.model.load_rknn(self.model_path)
        self.model.init_runtime()
        print('RKNN model loaded on NPU')

        # Load YOLO object detection model
        self.yolo_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'grapevine_canopy_yolo.rknn')
        self.detector = RKNNLite()
        self.detector.load_rknn(self.yolo_model_path)
        self.detector.init_runtime()
        print('YOLO model loaded on NPU')

        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def _image_capture(self):
        has_frame, frame = self.source.read()
        if not has_frame:
            print("Could not read frame")
            return

        # Fix frame rotation because the camera is mounted with a 90° rotation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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

    @staticmethod
    def _preprocess_rknn(frame, processing_size: int = 256, roi: int = 224):
        """
        Resize and center crop for classification
        Args:
            frame (ndarray): frame to process
            processing_size (int): size of resized image to process
            roi (int): Region of interest to crop
        """
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize image
        h, w = frame.shape[:2]
        # get the proportion in relation to the largest side of the frame
        scale = min(processing_size / w, processing_size / h)
        nw, nh = int(w * scale), int(h * scale)  # scale sides
        resized_frame = cv2.resize(frame, (nw, nh))

        # Center crop
        h, w = resized_frame.shape[:2]
        top = (h - 224) // 2
        left = (w - 224) // 2
        img = resized_frame[top:top + 224, left:left + 224]
        img = np.expand_dims(img, axis=0)

        return img

    @staticmethod
    def _preprocess_yolo(frame, processing_size: int):
        """
        Letterbox resize and format for YOLO RKNN
        Args:
            frame (ndarray): frame to process
            processing_size (int): size of resized image to process
        """
        # Letterbox resize
        h, w = frame.shape[:2]
        # get the proportion in relation to the largest side of the frame
        scale = min(processing_size / w, processing_size / h)
        nw, nh = int(w * scale), int(h * scale)  # scale sides
        resized_frame = cv2.resize(frame, (nw, nh))

        # # adds gray bars to fill the image
        pad_top = (processing_size - nh) // 2
        pad_left = (processing_size - nw) // 2
        canvas = np.full((processing_size, processing_size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized_frame

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        return img, scale, pad_top, pad_left

    @staticmethod
    def _postprocess_yolo(output, conf_threshold, iou_threshold, pad_left, pad_top, scale) -> list:
        """Returns an empty list if no predicted object passes the condifence threshold"""
        predictions = output[0]
        if predictions.ndim == 3:
            predictions = predictions[0]

        # Get the confidence of the object detections
        obj_conf = predictions[:, 4]
        # Eliminate all objects whose confidence doesn't pass the threshold
        mask = obj_conf > conf_threshold
        predictions = predictions[mask]
        if len(predictions) == 0:
            return []

        # Get the predicted bounding boxes
        cx, cy, bw, bh = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Remove all boxes whose combined confidence does not pass the threshold
        scores = predictions[:, 4] * predictions[:, 5]  # obj_conf * class_conf
        keep = scores > conf_threshold
        boxes = np.stack([x1, y1, x2, y2], axis=1)[keep]
        scores = scores[keep]

        if len(boxes) == 0:
            return []

        # Non-Maximum Supression
        # Keeps the highest scoring box among overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
        )
        if len(indices) == 0:
            return []

        results = []
        for i in indices.flatten():
            x1_ = int((boxes[i][0] - pad_left) / scale)
            y1_ = int((boxes[i][1] - pad_top) / scale)
            x2_ = int((boxes[i][2] - pad_left) / scale)
            y2_ = int((boxes[i][3] - pad_top) / scale)
            results.append((float(scores[i]), x1_, y1_, x2_, y2_))

        return results

    def run_leaf_detection(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self._running = True

        while self._running:
            frame = self._image_capture()

            preprocessed_frame, scale, pad_top, pad_left= self._preprocess_yolo(frame, processing_size=640)
            output = self.detector.inference(inputs=[preprocessed_frame])
            results = self._postprocess_yolo(output, conf_threshold, iou_threshold, pad_left, pad_top, scale)

            for (score, x1, y1, x2, y2) in results:
                label = f"Leaf: {score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # pass frame display to main processing thread
            with self._frame_lock:
                self._latest_frame = frame

        print("Leaf detection stopped")

    def hybrid_analysis(self):
        """
        Uses a YOLO model to identify leaves and draw bounding boxes. The image is cropped into multiple images
        of the bounding boxes and runs a CNN on each of them to classify them.
        """
        self._running = True

        while self._running:
            frame = self._image_capture()

            input_data = self._preprocess_rknn(frame)
            # Run object detection
            # detection_results = detector.inference(inputs=[input_data])

            final_label = "saludable"
            final_confidence = 0

            # Iterate over detections
            for result in detection_results:
                for box in result.boxes:
                    # Crop the RoI
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]

                    # Run classification on the RoI
                    outputs = self.model.inference(inputs=[roi])
                    pred = int(np.argmax(outputs[0].flatten()))
                    confidence = float(softmax(outputs[0].flatten())[pred] * 100)
                    label = self.class_names[pred]

                    if label == "no saludable":
                        final_label = "no saludable"

                    final_confidence = confidence # TODO: placeholder, do a proper confidence calculation

            annotated_frame = self._draw_prediction(detection_results[0].plot(), final_label, final_confidence)

            with self._frame_lock:
                self._latest_frame = annotated_frame

            time.sleep(0.1)

        print("Plant analysis stopped")

    def run_analysis(self):
        self._running = True

        while self._running:
            frame = self._image_capture()

            input_data = self._preprocess_rknn(frame)

            outputs = self.model.inference(inputs=[input_data])
            pred = int(np.argmax(outputs[0].flatten()))
            confidence = float(softmax(outputs[0].flatten())[pred] * 100)
            label = self.class_names[pred]

            annotated_frame = self._draw_prediction(frame, label, confidence)

            with self._frame_lock:
                self._latest_frame = annotated_frame

            time.sleep(0.1)

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