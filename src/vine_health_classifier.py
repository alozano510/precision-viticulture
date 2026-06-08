import threading
import cv2
import time
import numpy as np
import os
import csv
import datetime
import subprocess
import psutil
import re
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
        ram_before_cnn_weights = self._get_process_mem_mb()
        npu_before_cnn_weights = self._get_npu_mem_mb()
        self.model = RKNNLite()
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'vine_health_classifier.rknn')
        self.model.load_rknn(self.model_path)
        self.model.init_runtime()
        ram_after_cnn_weights = self._get_process_mem_mb()
        npu_after_cnn_weights = self._get_npu_mem_mb()
        self.cnn_weight_memory = ram_after_cnn_weights - ram_before_cnn_weights
        self.npu_weight_memory = npu_after_cnn_weights - npu_before_cnn_weights
        print('RKNN model loaded on NPU')

        # Load YOLO object detection model
        ram_before_yolo_weights = self._get_process_mem_mb()
        npu_before_yolo_weights = self._get_npu_mem_mb()
        self.yolo_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'vineyard_yolo.rknn')
        self.detector = RKNNLite()
        self.detector.load_rknn(self.yolo_model_path)
        self.detector.init_runtime()
        self.yolo_classes = None
        ram_after_yolo_weights = self._get_process_mem_mb()
        npu_after_yolo_weights = self._get_npu_mem_mb()
        self.cnn_weight_memory = ram_after_yolo_weights - ram_before_yolo_weights
        self.npu_weight_memory = npu_after_yolo_weights - npu_before_yolo_weights
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
    def _preprocess_rknn(frame):
        """
        Resize and center crop for classification
        Args:
            frame (ndarray): frame to process
            processing_size (int): size of resized image to process
            roi (int): Region of interest to crop
        """
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize image
        resized_frame = cv2.resize(rgb_img, (256, 256))

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

    def _postprocess_yolo(self, output, conf_threshold, iou_threshold, pad_left, pad_top, scale) -> list:
        """Returns an empty list if no predicted object passes the condifence threshold"""
        predictions = output[0]
        if predictions.ndim == 3:
            predictions = predictions[0]

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

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
        scores = predictions[:, 4]
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        num_classes = predictions.shape[1] - 5

        if num_classes <= 1:
            scores = predictions[:, 4]
            class_ids = np.zeros(len(scores), dtype=int)
            self.yolo_classes = ["leaf"]
        else:
            class_scores = predictions[:, 5:]  # (N, 3)
            class_ids = np.argmax(class_scores, axis=1)  # (N,)
            scores = predictions[:, 4] * class_scores[np.arange(len(class_ids)), class_ids]
            self.yolo_classes = ["grape", "ground", "branch", "leaf"]

        keep = scores > conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

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
            results.append((int(class_ids[i]), float(scores[i]), x1_, y1_, x2_, y2_))

        return results

    def leaf_detection(self, frame, conf_threshold: float = 0.25, iou_threshold: float = 0.45):

        t0 = time.perf_counter()
        preprocessed_frame, scale, pad_top, pad_left = self._preprocess_yolo(frame, processing_size=640)
        t1 = time.perf_counter()
        output = self.detector.inference(inputs=[preprocessed_frame])
        t2 = time.perf_counter()
        results = self._postprocess_yolo(output, conf_threshold, iou_threshold, pad_left, pad_top, scale)
        t3 = time.perf_counter()

        preprocessing_time = t1 - t0
        inference_time = t2 - t1
        postprocessing_time = t3 - t2
        total_time = t3 - t0

        runtime = {
                'preprocessing_time': preprocessing_time,
                'inference_time': inference_time,
                'postprocessing_time': postprocessing_time,
                'total_time': total_time,
        }

        memory = {
                'ram_mb': self._get_process_mem_mb(),
                'npu_mb': self._get_npu_mem_mb()
        }
        return results, runtime, memory

    def leaf_classification(self, frame):

        t0 = time.perf_counter()
        input_data = self._preprocess_rknn(frame)
        t1 = time.perf_counter()
        outputs = self.model.inference(inputs=[input_data])
        t2 = time.perf_counter()
        pred = int(np.argmax(outputs[0].flatten()))
        confidence = float(softmax(outputs[0].flatten())[pred] * 100)
        label = self.class_names[pred]
        t3 = time.perf_counter()

        preprocessing_time = t1 - t0
        inference_time = t2 - t1
        postprocessing_time = t3 - t2
        total_time = t3 - t0

        runtime = {
            'preprocessing_time': preprocessing_time,
            'inference_time': inference_time,
            'postprocessing_time': postprocessing_time,
            'total_time': total_time,
        }

        memory = {
            'ram_mb': self._get_process_mem_mb(),
            'npu_mb': self._get_npu_mem_mb()
        }

        return label, confidence, runtime, memory

    def _annotate_detected_objects(self, frame, predictions):

        COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

        for (cls_id, score, x1, y1, x2, y2) in predictions:
            color = COLORS[cls_id % len(COLORS)]
            label = f"{self.yolo_classes[cls_id]}: {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def run_leaf_detection(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self._running = True

        baseline_ram = self._get_process_mem_mb()
        baseline_npu = self._get_npu_mem_mb()

        performance = {
            'runtime': {},
            'memory': {},
        }
        i = 0 # iteration counter

        while self._running:

            frame = self._image_capture()

            results, runtime, memory= self.leaf_detection(
                frame=frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                )

            annotated_frame = self._annotate_detected_objects(frame, results)

            # pass frame display to main processing thread
            with self._frame_lock:
                self._latest_frame = annotated_frame

            # Save performance stats
            performance['runtime'][str(i)] = runtime
            performance['memory'][str(i)] = {
                k: v - baseline_ram if 'ram' in k else v - baseline_npu
                for k, v in memory.items()
            }

            i+=1
            time.sleep(0.3)

        self._save_results(performance)
        print("Leaf detection stopped")

    def run_leaf_classification(self):
        self._running = True

        baseline_ram = self._get_process_mem_mb()
        baseline_npu = self._get_npu_mem_mb()

        performance = {
            'runtime': {},
            'memory': {},
        }
        i = 0  # iteration counter

        while self._running:
            frame = self._image_capture()
            label, confidence, runtime, memory = self.leaf_classification(frame)
            annotated_frame = self._draw_prediction(frame, label, confidence)

            with self._frame_lock:
                self._latest_frame = annotated_frame

            # Save performance stats
            performance['runtime'][str(i)] = runtime
            performance['memory'][str(i)] = {
                k: v - baseline_ram if 'ram' in k else v - baseline_npu
                for k, v in memory.items()
            }

            i+=1
            time.sleep(0.3)

        self._save_results(performance)
        print("Plant analysis stopped")

    def hybrid_analysis(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Uses a YOLO model to identify leaves and draw bounding boxes. The image is cropped into multiple images
        of the bounding boxes and runs a CNN on each of them to classify them.
        """
        self._running = True

        baseline_ram = self._get_process_mem_mb()
        baseline_npu = self._get_npu_mem_mb()

        performance = {
            'runtime': {},
            'memory': {},
        }
        i = 0  # iteration counter

        while self._running:
            frame = self._image_capture()

            # Run leaf detection
            detection_results, detection_runtime, detection_memory = self.leaf_detection(
                            frame=frame,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            )

            final_label = "saludable"
            final_confidence = 0

            classification_runtime = 0

            # Iterate over detections
            for (cls_id, score, x1, y1, x2, y2) in detection_results:
                # Skip all detected objects that are not leaves
                if cls_id is not "leaf":
                    continue

                # Crop the Region of Interest
                roi = frame[y1:y2, x1:x2]

                # Run classification on the RoI
                label, confidence, runtime, memory = self.leaf_classification(roi)
                classification_runtime += runtime['total_time']

                final_confidence = confidence # TODO: placeholder, do a proper confidence calculation

                if label == "no saludable":
                    final_label = "no saludable"
                    break

            ram_sample = self._get_process_mem_mb()
            npu_sample = self._get_npu_mem_mb()

            annotated_frame = self._annotate_detected_objects(frame, detection_results)
            annotated_frame = self._draw_prediction(annotated_frame, final_label, final_confidence)

            with self._frame_lock:
                self._latest_frame = annotated_frame

            # Save performance stats
            performance['runtime'][str(i)] = {
                'detection_time': detection_runtime['total_time'],
                'classification_time': classification_runtime,
                'total_time': detection_runtime['total_time'] + classification_runtime
            }
            performance['memory'][str(i)] = {
                'ram_mb': ram_sample - baseline_ram,
                'npu_mb': npu_sample - baseline_npu,
                'num_detected_leaves': len(detection_results)
            }

            i+=1
            time.sleep(0.3)

        print("Plant analysis stopped")

    def get_latest_frame(self):
        """Retrieves the latest annotated frame. Used to display it on the main thread."""
        with self._frame_lock:
            return self._latest_frame

    def stop(self):
        self._running = False
        self.source.release()

    @staticmethod
    def _save_results(results):
        """Exports results as CSV file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realtime_performance_{timestamp}.csv"

        fieldnames = ['iteration'] + [key for key in results['runtime']['0']] + [key for key in results['memory']['0']]

        with open(filename, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, stats in results['runtime'].items():
                writer.writerow({
                    'iteration': i,
                    **results['runtime'][i],
                    **results['memory'][i],
                })

        print(f"Saved {filename}")

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

    @staticmethod
    def _get_npu_mem_mb():
        """Read NPU memory from /proc or sysfs (RK3588-specific)."""
        try:
            result = subprocess.run(
                ['cat', '/proc/rknpu/mem'],
                capture_output=True, text=True
            )
            match = re.search(r'(\d+)\s*bytes', result.stdout)
            if match:
                return int(match.group(1)) / 1024 / 1024
            return None
        except Exception:
            return "N/A"

    @staticmethod
    def _get_process_mem_mb():
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024