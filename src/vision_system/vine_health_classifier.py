import pathlib
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
    def __init__(self, camera: int, fps: int = 10):
        self.class_names = ['no saludable', 'saludable']
        self._running = False
        self._latest_frame = None
        self._fps = fps
        self._frame_lock = threading.Lock()
        self._method = None
        self._timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = pathlib.Path("../runs") / self._timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Camera settings
        self.camera = camera
        self.source = cv2.VideoCapture(self.camera)

        # Cap camera buffer for lighter data transfer
        self.source.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.source.set(cv2.CAP_PROP_FPS, 30)
        self.source.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Define codec and create VideoWriter
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filepath = self.output_dir / "recording.mp4"
        self.out = cv2.VideoWriter(filepath, self.fourcc, fps, (480, 640))

        if not self.out.isOpened():
            print("VideoWriter failed to open!")

        if not self.source.isOpened():
            raise ValueError(
                f"Error: Could not open camera from source {self.camera} \n Available cameras: {self.list_available_cameras()}")


        self._load_models()

    def _load_models(self):
        # Load classifier models
        self.device = "Orange Pi 5 / RK3588"
        from rknnlite.api import RKNNLite
        ram_before_cnn_weights = self._get_process_mem_mb()
        npu_before_cnn_weights = self._get_npu_mem_mb()
        self.model = RKNNLite()
        self.model_path = "model/vine_health_classifier.rknn"
        self.model.load_rknn(self.model_path)
        self.model.init_runtime()
        ram_after_cnn_weights = self._get_process_mem_mb()
        npu_after_cnn_weights = self._get_npu_mem_mb()
        self.cnn_weight_memory = ram_after_cnn_weights - ram_before_cnn_weights
        self.npu_weight_memory = npu_after_cnn_weights - npu_before_cnn_weights
        print('RKNN models loaded on NPU')

        # Load YOLO object detection models
        ram_before_yolo_weights = self._get_process_mem_mb()
        npu_before_yolo_weights = self._get_npu_mem_mb()
        self.yolo_model_path = "utils/coversion/best_rknn_model/best-rk3588.rknn"
        self.detector = RKNNLite()
        self.detector.load_rknn(self.yolo_model_path)
        self.detector.init_runtime()
        self.yolo_classes = None
        ram_after_yolo_weights = self._get_process_mem_mb()
        npu_after_yolo_weights = self._get_npu_mem_mb()
        self.cnn_weight_memory = ram_after_yolo_weights - ram_before_yolo_weights
        self.npu_weight_memory = npu_after_yolo_weights - npu_before_yolo_weights
        print('YOLO models loaded on NPU')

    def _image_capture(self):
        has_frame, frame = self.source.read()
        if not has_frame:
            print("Could not read frame")
            return

        # Fix frame rotation because the camera is mounted with a 90° rotation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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
        """Returns an empty list if no predicted object passes the confidence threshold"""
        
        # 1. Parse index 0 which holds boxes and scores
        predictions = output[0]
        if predictions.ndim == 3:
            predictions = predictions[0]  # Shape becomes (37, 8400)

        # Transpose so rows represent detections: shape becomes (8400, 37)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        # 2. Extract bounding boxes and class scores
        # YOLOv8 format: 0,1,2,3 are cx, cy, bw, bh. Index 4 is your "leaf" class score.
        # Indices 5 to 36 (the remaining 32 channels) are the mask coefficients.
        boxes_raw = predictions[:, :4]
        class_scores = predictions[:, 4:5]  # Pulls index 4 as a 2D column matrix
    
        # Get highest score and corresponding class ID per anchor point
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # 3. Filter out weak detections before processing math
        mask = scores > conf_threshold
        if not np.any(mask):
            return []  # Return cleanly if nothing is detected in the frame

        # Apply confidence mask to all vectors
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 4. Convert bounding boxes from center coordinates to corner coordinates
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 5. Non-Maximum Suppression (NMS) to clear out overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
        )
        if len(indices) == 0:
            return []

        # 6. Map normalized coordinates back to the original source frame scale
        if scale <= 0:
            scale = 1.0

        results = []
        for i in indices.flatten():
            if not np.isfinite(boxes[i]).all():
                continue

            x1_ = int((boxes[i][0] - pad_left) / scale)
            y1_ = int((boxes[i][1] - pad_top) / scale)
            x2_ = int((boxes[i][2] - pad_left) / scale)
            y2_ = int((boxes[i][3] - pad_top) / scale)
        
            # Explicit class name setting for 1-class setup
            self.yolo_classes = ["leaf"]
        
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
            label = f"Leaves: {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def run_leaf_detection(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self._running = True
        self._method = "YOLO - Object detection"

        baseline_ram = self._get_process_mem_mb()
        baseline_npu = self._get_npu_mem_mb()

        performance = {
            'runtime': {},
            'memory': {},
        }
        i = 0 # iteration counter

        while self._running:
            t_frame_start = time.perf_counter()
            frame = self._image_capture()

            results, runtime, memory= self.leaf_detection(
                frame=frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                )

            annotated_frame = self._annotate_detected_objects(frame, results)
            self.out.write(annotated_frame)

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

            elapsed = time.perf_counter() - t_frame_start
            sleep_time = 1 / self._fps - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._save_config()
        self._save_performance(performance)
        print("Leaf detection stopped")

    def run_leaf_classification(self):
        self._running = True
        self._method = "CNN - Classification"

        baseline_ram = self._get_process_mem_mb()
        baseline_npu = self._get_npu_mem_mb()

        performance = {
            'runtime': {},
            'memory': {},
        }
        i = 0  # iteration counter

        while self._running:
            t_frame_start = time.perf_counter()
            frame = self._image_capture()

            label, confidence, runtime, memory = self.leaf_classification(frame)
            annotated_frame = self._draw_prediction(frame, label, confidence)

            self.out.write(annotated_frame)

            with self._frame_lock:
                self._latest_frame = annotated_frame

            # Save performance stats
            performance['runtime'][str(i)] = runtime
            performance['memory'][str(i)] = {
                k: v - baseline_ram if 'ram' in k else v - baseline_npu
                for k, v in memory.items()
            }

            i+=1

            elapsed = time.perf_counter() - t_frame_start
            sleep_time = 1 / self._fps - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._save_config()
        self._save_performance(performance)
        print("Plant analysis stopped")

    def hybrid_analysis(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45, unhealthy_threshold: float = 0.4,):
        """
        Uses a YOLO models to identify leaves and draw bounding boxes. The image is cropped into multiple images
        of the bounding boxes and runs a CNN on each of them to classify them.
        """
        self._running = True
        self._method = "Hybrid"

        baseline_ram = self._get_process_mem_mb()
        baseline_npu = self._get_npu_mem_mb()

        performance = {
            'runtime': {},
            'memory': {},
        }
        i = 0  # iteration counter

        while self._running:
            t_frame_start = time.perf_counter()
            frame = self._image_capture()

            # Run leaf detection
            detection_results, detection_runtime, detection_memory = self.leaf_detection(
                            frame=frame,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            )

            confidence_healthy = []
            confidence_not_healthy = []

            classification_runtime = 0

            # Iterate over detections
            for (cls_id, score, x1, y1, x2, y2) in detection_results:
                # Skip all detected objects that are not leaves
                # if "Lea" not in self.yolo_classes[cls_id]:
                #    continue

                # 1. Get current image boundaries safely
                img_h, img_w = frame.shape[:2]
            
                # 2. Guard rail: Clamp coordinates so they NEVER exceed the frame boundaries [0, max_dim - 1]
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
            
                # 3. Defensive Gate: If the box is empty or inverted, skip it entirely
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                # Crop the Region of Interest
                roi = frame[y1:y2, x1:x2]

                # Run classification on the RoI
                label, confidence, runtime, memory = self.leaf_classification(roi)
                classification_runtime += runtime['total_time']

                if label == "saludable":
                    confidence_healthy.append(confidence)
                else:
                    confidence_not_healthy.append(confidence)

            # Calculate weighted majority
            w_healthy = sum(confidence_healthy)
            w_not_healthy = sum(confidence_not_healthy)
            w_total = w_healthy + w_not_healthy

            if w_total == 0:
                final_confidence = 0
                final_label = "inconcluso"

            else:
                # Calculate healthy to unhealthy ratio
                n_total = len(confidence_healthy) + len(confidence_not_healthy)
                ratio = len(confidence_not_healthy) / n_total

                if ratio > unhealthy_threshold:
                    final_confidence = w_not_healthy / len(confidence_not_healthy)
                    final_label = "no saludable"
                elif w_healthy > w_not_healthy:
                    final_confidence = w_healthy / len(confidence_healthy)
                    final_label = "saludable"
                else:
                    final_confidence = w_not_healthy / len(confidence_not_healthy)
                    final_label = "no saludable"

            ram_sample = self._get_process_mem_mb()
            npu_sample = self._get_npu_mem_mb()

            annotated_frame = self._annotate_detected_objects(frame, detection_results)
            annotated_frame = self._draw_prediction(annotated_frame, final_label, final_confidence)

            self.out.write(annotated_frame)

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

            elapsed = time.perf_counter() - t_frame_start
            sleep_time = 1 / self._fps - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._save_config()
        self._save_performance(performance)
        print("Plant analysis stopped")

    def get_latest_frame(self):
        """Retrieves the latest annotated frame. Used to display it on the main thread."""
        with self._frame_lock:
            return self._latest_frame

    def stop(self):
        self._running = False
        self.source.release()
        self.out.release()

    def _save_config(self):
        """Exports the configuration used to run the program as a text file"""
        filename = f"config.txt"
        filepath = os.path.join(self.output_dir, filename)

        config = {
            "device": self.device,
            "classification models": pathlib.Path(self.model_path).name,
            "detection models": pathlib.Path(self.yolo_model_path).name,
            "method": self._method
        }

        with open(filepath, "w") as f:
            f.write(f"Run configuration\n")
            f.write("=" * 40 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        print(f"Saved {filepath}")

    def _save_performance(self, performance):
        """Exports results as CSV file"""
        filename = f"realtime_performance.csv"
        filepath = os.path.join(self.output_dir, filename)

        fieldnames = ['iteration'] + [key for key in performance['runtime']['0']] + [key for key in performance['memory']['0']]

        with open(filepath, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, stats in performance['runtime'].items():
                writer.writerow({
                    'iteration': i,
                    **performance['runtime'][i],
                    **performance['memory'][i],
                })

        print(f"Saved {filepath}")

    @staticmethod
    def _draw_prediction(frame, label, confidence):
        color = (0, 255, 0) if label == 'saludable' else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2)

        return frame

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
        """Read shared DMA-BUF memory allocated by the rknpu driver on Kernel 6.1."""
        path = '/sys/kernel/debug/dma_buf/bufinfo'
        if not os.path.exists(path):
            return 0.0
    
        try:
            # Read the buffer info file (requires sudo)
            result = subprocess.run(
                ['sudo', 'cat', path],
                capture_output=True, text=True, check=True
            )
    
            total_npu_bytes = 0
            # Look for buffers explicitly allocated or attached to the rknpu driver
            for line in result.stdout.splitlines():
                if 'rknpu' in line or 'rk-npu' in line:
                    # DMA-BUF lines usually contain the size of the buffer in bytes
                    # Example: size: 16777216 or similar integer listings
                    matches = re.findall(r'\bsize:\s*(\d+)\b|\b(\d+)\s+bytes\b', line, re.IGNORECASE)
                    for match in matches:
                        # Snag whichever regex group matched the number
                        num = match[0] if match[0] else match[1]
                        total_npu_bytes += int(num)
    
            return total_npu_bytes / 1024 / 1024  # Convert to MB
    
        except subprocess.CalledProcessError:
            print("Permission Denied: Run your script with 'sudo' to inspect DMA-BUFs.")
        except Exception as e:
            print(f"Error parsing DMA buffers: {e}")
    
        return 0.0

    @staticmethod
    def _get_process_mem_mb():
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
