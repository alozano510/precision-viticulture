# For image processing
import cv2
import numpy as np
from PIL import Image
import argparse

# ROS2 libraries
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

# Neural Network Architecture
from rknnlite.api import RKNNLite
from scipy.special import softmax

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera', type=int, default=0)
args = parser.parse_args()

class VineHealthClassifier(Node):
    def __init__(self):
        super().__init__('vine_health_classifier_node')
        self.get_logger().info('Starting Vine Health Classifier node...')
        self.publisher = self.create_publisher(String, '/vine_health', 10)

        self.class_names = ['unhealthy', 'healthy']
        timer_period = 1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        # Camera settings
        self.camera = args.camera
        self.source = cv2.VideoCapture(self.camera)

        if not self.source.isOpened():
            raise ValueError(
                f"Error: Could not open camera from source {self.camera} \n Available cameras: {self.list_available_cameras()}")

        self.win_name = "Clasificador de salud de viñedos"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        # Load model
        self.model = RKNNLite()
        self.model.load_rknn('vine_health_classifier.rknn')
        self.model.init_runtime()
        self.get_logger().info('RKNN model loaded on NPU')

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

    def preprocess_rknn(self, frame):
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

    def timer_callback(self):
        frame = self.image_capture()

        input_data = self.preprocess_rknn(frame)
        outputs = self.model.inference(inputs=[input_data])
        pred = int(np.argmax(outputs[0].flatten()))
        confidence = float(softmax(outputs[0].flatten())[pred] * 100)
        label = self.class_names[pred]

        self.draw_prediction(frame, pred, confidence)
        cv2.waitKey(1)

        # Publish result
        result_msg = String()
        result_msg.data = f"{label} ({confidence:.1f}%)"
        self.publisher.publish(result_msg)
        self.get_logger().info(f'Prediction: {label} {confidence:.1f}%')

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

def main(args=None):
    rclpy.init(args=args)
    node = VineHealthClassifier()
    rclpy.spin(node)
    node.source.release()
    cv2.destroyWindow(node.win_name)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()