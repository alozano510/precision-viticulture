import argparse
import threading
import json
import cv2
from cmd import Cmd

from src.vision_system.vine_health_classifier import VineHealthClassifier
from src.drone.drone_mavlink_communication import DroneControl
from src.web_dashboard.dashboard_server import DashboardServer

'''
 Get program settings for command line.
--camera: all the camera devices connected to the computer as integers starting from 0. If not configured, the first camera is selected (0).
'''

# Values for RTSP stream server
FPS = 24
WIDTH = 640
HEIGHT = 480
# PORT for Flask server dashboard
FLASK_PORT = 5000

with open('config.json') as config_file:
    config = json.load(config_file)

parser = argparse.ArgumentParser()
parser.add_argument( 'cnn_model_path', type=str)
parser.add_argument('yolo_model_path', type=str)
parser.add_argument('-c', '--camera', type=int, default=0)
parser.add_argument('-lc', '--leaf_detection_conf_threshold', type=float, default=0.25)
parser.add_argument('-iou', '--leaf_detection_iou_threshold', type=float, default=0.45)
parser.add_argument('-hu', '--hybrid_unhealthy_threshold', type=float, default=0.4)
args = parser.parse_args()

class DroneShell(Cmd):
    intro = "Welcome to the drone shell. Type help to list commands.\n"
    prompt = "(drone)"

    def __init__(self, drone: DroneControl, vine_classifier: VineHealthClassifier):
        super().__init__()
        self.drone = drone
        self.vine_classifier = vine_classifier
        self._active_thread = None

    def _stop_active(self):
        if self.drone is not None:
            self.drone._running = False
        self.vine_classifier._running = False
        if self._active_thread and self._active_thread.is_alive():
            self._active_thread.join()

    def do_stop(self, arg):
        """Terminates program"""
        self._stop_active()
        return True

    def do_altitude_control(self, ref):
        """Controls the altitude of the drone at a given reference in meters. Type 'altitude_control <reference>'"""
        self._stop_active()
        ref = float(ref)
        if not self.drone:
            print("There is no drone or drone simulator connected. Connect to drone or simulator and restart the program to use this command.")
        else:
            self._active_thread = threading.Thread(target=self.drone.altitude_control, args=(ref,), daemon=True)
            self._active_thread.start()
            print(f"Starting altitude control at {ref} meters...")

    def do_ref(self, arg):
        """Change the target altitude during flight. Usage: ref <meters>"""
        try:
            new_ref = float(arg)
            self.drone._ref = new_ref
            print(f"Reference altitude changed to {new_ref}m")
        except (ValueError, IndexError):
            print("Invalid input. Usage: ref <value>")

    def do_vine_analysis(self, arg):
        """Starts the preloaded mission with a velocity of 2 m/s and analyzes plants every 0.5 seconds"""
        self._stop_active()
        if not self.drone:
            print("There is no drone or drone simulator connected. Connect to drone or simulator and restart the program to use this command.")
        else:
            def vine_analysis():
                """Runs both the route and the analysis functions in parallel threads"""
                route_thread = threading.Thread(target=self.drone.analysis_route, daemon=True)
                analysis_thread = threading.Thread(target=self.vine_classifier.hybrid_analysis, daemon=True)

                route_thread.start()
                analysis_thread.start()

                route_thread.join()
                analysis_thread.join()

            self._active_thread = threading.Thread(target=vine_analysis, daemon=True)
            self._active_thread.start()
            print("Starting vine analysis...")

    def do_manual_vision(self, model: str = None):
        """
        Starts the vision system without initializing a drone mission. A model must be specified (classification/detection/hybrid).
        Type manual_vision <classification/detection/hybrid>
        """
        self._stop_active()

        valid_models = {"classification", "detection", "hybrid"}

        while model not in valid_models:
            if model is not None:
                print(f"Invalid model '{model}'. Valid options: {', '.join(valid_models)}")
            model = input("Enter model (classification/detection): ").strip().lower()

        if model == "classification":
            self._active_thread = threading.Thread(target=self.vine_classifier.run_leaf_classification, daemon=True)
        elif model == "detection":
            self._active_thread = threading.Thread(target=self.vine_classifier.run_leaf_detection, daemon=True)
        elif model == "hybrid":
            self._active_thread = threading.Thread(target=self.vine_classifier.hybrid_analysis, daemon=True)

        self._active_thread.start()
        print(f"Starting vision system...")

def main():
    # Flask server for control dashboard
    dashboard = None
    if config.get('dashboard'):
        dashboard = DashboardServer(port=FLASK_PORT)
        dashboard.start()

    drone_config = config.get('drone_config')
    vine_health_classifier_config = config.get('vine_health_classifier')

    if config.get('orangepi'):
        vine_classifier = VineHealthClassifier(
            camera=args.camera,
            fps=vine_health_classifier_config.get('fps'),
            camera_config=config.get('camera'),
            cnn_model_path=args.cnn_model_path,
            yolo_model_path=args.yolo_model_path,
            leaf_detection_conf_threshold=args.leaf_detection_conf_threshold,
            leaf_detection_iou_threshold=args.leaf_detection_iou_threshold,
            hybrid_unhealthy_threshold=args.hybrid_unhealthy_threshold
        )
        if config.get('drone'):
            port = drone_config.get('port')
            drone = DroneControl(port, dashboard=dashboard, baudrate=drone_config.get('communication_baudrate'))
        else:
            drone = None
    else:
        from src.vision_system.vine_health_classifier_pytorch import VineHealthClassifierTorch
        vine_classifier = VineHealthClassifierTorch(
            camera=args.camera,
            fps=vine_health_classifier_config.get('fps'),
            camera_config=config.get('camera'),
            cnn_model_path=args.cnn_model_path,
            yolo_model_path=args.yolo_model_path,
            leaf_detection_conf_threshold=args.leaf_detection_conf_threshold,
            leaf_detection_iou_threshold=args.leaf_detection_iou_threshold,
            hybrid_unhealthy_threshold=args.hybrid_unhealthy_threshold
        )
        if config.get('simulator'):
            port = config.get('sitl_simulation_port')
            drone = DroneControl(port, dashboard=dashboard, baudrate=drone_config.get('communication_baudrate'))
        else:
            drone = None

    if not config.get('graphics') and not config.get('simulator'):
        # Video stream via RTSP
        from src.streaming.rtsp_stream_server import RTSPStreamServer
        rtsp_config = config.get('rtsp_streamer')
        rtsp = RTSPStreamServer(
            host=rtsp_config.get('host'),
            port=rtsp_config.get('port'),
            path=rtsp_config.get('path'),
            fps=rtsp_config.get('fps'),
            width=rtsp_config.get('width'),
            height=rtsp_config.get('height'),
            use_hw_encoder=True,
        )
        rtsp.set_frame_source(vine_classifier.get_latest_frame)
        rtsp.start()

    drone_shell = DroneShell(drone, vine_classifier)

    # If selected, run video feed locally
    if config.get('graphics'):
        shell_thread = threading.Thread(target=drone_shell.cmdloop, daemon=True)
        shell_thread.start()

        # Main thread runs the image display for the vision models
        cv2.namedWindow("Vine health classifier", cv2.WINDOW_NORMAL)
        while shell_thread.is_alive():
            frame = vine_classifier.get_latest_frame()
            if frame is not None:
                cv2.imshow("Vine health classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                drone_shell._stop_active()
                break
    else:
        drone_shell.cmdloop()

    vine_classifier.stop()

    if config.get('graphics'):
        cv2.destroyAllWindows()

    if config.get('drone') or config.get('simulator'):
        drone.close()

    print("Program terminated.")

if __name__ == '__main__':
    main()
