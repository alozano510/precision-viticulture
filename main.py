import argparse
import threading
import cv2
from cmd import Cmd

from src.vision_system.vine_health_classifier import VineHealthClassifier
from src.drone.drone_mavlink_communication import DroneControl
from src.web_dashboard.dashboard_server import DashboardServer

'''
 Get program settings for command line.
--camera: all the camera devices connected to the computer as integers starting from 0. If not configured, the first camera is selected (0).
--drone_port: the communication port to which the drone flight controller is connected. 
'''
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera', type=int, default=0)
parser.add_argument('-p', '--drone_port', type=str, default='/dev/ttyS4')
parser.add_argument('-s', '--simulator', type=bool, default=False)
parser.add_argument('-g', '--graphics', type=bool, default=False)
parser.add_argument('-d', '--dashboard', type=bool, default=False)
args = parser.parse_args()

class DroneShell(Cmd):
    intro = "Welcome to the drone shell. Type help to list commands.\n"
    prompt = "(drone)"

    def __init__(self, drone, vine_classifier):
        super().__init__()
        self.drone = drone
        self.vine_classifier = vine_classifier
        self._active_thread = None

    def _stop_active(self):
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

        def vine_analysis():
            """Runs both the route and the analysis functions in parallel threads"""
            route_thread = threading.Thread(target=self.drone.analysis_route, daemon=True)
            analysis_thread = threading.Thread(target=self.vine_classifier.run_analysis, daemon=True)

            route_thread.start()
            analysis_thread.start()

            route_thread.join()
            analysis_thread.join()

        self._active_thread = threading.Thread(target=vine_analysis, daemon=True)
        self._active_thread.start()
        print("Starting vine analysis...")

    def do_hybrid_vision(self, arg):
        """Starts the hybrid vision system without initializing a drone mission."""
        self._stop_active()
        self._active_thread = threading.Thread(target=self.vine_classifier.hybrid_analysis, daemon=True)
        self._active_thread.start()
        print(f"Starting vision system...")

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
    port = args.drone_port
    if args.simulator:
        from src.vision_system.vine_health_classifier_pytorch import VineHealthClassifierTorch
        port = 'tcp:127.0.0.1:5763'
        vine_classifier = VineHealthClassifierTorch(args.camera)
    else:
        vine_classifier = VineHealthClassifier(args.camera)

    if not args.graphics and not args.simulator:
        # Video stream via RTSP
        from src.streaming.rtsp_stream_server import RTSPStreamServer
        rtsp = RTSPStreamServer(
            fps=24,
            width=640,
            height=480,
            use_hw_encoder=True,
        )
        rtsp.set_frame_source(vine_classifier.get_latest_frame)
        rtsp.start()

    # Flask server for control dashboard
    dashboard = None
    if args.dashboard:
        dashboard = DashboardServer(port=5000)
        dashboard.start()
    
    drone = DroneControl(port, dashboard=dashboard)

    drone_shell = DroneShell(drone, vine_classifier)

    # If selected, run video feed locally
    if args.graphics:
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

    if args.graphics:
        cv2.destroyAllWindows()

    drone.close()
    print("Program terminated.")

if __name__ == '__main__':
    main()
