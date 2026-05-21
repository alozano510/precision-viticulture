import argparse
import threading
from cmd import Cmd
from vine_health_classifier import VineHealthClassifier
from drone_mavlink_communication import DroneControl
from dashboard_server import DashboardServer

'''
 Get program settings for command line.
--camera: all the camera devices connected to the computer as integers starting from 0. If not configured, the first camera is selected (0).
--drone_port: the communication port to which the drone flight controller is connected. 
'''
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera', type=int, default=0)
parser.add_argument('-p', '--drone_port', type=str, default='/dev/ttyS4')
parser.add_argument('-s', '--simulator', type=bool, default=False)
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

    def do_manual_vision(self, arg):
        """Starts the vision system without initializing a drone mission."""
        self._stop_active()
        self._active_thread = threading.Thread(target=self.vine_classifier.run_analysis, daemon=True)
        self._active_thread.start()
        print(f"Starting vision system...")

def main():
    port = args.drone_port
    if args.simulator:
        from vine_health_classifier_pytorch import VineHealthClassifierTorch
        port = 'tcp:127.0.0.1:5763'
        vine_classifier = VineHealthClassifierTorch(args.camera)
    else:
        vine_classifier = VineHealthClassifier(args.camera)

    dashboard = DashboardServer(port=5000)
    dashboard.set_frame_source(vine_classifier.get_latest_frame)
    dashboard.start()

    drone = DroneControl(port, dashboard=dashboard)

    drone_shell = DroneShell(drone, vine_classifier)
    drone_shell.cmdloop()

    vine_classifier.stop()
    drone.close()
    print("Program terminated.")

if __name__ == '__main__':
    main()