import csv
import time
import threading
import os
from datetime import datetime

from dronekit import connect, VehicleMode, LocationGlobalRelative
from dashboard_server import DashboardServer

"""
Notas para bitácora:
La librería de Dronekit permite el cambio de parámetros en el dron, pero no asegura que estos se hagan.
No hay soporte para saber si el intercambio de datos fue exitoso o no.
Actualmente la liberería ya no recibe soporte, por lo que estos features nunca serán arreglados.
"""

class DroneControl:
    def __init__(self, port: str = '/dev/ttyS0',
                 dashboard: DashboardServer = None,):
        self.drone = connect(port, baud=57600, wait_ready=True)
        if self.drone is not None:
            print("Successfully connected to drone")

        self._ref = None

        self._log_data = []
        self._logging = False
        self._log_thread = None
        self._dashboard = dashboard

    def take_off(self):
        """Arms drone and prepares it to take off"""
        #Asegurar que el dron tiene una buena localizacion tomada con el GPS
        while not self.drone.home_location:
            cmds = self.drone.commands
            cmds.download()
            cmds.wait_ready()
            if not self.drone.home_location:
                print("Waiting for GPS location...")

        if not self.drone.armed:
            print("Arming motors...")
            self.drone.armed = True
        # Make sure that the commands where changed
        while not self.drone.armed:
            print("Getting ready to take off ...")
            time.sleep(1)

        print("Motors armed")
        # Give some time to the autopilot to stabilize
        time.sleep(2)
        if self.drone.location.global_relative_frame.alt < 0.2: # given the drone has an altitude accuracy of 0.5, adjust if necessary
            print("Taking off...")
            self.drone.simple_takeoff(alt=1) # 1 meter
            # Give some time to the drone to gain elevation
            time.sleep(3)

    def close(self):
        self.drone.close()

    def altitude_control(self, ref: float):

        self.drone.mode = VehicleMode('GUIDED')
        while not self.drone.mode.name == 'GUIDED':
            print("Changing drone mode to GUIDED...")
            time.sleep(1)
        print("Drone mode GUIDED")

        self.take_off()

        self._running = True
        self._ref = ref
        # Flag for reaching the reference altitude
        reached = False

        self.start_logging()

        # Set the target altitude
        try:
            while self._running:
                location = self.drone.location.global_relative_frame
                if not self._ref * 0.95 <= location.alt <= self._ref * 1.05:
                    reached = False
                    print(f"Altitude: {location.alt:.2f}m")
                    new_location = LocationGlobalRelative(
                        location.lat, location.lon, self._ref)
                    self.drone.simple_goto(new_location)
                else:
                    if not reached:
                        reached = True
                        print(f"Altitude: {location.alt:.2f}m — Reference reached.")
                time.sleep(1)

        except KeyboardInterrupt:
            print("Interrupted by user.")
            self._running = False

        finally:
            self.land()
            csv_path = self.stop_logging()
            print("Altitude control stopped.")

    def analysis_route(self):
        self.drone.mode = VehicleMode('AUTO')
        while not self.drone.mode.name == 'AUTO':
            print("Changing drone mode...")
            time.sleep(2)
        print(self.drone.mode.name)
        print("Changed drone mode to AUTO")

    def land(self):
        self.drone.parameters['LAND_SPEED'] = 30 # cm/s
        self.drone.parameters['LAND_SPEED_HIGH'] = 60 # cm/s

        # Give some time for the parameters to change
        time.sleep(1)

        print("Landing...")
        self.drone.mode = VehicleMode('LAND')
        while not self.drone.mode.name == 'LAND':
            print("Switching to LAND mode...")
            time.sleep(1)

        while self.drone.location.global_relative_frame.alt > 0.15:
            print(f"Altitude: {self.drone.location.global_relative_frame.alt:.2f}m")
            time.sleep(1)

        print("Landed.")
        self.drone.armed = False

    def start_logging(self, sample_period: float = 0.2):
        """
        Starts background logging thread.

        Args:
            sample_period (float, optional): Sample period in seconds. Defaults to 0.2.
        """
        self._logging = True
        self._log_data = []
        self._log_thread = threading.Thread(
            target=self._log_loop,
            args=(sample_period,),
            daemon=True
        )
        self._log_thread.start()
        print(f"Logging started at {1 / sample_period:.1f} Hz")

    def _log_loop(self, sample_period: float):
        """
        Background thread: samples drone state and appends to log
        Args:
            sample_period (float, optional): Sample period in seconds.
        """
        t0 = time.time()
        while self._logging:
            try:
                t = time.time() - t0
                alt = self.drone.location.global_relative_frame.alt
                error = self._ref - alt
                error_per = (error / self._ref) * 100
                current = self.drone.battery.current or 0.0  # Amps; -1 if unsupported
                entry = {
                    'time_s': round(t, 3),
                    'altitude_m': round(alt, 3),
                    'reference_m': round(self._ref, 3),
                    'error_m': round(error, 3),
                    'error_per': round(error_per, 3),
                    'current_a': round(current, 3),
                }
                self._log_data.append(entry)

                if self._dashboard:
                    self._dashboard.emit(entry)

            except Exception as e:
                print(f"Logging error: {e}")

            time.sleep(sample_period)

    def stop_logging(self):
        """
        Stops logging, saves CSV, and returns the log data.
        Returns:
            Path to the saved CSV file
        """
        self._logging = False
        if self._log_thread:
            self._log_thread.join(timeout=2)

        if not self._log_data:
            print("No data to save.")
            return None

        export_path = os.path.join(os.path.dirname(__file__), '..', 'control_log')
        os.makedirs(export_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(export_path, f'altitude_log_{timestamp}.csv')

        fieldnames = self._log_data[0].keys()
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._log_data)

        print(f"Log saved to: {filepath}")
        return filepath

    def _plot_control_state(self):
        # TODO: implement function to show control plots. This has to somehow always work. Maybe implement it as part of the altitude control function. Check section on 'Observing attribute changes' in dronekit documentation.
        raise NotImplementedError

"""
Lógica de uso:

Metas:
- Controlar altura de dron
  - Botón para iniciar modo de control de altura
  - Interfaz para establecer la referencia y la altura deseada
  - Stream de altura en tiempo real para graficar los cambios
  - Stream de cambio de energía para graficas los cambios:
    - Puedo obtener Groundspeed o Velocity TODO:investigar diferencia
    - TODO: investigar si puedo obtener la velocidad de los motores
- Tomar captura de plantas para clasificación:
  Opciones para el control de vuelo para esta parte:
  - Escenario de aplicación real:
    - Cargar misión con Waypoints usando QGC. Necesitaría posicionar muy precisamente las plantas
  - Para demostración de funcionamiento de clasificación:
    - Controlar dron manualmente con RC y ver la forma de programar un botón del RC para iniciar el escaneo

Funcionamiento a largo plazo:
1. Diseñar una ruta que siga el dron
  - Esta ruta puede hacerse desde QGC
2. Al ejecutar el programa, el dron inicia automáticamente su ruta
3. Avanza a una velocidad de 2 m/s, analizando las plantas cada 0.5 seg.
4. Guarda una captura de las plantas que requieran algún tipo de tratamiento junto con su localización
5. En una computadora remota, el usuario puede ver el feed en tiempo real del dron así como su posición. Para esto necesitaría correr QGC + la terminal del programa. 
"""