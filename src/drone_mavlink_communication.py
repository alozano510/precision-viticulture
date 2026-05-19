import time
import threading
from dronekit import connect, VehicleMode, LocationGlobalRelative
from scipy.stats import false_discovery_control

"""
Notas para bitácora:
La librería de Dronekit permite el cambio de parámetros en el dron, pero no asegura que estos se hagan.
No hay soporte para saber si el intercambio de datos fue exitoso o no.
Actualmente la liberería ya no recibe soporte, por lo que estos features nunca serán arreglados.
"""

class DroneControl:
    def __init__(self, port: str = '/dev/ttyS0'):
        self.drone = connect(port, baud=57600, wait_ready=True)
        if self.drone is not None:
            print("Successfully connected to drone")

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
        # Slow down vertical navigation speed for gentler altitude changes
        self.drone.parameters['WPNAV_SPEED_DN'] = 50  # cm/s (default is 150)
        self.drone.parameters['WPNAV_SPEED_UP'] = 100  # cm/s

        # Give some time for the parameters to change
        while not self.drone.parameters['WPNAV_SPEED_DN'] == 50 and self.drone.parameters['WPNAV_SPEED_UP'] == 100:
            print("Adjusting speed...")
            time.sleep(1)

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

        # Set the target altitude
        while self._running:
            location = self.drone.location.global_relative_frame
            if not self._ref * 0.95 <= location.alt <= self._ref * 1.05:
                reached = False
                print(f"Altitude: {location.alt:.2f}m")
                new_location = LocationGlobalRelative(location.lat, location.lon, self._ref)
                self.drone.simple_goto(new_location)
            else:
                if not reached:
                    reached = True
                    print(f"Altitude: {location.alt:.2f}m")
                    print("Reference altitude reached.")
            time.sleep(1)

        print("Altitude control stopped")

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

    def _plot_control_state(self):
        # TODO: implement function to show control plots. This has to somehow always work. Maybe implement it as part of the altitude control function. Check section on 'Observing attribute changes' in dronekit documentation.
        alt = self.drone.location.global_relative_frame.alt
        vel = self.drone.velocity
        speed = self.drone.groundspeed
"""
    def _cli_listener(self):
        print("Altitude control active.")
        print("Commands: 'exit' to stop | 'ref <value>' to change altitude")
        while self._running:
            user_input = input("> ").strip()
            if user_input.lower() == "exit":
                self._running = False
            elif user_input.lower().startswith("ref "):
                try:
                    new_ref = float(user_input.split()[1])
                    self._ref = new_ref
                    print(f"Reference altitude changed to {self._ref}m")
                except (ValueError, IndexError):
                    print("Invalid input. Usage: ref <value>")
            else:
                print("Unknown command. Use 'exit' or 'ref <value>'")
"""
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