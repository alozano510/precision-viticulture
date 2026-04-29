import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition


class AltitudeListener(Node):
    def __init__(self):
        super().__init__('altitude_listener_node')

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self.listener_callback,
            qos_profile
        )

    def listener_callback(self, msg):
        print('\n' * 24)
        print('RECEIVED DRONE POSITION')
        print('=============================')
        print(f'ts: {msg.timestamp}')
        print(f'Altitude AMSL: {msg.alt} m')
        print(f'Std dev {msg.epv} m')

def main(args=None):
    print('Starting altitude listener node...')
    rclpy.init(args=args)
    node = AltitudeListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()