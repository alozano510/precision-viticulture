from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    altitude_listener_node = Node(
        package = 'px4_ros_com',
        executable = 'altitude_listener',
        output = 'screen',
        shell = True)

    return LaunchDescription([altitude_listener_node])

