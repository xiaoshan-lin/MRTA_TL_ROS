from launch import LaunchDescription
from launch_ros.actions import Node
import os
import yaml


def generate_launch_description():
    config_path = os.path.join('/home/xslin/Project/marl/configs', 'config.yaml')
    robot_config_path = os.path.join('/home/xslin/Project/marl/configs', 'robot_config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_robots = config['environment']['num_robots']
    robot_types = config['environment']['robot types']

    robot_nodes = [
        Node(
            package='test_coordinator',  # Replace with your package name
            executable='client',  # Replace with your node executable name
            name=f'robot_{robot_id}',  # Unique node name per robot
            output='screen',
            parameters=[
                {'config_path': config_path},
                {'robot_config_path': robot_config_path},
                {'robot_id': robot_id},  # Assign unique robot_id
                {'robot_type': robot_types[robot_id]}  # Assign unique robot_type if needed
            ]
        ) for robot_id in range(num_robots)  # Adjust range based on the number of robots
    ]

    return LaunchDescription([
        Node(
            package='test_coordinator',  # Replace with your package name
            executable='service',  # Replace with the name of your node executable
            name='coordinator',  # Name of the node
            output='screen',  # Prints the node output to the terminal
            parameters=[
                {'config_path': config_path}
            ]
        ),

        *robot_nodes
    ])

