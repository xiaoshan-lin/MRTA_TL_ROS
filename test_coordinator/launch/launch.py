from launch import LaunchDescription
from launch_ros.actions import Node
import os
import yaml
import datetime
import shutil
from tkinter import simpledialog, messagebox


def generate_launch_description():
    config_path = os.path.join('/home/xslin/Project/marl/configs', 'global_config.yaml')
    robot_config_path = os.path.join('/home/xslin/Project/marl/configs', 'robot_config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_robots = config['environment']['num_robots']
    robot_types = config['environment']['robot types']
    robot_class = config['environment']['robot class']
    num_episodes = config['Q-learning config']['number of episodes']
    num_iterations = config['Q-learning config']['repeat']
    base_seed = config['seed']

    proj_dir = '/home/xslin/Project/marl/result'
    proj_created = False

    while not proj_created:
        try:
            dt = datetime.datetime.today()
            folder_name = simpledialog.askstring('Name your project',
                                                 'Please enter the name of your project',
                                                 initialvalue=f'itrs_{num_iterations}_episode_{num_episodes}')

            proj_dir = os.path.abspath(os.path.join(proj_dir, folder_name))
            os.mkdir(proj_dir)
            shutil.copy(config_path, os.path.join(proj_dir, 'global_config.yaml'))
            shutil.copy(robot_config_path, os.path.join(proj_dir, 'robot_config.yaml'))
            proj_created = True
        except FileExistsError:
            messagebox.showwarning("Warning", "Folder exists. Please try another name.")

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
                {'robot_type': robot_types[robot_id]},  # Assign unique robot_type if needed
                {'robot_class': robot_class[robot_id]},
                {'proj_dir': proj_dir},
                {'base_seed': base_seed}
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
                {'config_path': config_path},
                {'proj_dir': proj_dir}
            ]
        ),

        *robot_nodes
    ])

