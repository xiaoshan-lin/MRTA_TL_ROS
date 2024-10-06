import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor
from MRTAALwTL.coordinator import Coordinator

import rclpy
from rclpy.node import Node
from test_interfaces.srv import RequestAllocation, SendData  # Custom service with robot_id and float array
from test_interfaces.msg import TaskAllocation
from std_msgs.msg import String

from tqdm import tqdm
import sys


class CoordinatorROS(Node):
    def __init__(self):
        super().__init__(
            "coordinator",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        config_path = self.get_parameter('config_path').get_parameter_value().string_value

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.num_robots = config['environment']['num_robots']
        self.des_probs = [task["desired satisfaction probability"]
                          for task in config['TWTL constraint']['tasks'].values()]
        self.num_tasks = len(self.des_probs)
        self.repeat_iters = config['Q-learning config']['repeat']
        self.num_episodes = config['Q-learning config']['number of episodes']

        self.coordinator = Coordinator(self.num_robots, self.num_tasks, self.des_probs)

        self.repeat_count = 0
        self.episode = 0
        self.received_values = {}  # Store values from each robot by ID
        self.received_probs = {}
        self.backup_probs = {}
        self.received_rewards = {}
        self.received_twtl_sat = {}

        self.thread_pool_executor = ThreadPoolExecutor()

        # ROS2 service to receive values from robots
        self.srv = self.create_service(RequestAllocation, 'send_values', self.receive_values_callback)
        self.data_srv = self.create_service(SendData, 'send_data', self.receive_data_callback)

        # Publisher to send the result (task allocations) to robots
        self.publisher = self.create_publisher(TaskAllocation, 'task_allocations', 10)
        self.string_publisher = self.create_publisher(String, 'proj_dir', 10)

        # self.progress_bar = tqdm(total=self.num_episodes, desc="Progress", unit="episode")

        self.get_logger().info('Coordinator is ready and waiting for values from robots.')

    def publish_projdir(self):
        # Create and publish the message
        msg = String()
        msg.data = self.coordinator.proj_dir
        self.string_publisher.publish(msg)

    def receive_values_callback(self, request, response):
        # Extract the robot ID and values from the request
        robot_id = request.robot_id
        robot_values = request.robot_values  # List of floats
        robot_probs = request.robot_probs
        backup_probs = request.backup_probs
        episode = request.episode

        # self.get_logger().info(f'Received values from Robot {robot_id}: {robot_values}')

        received = robot_id in self.received_values
        mismatched = episode != self.episode
        id_exceed_maximum = robot_id >= self.num_robots

        # Store the robot's values based on the received robot ID
        if not received and not mismatched and not id_exceed_maximum:
            self.received_values[robot_id] = robot_values
            self.received_probs[robot_id] = robot_probs
            self.backup_probs[robot_id] = backup_probs

            message, accepted = f'Received values from Robot {robot_id}', True
            # Check if all robots have sent their values
            if len(self.received_values) == self.num_robots:
                # Run task allocation asynchronously
                future = self.thread_pool_executor.submit(self.compute_prob)
                future.add_done_callback(self.publish_task_allocation)

        else:
            message, accepted = '', False
            if mismatched:
                message += f'Received values from Robot {robot_id} for episode {episode}' + \
                           f'but the current episode accepted is {self.episode}'
            if received:
                message += f'Values from Robot {robot_id} for episode {episode} already received'

            if id_exceed_maximum:
                message += f'Robot {robot_id} exceeds the maximum number of robots'

        response.accepted = accepted
        response.message = message

        return response  # Immediately return the response to the robot

    def receive_data_callback(self, request, response):
        robot_id = request.robot_id
        repeat_count = request.repeat_count
        ep_rewards = request.ep_rewards
        twtl_sat_flat = request.twtl_sat

        # Reconstruct the 2D list from the flattened twtl_sat
        twtl_sat = [twtl_sat_flat[i:i + self.num_tasks] for i in range(0, len(twtl_sat_flat), self.num_tasks)]

        # self.get_logger().info(f'Received data from Robot {robot_id}: {ep_rewards}')
        # self.get_logger().info(f'Received data from Robot {robot_id}: {twtl_sat}')

        self.received_rewards[robot_id] = ep_rewards
        self.received_twtl_sat[robot_id] = twtl_sat

        # Print received data or process it
        self.get_logger().info(f'Received data from Robot {robot_id}')

        if len(self.received_rewards) == self.num_robots:
            # Run task allocation asynchronously
            future = self.thread_pool_executor.submit(self.process_data, repeat_count)
            future.add_done_callback(self.save_results)

        # Send response indicating success
        response.accepted = True
        return response

    def process_data(self, repeat_count):
        # Step 1: Construct the total ep_rewards array
        ep_rewards_array = np.zeros((self.num_episodes, self.num_robots))  # Initialize the array

        for robot_id, ep_rewards in self.received_rewards.items():
            ep_rewards_array[:, robot_id] = ep_rewards  # Fill in the ep_rewards for each robot

        # Step 2: Combine self.received_twtl_sat
        combined_twtl_sat = [[0] * self.num_tasks for _ in range(self.num_episodes)]  # Initialize the combined twtl_sat

        for p in range(self.num_episodes):  # For each episode
            for idx in range(self.num_tasks):  # For each task
                task_success = False  # Default to failure for this task

                # Check all robots' results for this task in episode p
                for robot_id in range(self.num_robots):
                    if self.received_twtl_sat[robot_id][p][idx] == 1:
                        task_success = True  # If any robot has success, set to True
                        break  # No need to check further, this task is successful

                combined_twtl_sat[p][idx] = 1 if task_success else 0  # Set the combined result for this task

        return ep_rewards_array, combined_twtl_sat, repeat_count

    def save_results(self, future):
        ep_rewards, twtl_sat, repeat_count = future.result()
        self.coordinator.save_results(ep_rewards, twtl_sat, repeat_count)

    def publish_task_allocation(self, future):
        # Perform task allocation or any required computation
        x_values, constraint_satisfied = future.result()

        # if self.episode < 60:
        # self.get_logger().info(f'{x_values}')

        msg = TaskAllocation()
        msg.x_values = x_values.flatten().tolist()  # Flatten and convert to list

        msg.constraint_satisfied = bool(constraint_satisfied)
        self.publisher.publish(msg)

        # Clear the received values for the next round
        self.episode += 1
        self.get_logger().info(f'Episode: {self.episode}')

        self.received_values.clear()
        self.received_probs.clear()
        self.backup_probs.clear()

        if self.episode >= self.num_episodes:
            self.repeat_count += 1
            # TODO: save the results
            if self.repeat_count >= self.repeat_iters:
                self.get_logger().info(f'Reached the desired learning episode for {self.repeat_iters} iterations')
                self.publish_projdir()
                # self.plot_results()
            else:
                self.get_logger().info(f'Reached the desired learning episode for iteration {self.repeat_count}')
                self.reset()  # reset and wait for the next iteration

    def compute_prob(self):
        """
        This function implements a heuristic method to solve the optimization problem using SciPy's minimize function.
        Args:
        Returns:
            List: List of switching probabilities
            Bool: Bool variable to indicate if the constraints are satisfied or not
        """
        values = np.array([[v for v in self.received_values[i]] for i in range(self.num_robots)])
        probabilities = np.array([[p for p in self.received_probs[i]] for i in range(self.num_robots)])
        backup_probs = np.array([[p for p in self.backup_probs[i]] for i in range(self.num_robots)])

        x_values, constraint_satisfied = self.coordinator.compute_prob(values, probabilities)

        if not constraint_satisfied:
            x_values, constraint_satisfied = self.coordinator.compute_prob(values, backup_probs)
            if not constraint_satisfied:
                self.get_logger().info(f'Probability: {backup_probs}; values: {values}')
                raise RuntimeError("Unable to find feasible task allocation.")

        return x_values, constraint_satisfied

    def reset(self):
        self.episode = 0
        self.received_values = {}  # Store values from each robot by ID
        self.received_probs = {}

    def plot_results(self):
        self.coordinator.plot_results()


def main(args=None):
    rclpy.init(args=args)
    try:
        coordinator = CoordinatorROS()
        rclpy.spin(coordinator)
    except Exception as exception:
        raise exception
    else:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()