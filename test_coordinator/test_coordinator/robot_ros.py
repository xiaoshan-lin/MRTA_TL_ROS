import rclpy
from rclpy.node import Node
from test_interfaces.msg import TaskAllocation
from test_interfaces.srv import RequestAllocation, SendData  # Assuming RobotTask service exists
import numpy as np
from MRTAALwTL.robot import Robot
import yaml


class RobotROS(Node):
    def __init__(self):
        super().__init__('robot',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True,
                         )

        config_path = self.get_parameter('config_path').value
        robot_config_path = self.get_parameter('robot_config_path').value
        robot_id = self.get_parameter('robot_id').value
        robot_type = self.get_parameter('robot_type').value
        robot_class = self.get_parameter('robot_class').value
        proj_dir = self.get_parameter('proj_dir').value
        base_seed = self.get_parameter('base_seed').value

        # Load YAML configurations
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        with open(robot_config_path, 'r') as f:
            robot_config = yaml.safe_load(f)

        # Create the Robot instance (learning and estimation logic)
        robot = Robot(robot_id, robot_class, robot_type, config, robot_config, proj_dir, base_seed)

        self.robot = robot  # Use the Robot instance
        self.num_robots = self.robot.num_robots
        self.num_tasks = self.robot.num_tasks
        self.client = self.create_client(RequestAllocation, 'send_values')
        self.data_client = self.create_client(SendData, 'send_data')

        # Subscriber to listen to the task allocation
        self.subscription = self.create_subscription(
            TaskAllocation,
            'task_allocations',
            self.task_allocation_callback,
            10)

        # self.string_subscription = self.create_subscription(
        #     String,
        #     'proj_dir',
        #     self.pickle_callback,
        #     10)

        # Wait for the coordinator’s service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for coordinator service to be available...')

        while not self.data_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

        self.get_logger().info('Connected to coordinator service')

        self.send_values()

    # def pickle_callback(self, msg):
    #     proj_dir = msg.data
    #     self.get_logger().info(f'Robot {self.robot.robot_id} received project dir')

    def send_values(self):
        values, probs, backup_probs = self.robot.get_values_and_probs()
        values = np.array(values, dtype=np.float32).tolist()
        probs = np.array(probs, dtype=np.float32).tolist()
        backup_probs = np.array(backup_probs, dtype=np.float32).tolist()

        # self.get_logger().info(f'mdp:{self.robot.current_mdp_state}')
        # self.get_logger().info(f'lb: {self.robot.estimator.lower_bound[0][self.robot.current_mdp_state]}')
        # a = self.robot.estimator.result_count[0][self.robot.current_mdp_state]['number']
        # self.get_logger().info(f'lb: {a}')

        request = RequestAllocation.Request()
        request.robot_id = self.robot.robot_id
        request.episode = self.robot.current_episode
        request.robot_values = values
        request.robot_probs = probs
        request.backup_probs = backup_probs
        self.client.call_async(request)  # Asynchronously send the service request

    def task_allocation_callback(self, msg):
        # Convert received data to 2D NumPy array (assume known shape)
        data = np.array(msg.x_values)
        feasible = msg.constraint_satisfied

        if feasible:
            task_allocation = data.reshape(self.num_robots, self.num_tasks + 1)

            # Process the task allocation with the Robot class
            status = self.robot.execute(task_allocation)

            # self.get_logger().info(f'robot {self.robot.robot_id}, result: {action_result}')
            # self.get_logger().info(f'lb: {self.robot.estimator.lower_bound[0][self.robot.current_mdp_state]}')

            if status == 'Continue':
                self.send_values()

            else:
                self.send_results()
                if status == 'Iteration end':
                    self.get_logger().info(f'Reached the desired learning episode for iteration {self.robot.repeat_count}')
                    self.robot.pickle()
                    self.reset()
                    self.send_values()
                else:
                    self.get_logger().info(f'Reached the desired learning episode for {self.robot.repeat_iters} iterations')
                    self.robot.pickle()
        else:
            raise RuntimeError("Unalbe to find feasible task allocation.")

    def reset(self):
        self.robot.reset()

    def send_results(self):
        ep_rewards, twtl_sat = self.robot.get_results()
        flattened_twtl_sat = [item for sublist in twtl_sat for item in sublist]

        request = SendData.Request()
        request.repeat_count = self.robot.repeat_count - 1 # TODO
        request.robot_id = self.robot.robot_id
        request.ep_rewards = ep_rewards.tolist()  # Convert numpy array to list
        request.twtl_sat = flattened_twtl_sat

        self.data_client.call_async(request)


def main(args=None):
    rclpy.init(args=args)

    try:
        # Now create the actual ROS interface using the Robot instance
        robot = RobotROS()
        rclpy.spin(robot)
    except Exception as exception:
        raise exception
    else:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
