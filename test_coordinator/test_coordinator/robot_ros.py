import rclpy
from rclpy.node import Node
from test_interfaces.msg import TaskAllocation
from test_interfaces.srv import RequestAllocation  # Assuming RobotTask service exists
import numpy as np
from MRTAALwTL.robot import Robot
import yaml


class RobotROS(Node):
    def __init__(self, robot, robot_id):
        super().__init__('robot_' + str(robot_id))
        self.robot = robot  # Use the Robot instance
        self.num_robots = self.robot.num_robots
        self.num_tasks = self.robot.num_tasks
        self.client = self.create_client(RequestAllocation, 'send_values')

        # Subscriber to listen to the task allocation
        self.subscription = self.create_subscription(
            TaskAllocation,
            'task_allocations',
            self.task_allocation_callback,
            10)

        # Wait for the coordinator’s service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for coordinator service to be available...')

        self.get_logger().info('Connected to coordinator service')
        values, probs = self.robot.get_values_and_probs()
        self.send_values(values, probs)

    def send_values(self, values, probs):
        request = RequestAllocation.Request()
        request.robot_id = self.robot.robot_id
        request.episode = self.robot.current_episode
        request.robot_values = values
        request.robot_probs = probs
        self.client.call_async(request)  # Asynchronously send the service request

    def task_allocation_callback(self, msg):
        # Convert received data to 2D NumPy array (assume known shape)
        data = np.array(msg.x_values)
        feasible = msg.constraint_satisfied

        if feasible:
            task_allocation = data.reshape(self.num_robots, self.num_tasks + 1)

            probs = np.array(task_allocation[self.robot.robot_id])
            probs = probs / probs.sum()  # normalized
            task = np.random.choice(self.num_tasks + 1, p=probs)

            # Process the task allocation with the Robot class
            stop, values, probs = self.robot.execute(task)

            if stop:
                self.get_logger().info('Reached the desired learning episode, terminating the node.')
                # self.destroy_node()  # Cleanly stop the node
                # rclpy.shutdown()  # Shutdown ROS2
            else:
                self.send_values(values, probs)

        else:
            raise RuntimeError("Unalbe to find feasible task allocation.")


def main(args=None):
    rclpy.init(args=args)

    # Initialize a temporary ROS2 node to handle parameter retrieval
    robot_node = Node('temp_node_to_get_parameters')

    # Get the parameters for the configuration file paths, robot_id, and robot_type
    config_path = robot_node.declare_parameter('config_path', '').value
    robot_config_path = robot_node.declare_parameter('robot_config_path', '').value
    robot_id = robot_node.declare_parameter('robot_id', 0).value
    robot_type = robot_node.declare_parameter('robot_type', 'type_1').value

    # Load YAML configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(robot_config_path, 'r') as f:
        robot_config = yaml.safe_load(f)

    # Create the Robot instance (learning and estimation logic)
    robot = Robot(robot_id, robot_type, config=config, robot_config=robot_config)

    # Destroy the temporary node after parameters are fetched
    robot_node.destroy_node()

    try:
        # Now create the actual ROS interface using the Robot instance
        robot_ros = RobotROS(robot, robot_id)

        # Spin the node (this is the actual ROS2 node handling the logic)
        rclpy.spin(robot_ros)
    except Exception as exception:
        raise exception
    else:
        # Clean up and shutdown
        robot_ros.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
