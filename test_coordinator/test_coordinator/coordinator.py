import rclpy
from rclpy.node import Node
from test_interfaces.srv import RequestAllocation  # Custom service with robot_id and float array
from functools import partial
import numpy as np
from scipy.optimize import minimize
from test_interfaces.msg import TaskAllocation
from concurrent.futures import ThreadPoolExecutor
import yaml
from tqdm import tqdm


class Coordinator(Node):
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

        self.episode = 0
        self.received_values = {}  # Store values from each robot by ID
        self.received_probs = {}

        self.thread_pool_executor = ThreadPoolExecutor()

        # ROS2 service to receive values from robots
        self.srv = self.create_service(RequestAllocation, 'send_values', self.receive_values_callback)

        # Publisher to send the result (task allocations) to robots
        self.publisher = self.create_publisher(TaskAllocation, 'task_allocations', 10)
        self.get_logger().info('Coordinator is ready and waiting for values from robots.')

        self.num_episodes = config['Q-learning config']['number of episodes']

        self.shut_down = False

    def receive_values_callback(self, request, response):
        # Extract the robot ID and values from the request
        robot_id = request.robot_id
        robot_values = request.robot_values  # List of floats
        robot_probs = request.robot_probs
        episode = request.episode

        # self.get_logger().info(f'Received values from Robot {robot_id}: {robot_values}')

        received = robot_id in self.received_values
        mismatched = episode != self.episode
        id_exceed_maximum = robot_id >= self.num_robots

        # Store the robot's values based on the received robot ID
        if not received and not mismatched and not id_exceed_maximum:
            self.received_values[robot_id] = robot_values
            self.received_probs[robot_id] = robot_probs

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

    def publish_task_allocation(self, future):
        # Perform task allocation or any required computation
        x_values, constraint_satisfied = future.result()

        msg = TaskAllocation()
        msg.x_values = x_values.flatten().tolist()  # Flatten and convert to list

        msg.constraint_satisfied = bool(constraint_satisfied)
        self.publisher.publish(msg)
        # self.get_logger().info('Published task allocation')

        # Clear the received values for the next round
        self.episode += 1
        self.received_values.clear()

        if self.episode >= self.num_episodes:
            self.get_logger().info('Reached the desired learning episode, terminating the node.')
            # self.destroy_node()  # Cleanly stop the node
            # rclpy.shutdown()

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
        # print(values)
        # print(probabilities)
        # values = np.array([[0., 0., ]])
        # probabilities = np.array([[0.99922818, 0.99966264, ]])
        # Number of decision variables: n_robot * (n_task + 1)
        n_vars = self.num_robots * (self.num_tasks + 1)

        # Flatten initial guess for decision variables (starting with equal distribution)
        x0 = np.ones(n_vars) / (self.num_tasks + 1)

        # Objective function (maximize sum(x[i, j] * values[i, j]), so minimize the negative)
        def objective(x):
            x = np.array(x).reshape(self.num_robots, self.num_tasks + 1)  # Reshape x into a 2D array
            return -np.sum(x * values)

        # Row sum constraint: sum of probabilities for each robot across all tasks equals 1
        def row_sum_constraint(x, i):
            x = np.array(x).reshape(self.num_robots, self.num_tasks + 1)  # Reshape x into a 2D array
            return np.sum(x[i, :]) - 1  # Sum over all tasks for robot i

        # Nonlinear column probability constraint
        def col_pr_constraint(x, j):
            x = np.array(x).reshape(self.num_robots, self.num_tasks + 1)  # Reshape x into a 2D array
            if j < self.num_tasks:
                product = np.prod(1 - x[:, j] * probabilities[:, j])
                return 1 - product - self.des_probs[j]
            else:
                return 0  # No constraint for the extra task

        # Create constraint dictionaries for minimize function
        cons = []

        # Add row sum constraints (equality)
        for i in range(self.num_robots):
            cons.append({'type': 'eq', 'fun': partial(row_sum_constraint, i=i)})

        # Add column probability constraints (inequality)
        for j in range(self.num_tasks):
            cons.append({'type': 'ineq', 'fun': partial(col_pr_constraint, j=j)})

        # Define bounds (between 0 and 1 for each variable)
        bounds = [(0, 1) for _ in range(n_vars)]

        # Solve the problem using SLSQP solver from scipy.optimize.minimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        # Extract solution and reshape back into the robot-task matrix
        x_values = np.reshape(result.x, (self.num_robots, self.num_tasks + 1))

        # Check if the constraints are satisfied (if optimization succeeded)
        constraint_satisfied = result.success

        if not constraint_satisfied:
            self.get_logger().info(f'Probability: {probabilities}; values: {values}')
            raise RuntimeError("Unalbe to find feasible task allocation.")

        return x_values, constraint_satisfied


def main(args=None):
    rclpy.init(args=args)
    try:
        coordinator = Coordinator()
        rclpy.spin(coordinator)
    except Exception as exception:
        raise exception
    else:
        # Clean up and shutdown
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()