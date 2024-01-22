import casadi as ca
import numpy as np
from model import DiffDrive
import math
import matplotlib.pyplot as plt
from metrics_logger import MetricsLogger
from matplotlib.patches import Circle, Arrow
from matplotlib.animation import FuncAnimation

class MPC_Base:
    def __init__(self, initial_state, final_state, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=None, ref=None, feedback=None):
        self.num_agent = mpc_params['num_agents']
        self.dt = mpc_params['dt']
        self.N = mpc_params['N']
        self.rob_dia = mpc_params['rob_dia']
        self.v_lim = mpc_params['v_lim']
        self.omega_lim = mpc_params['omega_lim']
        self.total_sim_timestep = mpc_params['total_sim_timestep']
        self.goal_tolerence = mpc_params['goal_tolerence']
        self.epsilon_o = mpc_params['epsilon_o']
        self.epsilon_r = mpc_params['epsilon_r']
        self.safety_margin = mpc_params['safety_margin']
        self.linearized_ca = mpc_params['linearized_ca']
        self.initial_state = initial_state
        self.final_state = final_state
        self.cost_func_params = cost_func_params
        self.scenario = scenario
        self.trial = trial
        self.uncontrolled_fleet = uncontrolled_fleet
        self.uncontrolled_fleet_data = uncontrolled_fleet_data
        self.delta = 0.03
        self.num_modes = 3
        self.robust_horizon = 2
        self.feedback = feedback

        self.model = DiffDrive(self.rob_dia)

        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N)) for agent_id in range(self.num_agent)}

        # variables holding previous solutions
        self.prev_states = {agent_id: np.zeros((self.N+1, 3)) for agent_id in range(self.num_agent)}
        self.prev_controls = {agent_id: np.zeros((self.N, 2)) for agent_id in range(self.num_agent)}
        self.prev_epsilon_o = {agent_id: np.zeros((self.N+1, 1)) for agent_id in range(self.num_agent)}
        self.prev_epsilon_r = {agent_id: np.zeros((self.N+1, 1)) for agent_id in range(self.num_agent)}
        
        self.current_state = {}
        for i in range(self.num_agent):
            self.current_state[i] = self.initial_state[i]

        self.obs = obs
        self.dyn_obs = obs["dynamic"]
        self.static_obs = obs["static"]

        self.ref = ref
        self.map = map

        self.num_timestep = 0
        self.infeasible_count = 0

        # check for failures after simulation is done
        self.max_time_reached = False
        self.execution_collision = False
        self.infeasible = False

        # metrics for logging
        self.algorithm_name = ""
        self.trial_num = 0
        self.avg_comp_time = []
        self.max_comp_time = 0.0
        self.traj_length = 0.0
        self.makespan = 0.0
        self.avg_rob_dist = 0.0
        self.c_avg = []
        self.success = False

        self.logger = MetricsLogger()
    
    def shift_movement(self, x0, u, f):
        f_value = f(x0, u[0])
        st = x0 + self.dt*f_value

        return st

    def prediction_state(self, x0, u, dt, N):
        # define prediction horizon function
        states = np.zeros((N+1, 3))
        states[0, :] = x0
        for i in range(N):
            states[i+1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * dt
            states[i+1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * dt
            states[i+1, 2] = states[i, 2] + u[i, 1] * dt
        return states

    # create model
    def f(self, x_, u_): return ca.vertcat(
        *[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

    def f_np(self, x_, u_): return np.array(
        [u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])
    
    def find_collisions(self):
        collision_map = {i: [] for i in range(self.num_agent)}
        for i in range(self.num_agent):
            for j in range(i+1, self.num_agent):
                if i == j:
                    continue
                agent_1_traj = self.prediction_cache[i]
                agent_2_traj = self.prediction_cache[j]

                for index, (wp_1, wp_2) in enumerate(zip(agent_1_traj, agent_2_traj)):
                    distance = math.sqrt((wp_1[0] - wp_2[0])**2 + (wp_1[1] - wp_2[1])**2)
                    if distance < self.rob_dia:
                        print("Collision detected between " + str(i) + " and " + str(j) + " at index " + str(index))
                    
                        if (j, index) not in collision_map[i]:
                            collision_map[i].append((j, index))
        
                        if (i, index) not in collision_map[j]:
                            collision_map[j].append((i, index))
                        break
                        
        return collision_map
    
    def are_all_agents_arrived(self):
        for i in range(self.num_agent):
            current_state = np.array(self.current_state[i])
            final_state = np.array(self.final_state[i])
            print(np.linalg.norm(current_state-final_state))
            if(np.linalg.norm(current_state-final_state) > self.goal_tolerence):
                return False
        return True
    
    def check_for_collisions(self, state_cache):
        for i in range(self.num_agent):
            for j in range(i + 1, self.num_agent):
                traj_1 = state_cache[i]
                traj_2 = state_cache[j]
                min_length = min(len(traj_1), len(traj_2))
                
                for k in range(min_length):
                    wp_1 = traj_1[k]
                    wp_2 = traj_2[k]
                    distance = math.sqrt((wp_1[0] - wp_2[0])**2 + (wp_1[1] - wp_2[1])**2)
                    
                    if distance < self.rob_dia:
                        print("COLLISION")
                        return True
        return False

    def is_solution_valid(self, state_cache):
        if self.check_for_collisions(state_cache):
            print("Executed trajectory has collisions")
            self.execution_collision = True
            return False
        elif self.num_timestep == self.total_sim_timestep:
            print("Maximum time is reached")
            self.max_time_reached = True
            return False
        else: 
            return True
        
    def find_closest_waypoint(self, waypoints, state):
        closest_distance = math.inf
        closest_index = None

        for i, waypoint in enumerate(waypoints):
            x_diff = waypoint['x'] - state[0]
            y_diff = waypoint['y'] - state[1]
            distance = math.sqrt(x_diff**2 + y_diff**2)

            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        return closest_index

    def extract_trajectory_segment(self, current_state):
        segment_dict = {}

        for robot_id, waypoints in self.ref.items():
            closest_index = self.find_closest_waypoint(waypoints, current_state)

            if closest_index is not None:
                segment_waypoints = []
                waypoints_len = len(waypoints)

                for i in range(closest_index, waypoints_len):
                    waypoint = waypoints[i]
                    segment_waypoints.append(waypoint)

                    if len(segment_waypoints) >= self.N:
                        break

                last_waypoint = segment_waypoints[-1]
                while len(segment_waypoints) < self.N:
                    segment_waypoints.append(last_waypoint)

                segment_dict[robot_id] = segment_waypoints

        return segment_dict

    def setup_visualization(self):
        self.fig = plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

        # Main plot for GMM Means and Agent State Visualization
        self.ax = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
        self.ax.set_title('GMM Means and Agent State Visualization')
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        # Subplot for Mode Probabilities
        self.ax_prob = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
        self.ax_prob.set_xlabel('Modes')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.set_ylim(0, 1)  # Assuming probabilities are between 0 and 1

        plt.ion()  # Turn on interactive mode

    def plot_gmm_means_and_state(self, current_state, current_prediction, gmm_data=None, mode_prob=None):
        self.ax.clear()  # Clear the main axes
        self.ax_prob.clear()  # Clear the mode probability axes

        # Set the title and labels for the main plot
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)

        # Plotting the GMM predictions as scattered points
        colors = plt.cm.get_cmap('hsv', len(gmm_data)+1)
        for mode, data in enumerate(gmm_data.values(), start=0):
            means = np.array(data['means'])
            cov = np.array(data['covariances'])
            for mean, cov_matrix in zip(means, cov):
                # Assume the covariance matrix is 2x2 and compute the radius for the circle
                # Here, we're taking the average of the variances for simplicity
                radius = np.sqrt((cov_matrix[0, 0] + cov_matrix[1, 1]) / 2)
                circle = plt.Circle((mean[0], mean[1]), radius, color=colors(mode), alpha=0.2)
                self.ax.add_patch(circle)

        # Plotting predictions
        if(isinstance(current_prediction, list)):
            for pred in current_prediction:
                self.ax.plot(pred[:, 0], pred[:, 1])
        else:
            self.ax.plot(current_prediction[0, :], current_prediction[1, :])

        # Plotting current state with a circle and arrow
        circle = plt.Circle((current_state[0], current_state[1]), 0.15, fill=True, color='blue')
        self.ax.add_patch(circle)
        arrow_length = 0.3
        arrow = plt.Arrow(current_state[0], current_state[1],
                          arrow_length * np.cos(current_state[2]), arrow_length * np.sin(current_state[2]),
                          width=0.1, color='yellow')
        self.ax.add_patch(arrow)

        # Plotting the mode probabilities as a bar chart
        if mode_prob is not None:
            modes = range(len(mode_prob))
            self.ax_prob.bar(modes, mode_prob, color='green', alpha=0.6)
            self.ax_prob.set_ylim(0, 1)  # Assuming probabilities are between 0 and 1
            self.ax_prob.set_ylabel('Mode Probabilities')

        plt.draw()
        plt.pause(0.1)

    def simulate(self):
        raise NotImplementedError("Subclasses must implement the functionality")