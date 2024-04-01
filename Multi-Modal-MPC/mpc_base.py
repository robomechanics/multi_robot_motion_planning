import casadi as ca
import numpy as np
from model import DiffDrive
import math
import matplotlib.pyplot as plt
from metrics_logger import MetricsLogger
from matplotlib.patches import Circle, Arrow
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.stats import multivariate_normal
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

class MPC_Base:
    def __init__(self, initial_state, final_state, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=None, ref=None, feedback=None, robust_horizon=None, mle=False):
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
        self.num_modes = 2
        self.robust_horizon = robust_horizon
        self.feedback = feedback
        self.mle = mle

        self.model = DiffDrive(self.rob_dia)

        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: [] for agent_id in range(self.num_agent)}

        # variables holding previous solutions
        self.prev_states = {agent_id: np.zeros((self.N+1, 3)) for agent_id in range(self.num_agent)}
        self.prev_controls = {agent_id: np.zeros((self.N, 2)) for agent_id in range(self.num_agent)}
        self.prev_epsilon_o = {agent_id: np.zeros((self.N+1, 1)) for agent_id in range(self.num_agent)}
        self.prev_epsilon_r = {agent_id: np.zeros((self.N+1, 1)) for agent_id in range(self.num_agent)}
        
        self.feedback_gains = {mode_id: np.zeros((2*self.N, 2*self.N)) for mode_id in range(self.num_modes)}
        self.feedback_gains_cache = {mode_id: [] for mode_id in range(self.num_modes)}
        self.heatmaps = []
        self.current_state = {}
        for i in range(self.num_agent):
            self.current_state[i] = self.initial_state[i]

        self.n_obs=len(self.uncontrolled_fleet_data)
        self.main_frame_counter = 0
        self.fb_frame_counter = 0
        self.frame_limit = 20

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
        self.feedback_gain_avg = 0.0
        self.success = False

        self.logger = MetricsLogger()

        self.rob_affine_model = {}
        self.obs_affine_model = {}
    
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
        
    def find_closest_waypoint(self, state):
        closest_distance = math.inf
        closest_index = None
        for i, waypoint in enumerate(self.ref):
            x_diff = waypoint[0] - state[0]
            y_diff = waypoint[1] - state[1]
            distance = math.sqrt(x_diff**2 + y_diff**2)

            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        return closest_index

    def extract_trajectory_segment(self, current_state):
        segment_dict = []

        closest_index = self.find_closest_waypoint(current_state)

        if closest_index is not None:
            segment_waypoints = []
            waypoints_len = len(self.ref)

            for i in range(closest_index, waypoints_len):
                waypoint = self.ref[i]
                segment_waypoints.append(waypoint)

                if len(segment_waypoints) >= self.N:
                    break

            last_waypoint = segment_waypoints[-1]
            while len(segment_waypoints) < self.N:
                segment_waypoints.append(last_waypoint)

            segment_dict.append(segment_waypoints)

        return segment_dict

    def setup_visualization(self):
        self.fig1 = plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

        # Main plot for GMM Means and Agent State Visualization
        self.ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
        self.ax1.set_title('GMM Means and Agent State Visualization')
        self.ax1.set_xlabel('X coordinate')
        self.ax1.set_ylabel('Y coordinate')
        self.ax1.set_xlim(-10, 10)
        self.ax1.set_ylim(-10, 10)

        # Subplot for Mode Probabilities
        self.ax_prob = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
        self.ax_prob.set_xlabel('Modes')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.set_ylim(0, 1)  # Assuming probabilities are between 0 and 1

        plt.ion()  # Turn on interactive mode

    def plot_gmm_means_and_state(self, current_state, current_prediction, gmm_data=None, mode_prob=None, ref=None):
        self.ax1.clear()  # Clear the main axes
        self.ax_prob.clear()  # Clear the mode probability axes
        self.ax1.axvline(x=-0.5, color='k', linestyle='--')
        self.ax1.axvline(x=0.5, color='k', linestyle='--')

        # Set the title and labels for the main plot
        self.ax1.set_xlim(-4, 4)
        self.ax1.set_ylim(-4, 4)

        # Plotting the GMM predictions as scattered points
        colors = plt.cm.get_cmap('hsv', self.n_obs*self.num_modes+1)
        for agent_pred in gmm_data:
            for mode, data in enumerate(agent_pred.values(), start=0):
                means = np.array(data['means'])
                cov = np.array(data['covariances'])
                for mean, cov_matrix in zip(means, cov):
                    # Assume the covariance matrix is 2x2 and compute the radius for the circle
                    # Here, we're taking the average of the variances for simplicity
                    radius = np.sqrt((cov_matrix[0, 0] + cov_matrix[1, 1]) / 2)
                    circle = plt.Circle((mean[0], mean[1]), radius, color=colors(mode), alpha=0.2)
                    # self.ax.add_patch(circle)

        # Plotting predictions
        if(isinstance(current_prediction, list)):
            trajectories = self.get_robot_feedback_policy()
            
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            for mode in trajectories.keys():
                # Extract obstacle and robot samples
                obs_samples = trajectories[mode]['obs']
                rob_samples = trajectories[mode]['rob']

                # Select color for current mode
                # color = colors[mode % len(colors)]

                # Plot the trajectories
                # For each sample in the mode, extract the positions and plot them
                for obs, rob in zip(obs_samples, rob_samples):
                    # Reshape the sample arrays to have pairs of (x, y) positions
                    obs_xy = np.ravel(obs).reshape(-1, 2)  # Reshape to (-1, 2) where -1 infers the correct length
                    rob_xy = np.ravel(rob).reshape(-1, 2)

                    # Plot obstacles trajectory for this sample
                    self.ax1.scatter(obs_xy[:, 0], obs_xy[:, 1], color=colors[mode], alpha=1, linewidth=2, label=f'Obstacles Mode {mode}' if obs is obs_samples[0] else "")

                    # Plot robot trajectory for this sample
                    self.ax1.plot(rob_xy[:, 0], rob_xy[:, 1], color=colors[mode], alpha=0.2, linewidth=1, label=f'Robot Mode {mode}' if rob is rob_samples[0] else "") 
        else:
            self.ax1.plot(current_prediction[0, :], current_prediction[1, :])

        # Plotting current state with a circle and arrow
        circle = plt.Circle((current_state[0], current_state[1]), 0.15, fill=True, color='blue')
        self.ax1.add_patch(circle)
        arrow_length = 0.3
        arrow = plt.Arrow(current_state[0], current_state[1],
                          arrow_length * np.cos(current_state[2]), arrow_length * np.sin(current_state[2]),
                          width=0.1, color='yellow')
        self.ax1.add_patch(arrow)

        # Plotting the mode probabilities as a bar chart
        # if mode_prob is not None:
        #     modes = range(len(mode_prob))
        #     mode_labels = [f"Mode {i+1}" for i in modes]  # Create labels for each mode
        #     bar_colors = [colors(i) for i in modes]  # Use the same color scheme as for the circles
        #     bars = self.ax_prob.bar(modes, mode_prob, color=bar_colors, alpha=0.6)
        #     self.ax_prob.set_ylim(0, 1)
        #     self.ax_prob.set_ylabel('Mode Probabilities')
        #     self.ax_prob.set_xticks(modes)  # Set the x-ticks to be at the modes
        #     self.ax_prob.set_xticklabels(mode_labels)  # Label the x-ticks

        plt.draw()

        if self.main_frame_counter < self.frame_limit:
            frame_filename = f'frame_{self.main_frame_counter}.png'  # Define the file name
            self.fig1.savefig(frame_filename)  # Save the figure
            self.main_frame_counter += 1  # Increment the frame counter
        plt.pause(0.1)

    def setup_visualization_heatmap(self):
        num_plots = len(self.feedback_gains)

        # Calculate figure width dynamically based on the number of plots
        # Assuming each plot requires a width of 4 units and height remains 6 units
        # Adding some extra space for padding between plots
        fig_width = 4 * num_plots if num_plots > 1 else 6  # Adjust the multiplier as needed
        fig_height = 6

        # Create a figure with subplots
        self.fig, self.axes = plt.subplots(1, num_plots, figsize=(fig_width, fig_height))

        # If there's more than one plot, adjust subplots to provide enough space between them
        if num_plots > 1:
            plt.subplots_adjust(wspace=0.3)  # Adjust the space between plots as needed

        plt.ion()  # Turn on interactive mode

    def plot_feedback_gains(self):
        # Define colors for modes, and use white for zero values
        color_map = {0: '#ff3333', 1: '#33ff33'}
        zero_color = 'white'  # Color for zero value

        for i, (mode, gain_matrix) in enumerate(self.feedback_gains.items()):
            ax = self.axes[i]
            ax.clear()  # Clear the axes for the new heatmap

            diagonal_elements = []

            N = gain_matrix.shape[0] // 2
            for j in range(N):
                block = gain_matrix[2*j:2*j+2, 2*j:2*j+2]
                diagonal_elements.extend([block[0, 0], block[1, 1]])

            diagonal_matrix = np.array(diagonal_elements).reshape(-1, 2)

            # Use the maximum absolute value for scaling, ensuring non-zero.
            max_val = np.max(np.abs(diagonal_matrix))
            if max_val == 0:
                max_val = 1  # Avoid division by zero.

            # Iterate through each cell in the diagonal_matrix to set colors and alpha.
            for y in range(diagonal_matrix.shape[0]):
                for x in range(diagonal_matrix.shape[1]):
                    value = diagonal_matrix[y, x]
                    # Treat values with |value| < 1e-3 as zero.
                    if abs(value) < 1e-3:
                        color = zero_color
                        alpha = 0  # Treat as fully transparent or white.
                    else:
                        color = color_map[mode]
                        # Scale alpha based on the absolute value relative to the max value.
                        alpha = abs(value) / max_val

                    rect = Rectangle((x, y), 1, 1, color=color, alpha=alpha)
                    ax.add_patch(rect)

            # Configure the axis settings for heatmap display.
            ax.set_xlim(0, diagonal_matrix.shape[1])
            ax.set_ylim(0, diagonal_matrix.shape[0])
            ax.set_aspect('equal')

            # Label axes.
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(['x', 'y'])
            ax.set_yticks(np.arange(diagonal_matrix.shape[0]) + 0.5)
            ax.set_yticklabels(np.arange(1, diagonal_matrix.shape[0] + 1))

        self.fig.tight_layout()
        plt.draw()

        if self.fb_frame_counter < self.frame_limit:
            frame_filename = f'frame_feedback_{self.fb_frame_counter}.png'  # Define the file name
            self.fig.savefig(frame_filename)  # Save the figure
            self.fb_frame_counter += 1  # Increment the frame counter
        plt.pause(0.01)

    def get_robot_feedback_policy(self):
        N_samples = 20
        N_t = multivariate_normal.rvs(np.zeros(self.N), np.eye(self.N), N_samples)
        current_state_obs = self.current_uncontrolled_state[0]

        # Initialize the data structure
        trajectories = {}

        for mode in range(self.num_modes):
            # Initialize the lists to store samples for current mode
            obs_samples = []
            rob_samples = []

            obs_mean = (self.obs_affine_model[0][mode]['T'] @ current_state_obs[:2]).reshape((-1, 1)) + self.obs_affine_model[0][mode]['c']
            rob_mean = ca.vec(self.prev_states[0][mode].T)

            for sample in range(N_samples):
                obs_dist = obs_mean + self.obs_affine_model[0][mode]['E'] @ N_t[sample, :].reshape((-1, 1))
                rob_dist = rob_mean + self.rob_affine_model['B'].toarray() @ self.feedback_gains[mode] @ (obs_dist[:-2] - obs_mean[:-2])

                rob_dist = rob_dist.reshape((3,13))
                rob_dist_xy = rob_dist[:2, :].reshape(((self.N+1)*2, 1))

                # Store the samples
                obs_samples.append(obs_dist)
                rob_samples.append(rob_dist_xy)

            # Store the samples for this mode
            trajectories[mode] = {'obs': obs_samples, 'rob': rob_samples}

        # plt.figure(figsize=(12, 9))

        # # Define some colors to differentiate between modes
        # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

        # for mode in trajectories.keys():
        #     # Extract obstacle and robot samples
        #     obs_samples = trajectories[mode]['obs']
        #     rob_samples = trajectories[mode]['rob']

        #     # Select color for current mode
        #     color = colors[mode % len(colors)]

        #     # Plot the trajectories
        #     # For each sample in the mode, extract the positions and plot them
        #     for obs, rob in zip(obs_samples, rob_samples):
        #         # Reshape the sample arrays to have pairs of (x, y) positions
        #         obs_xy = np.ravel(obs).reshape(-1, 2)  # Reshape to (-1, 2) where -1 infers the correct length
        #         rob_xy = np.ravel(rob).reshape(-1, 2)

        #         # Plot obstacles trajectory for this sample
        #         plt.plot(obs_xy[:, 0], obs_xy[:, 1], color=color, alpha=0.5, label=f'Obstacles Mode {mode}' if obs is obs_samples[0] else "")

        #         # Plot robot trajectory for this sample
        #         plt.scatter(rob_xy[:, 0], rob_xy[:, 1], color=color, alpha=0.75, label=f'Robot Mode {mode}' if rob is rob_samples[0] else "")

        # # Add plot legend, labels, and title
        # plt.legend()
        # plt.title('Trajectories of Obstacles and Robot Across All Modes')
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        # plt.show()
    
        return trajectories

    def simulate(self):
        raise NotImplementedError("Subclasses must implement the functionality")