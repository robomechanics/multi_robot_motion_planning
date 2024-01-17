import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualizer import Visualizer

class UncontrolledAgent:
    def __init__(self, dt=0.2, T=40, H=10, min_action_duration=1, num_switches=2, action_variance=None):
        self.dt = dt
        self.T = T 
        self.H = H
        self.min_action_duration = min_action_duration
        self.init_state_variance = [0.01, 0.01, 0.01]
        self.num_switches = num_switches
        self.omega_variance = 0.2 if action_variance is None else action_variance   # Variances for omega corresponding to each action
        self.v_variance = 0.2 if action_variance is None else action_variance
        self.action_prob = [0.5, 0.5]
        self.actions = [
        (np.random.normal(0.3, 0.05), np.random.normal(0.05, 0.02)), 
        (np.random.normal(0.3, 0.05), np.random.normal(0.0, 0.01))]
        # (np.random.normal(0.3, 0.05), np.random.normal(-0.2, 0.02))]
        self.noise_range = [-0.02, 0.02]
        self.prior_likelihood = [0.5, 0.5]
        self.alpha = 0.2
        self.min_probability = 0.1

    def propagate_state(self, x, y, theta, v, omega):
        noise_x = np.random.uniform(self.noise_range[0], self.noise_range[1])
        noise_y = np.random.uniform(self.noise_range[0], self.noise_range[1])

        x += v * np.cos(theta) * self.dt + noise_x
        y += v * np.sin(theta) * self.dt + noise_y
        theta += omega * self.dt

        return x, y, theta, noise_x, noise_y

    def generate_switch_times(self):
        segment_length = self.T / self.num_switches
        switch_times = []
        for i in range(self.num_switches):
            segment_start = i * segment_length
            segment_end = segment_start + segment_length
            if self.min_action_duration > 0 and i > 0:
                segment_start = max(segment_start, switch_times[-1] + self.min_action_duration)
            switch_time = np.random.uniform(segment_start, segment_end)
            switch_times.append(switch_time)

        return np.sort(switch_times)

    def calculate_likelihood(self, observed_action):
        likelihood = np.full(len(self.actions), 0.4)  # Start with 10% likelihood for all modes

        for i, action in enumerate(self.actions):
            if action == observed_action:
                likelihood[i] = 0.6  # Set 80% likelihood for the observed action
                break

        return likelihood

    def simulate_diff_drive(self, x0=0, y0=0, theta0=0):
        predictions = [[] for _ in self.actions]
        executed_traj = []
        mode_probabilities = []

        self.switch_times = self.generate_switch_times()
        switch_index = 0

        x, y, theta = x0, y0, theta0
        executed_traj.append((x,y,theta))
        current_action_duration = 0
        selected_action = self.actions[np.random.choice(len(self.actions), p=self.action_prob)]
        state_prob = self.prior_likelihood

        for t in np.arange(0, self.T, self.dt):
            if switch_index < len(self.switch_times) and t >= self.switch_times[switch_index]:
                selected_action = self.actions[np.random.choice(len(self.actions), p=self.action_prob)]
                switch_index += 1
                state_prob = self.prior_likelihood
            
            likelihood = self.calculate_likelihood(selected_action)
            new_state_prob = (likelihood * state_prob) / np.sum(likelihood * state_prob)

            # Apply smoothing to each element
            state_prob = [self.alpha * new_state_prob[i] + (1 - self.alpha) * state_prob[i] for i in range(len(state_prob))]

            state_prob = [max(p, self.min_probability) for p in state_prob]

            # Renormalize the probabilities to ensure they sum to 1
            total_prob = sum(state_prob)
            state_prob = [p / total_prob for p in state_prob]

            mode_probabilities.append(state_prob)

            for i, (v, omega) in enumerate(self.actions):
                temp_x, temp_y, temp_theta = x, y, theta
                traj = []
                accumulated_noise_x, accumulated_noise_y = 0, 0
                for _ in np.arange(0, self.H, self.dt):
                    temp_x, temp_y, temp_theta, noise_x, noise_y = self.propagate_state(temp_x, temp_y, temp_theta, v, omega)
                    accumulated_noise_x += noise_x
                    accumulated_noise_y += noise_y
                    traj.append((temp_x, temp_y, accumulated_noise_x, accumulated_noise_y))
                predictions[i].append(traj)

            # Update robot's actual state using the selected action
            x, y, theta, noise_x, noise_y = self.propagate_state(x, y, theta, selected_action[0], selected_action[1])
            executed_traj.append((x,y,theta))
            current_action_duration += self.dt

        return predictions, executed_traj, mode_probabilities

    def get_gmm_predictions(self):
        # Single dictionary for the single agent
        agent_prediction = {}

        # Calculate the mean and covariance for each action at each timestep within the prediction horizon
        for mode, (v, omega) in enumerate(self.actions):
            # Mean and covariance vectors for the entire prediction horizon
            means = []
            covariances = []

            # Populate the means and covariances for each timestep within the prediction horizon
            for _ in np.arange(0, self.H, self.dt):
                means.append([v, omega])  # The mean of v and omega is the action's value
                covariance = np.diag([self.v_variance**2, self.omega_variance**2])  # Diagonal covariance matrix
                covariances.append(covariance)
    
            # Assign the mean and covariance vectors to the corresponding mode
            agent_prediction[mode] = {
                'means': means,  # List of means over the prediction horizon
                'covariances': covariances  # List of covariance matrices over the prediction horizon
            }

        # The predictions for all modes of the single agent are encapsulated in a list
        gmm_predictions = [agent_prediction]

        return gmm_predictions

    def get_gmm_predictions_from_current(self, current_state):
        # Single dictionary for the single agent
        agent_prediction = {}

        # Calculate the mean and covariance for each action at each timestep within the prediction horizon
        for mode, (v, omega) in enumerate(self.actions):
            # Initial state
            x ,y, theta = current_state[0], current_state[1], current_state[2]

            # Initialize the covariance matrix for the initial state
            covariance = np.diag([self.init_state_variance[0], self.init_state_variance[1], self.init_state_variance[2]])

            # Mean and covariance vectors for the entire prediction horizon
            means = []
            covariances = []

            # Process noise matrix, assuming it's constant over time
            Q = np.diag([self.v_variance**2, self.v_variance**2, self.omega_variance**2])

            # Populate the means and covariances for each timestep within the prediction horizon
            for step in np.arange(0, self.H, self.dt):
                # Propagate the state without additional noise
                x, y, theta, noise_x, noise_y = self.propagate_state(x, y, theta, v, omega)
                means.append([x, y, theta])  # Mean of the state after propagation

                # Predict the new covariance matrix
                covariance = covariance + Q * self.dt  # Simplified prediction step for covariance

                # Store the predicted covariance matrix
                covariances.append(covariance)
      
            # Assign the mean and covariance vectors to the corresponding mode
            agent_prediction[mode] = {
                'means': means,  # List of means over the prediction horizon
                'covariances': covariances  # List of covariance matrices over the prediction horizon
            }

        # The predictions for all modes of the single agent are encapsulated in a list
        gmm_predictions = [agent_prediction]

        return gmm_predictions

# agent = UncontrolledAgent()
# traj, prediction, mode_probabilities = agent.simulate_diff_drive()

# predicitons = agent.get_gmm_predictions()

# vis = Visualizer(traj, agent.actions, agent.switch_times, mode_probabilities)
# vis.show()