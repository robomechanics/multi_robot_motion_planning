import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection

class UncontrolledAgent:
    def __init__(self, initial_state, dt=0.05, action_duration=20, v_noise_std=0.08, 
                 omega_noise_std=0.08, prediction_horizon=20):
        """
        :param initial_state: [x, y, theta] - initial position and orientation
        :param dt: time step
        :param action_duration: number of timesteps to persist with the same action
        :param v_noise_std: standard deviation of Gaussian noise for linear velocity
        :param omega_noise_std: standard deviation of Gaussian noise for angular velocity
        :param prediction_horizon: number of timesteps to predict into the future
        """
        self.state = np.array(initial_state)
        self.dt = dt
        self.action_space = ['cruise', 'turn_left', 'turn_right']
        
        self.action_duration = action_duration
        self.timesteps_elapsed = 0
        self.current_action = self.sample_action()

        # Noise parameters
        self.v_noise_std = v_noise_std
        self.omega_noise_std = omega_noise_std

        # Prediction horizon
        self.prediction_horizon = prediction_horizon

    def reset(self, state):
        """Reset the state of the agent and the action-related attributes."""
        self.state = np.array(state)
        self.timesteps_elapsed = 0
        self.current_action = self.sample_action()

    def step(self):
        """Simulate one time step based on the current or new action."""
        if self.timesteps_elapsed == self.action_duration:
            self.current_action = self.sample_action()
            self.timesteps_elapsed = 0

        v = 0.5 # constant speed for simplicity

        if self.current_action == 'cruise':
            omega = 0
            v += np.random.normal(0, self.v_noise_std)
        elif self.current_action == 'turn_left':
            v = 0.25
            omega = 0.5 + np.random.normal(0, self.omega_noise_std)
        elif self.current_action == 'turn_right':
            v = 0.25
            omega = -0.5 + np.random.normal(0, self.omega_noise_std)
        else:
            raise ValueError("Unknown action")

        self.state[0] += v * np.cos(self.state[2]) * self.dt
        self.state[1] += v * np.sin(self.state[2]) * self.dt
        self.state[2] += omega * self.dt
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi

        self.timesteps_elapsed += 1

        # Cache agent predictions for all actions for the prediction horizon
        self.predicted_trajectories = self.predict_horizon()

        return self.state.copy()

    def predict_horizon(self):
        """Predicts the state trajectories for all actions over the prediction horizon."""
        trajectories = {}
        for action in self.action_space:
            trajectory = []
            simulated_state = self.state.copy()
            for _ in range(self.prediction_horizon):
                new_state = self.simulate_action(simulated_state, action)
                trajectory.append(new_state)
                simulated_state = new_state.copy()
            trajectories[action] = trajectory
        return trajectories

    def simulate_action(self, state, action):
        """Simulate the effect of a given action on the provided state without changing the agent's internal state."""
        v = 0.5

        if action == 'cruise':
            omega = 0
            v += np.random.normal(0, self.v_noise_std)
        elif action == 'turn_left':
            v = 0.25
            omega = 0.5 + np.random.normal(0, self.omega_noise_std)
        elif action == 'turn_right':
            v = 0.25
            omega = -0.5 + np.random.normal(0, self.omega_noise_std)
        else:
            raise ValueError("Unknown action")

        state[0] += v * np.cos(state[2]) * self.dt
        state[1] += v * np.sin(state[2]) * self.dt
        state[2] += omega * self.dt
        state[2] = (state[2] + np.pi) % (2 * np.pi) - np.pi
        return state

    def sample_action(self, prob_distribution=None):
        """Sample an action based on a given probability distribution or uniformly if none provided."""
        if prob_distribution is None:
            prob_distribution = [0.5, 0.25, 0.25]
        return np.random.choice(self.action_space, p=prob_distribution)
    
    def sample_trajectories(self, action, num_samples=10):
        all_trajectories = []
        for _ in range(num_samples):
            trajectory = []
            simulated_state = self.state.copy()
            for _ in range(self.prediction_horizon):
                new_state = self.simulate_action(simulated_state, action)
                trajectory.append(new_state)
                simulated_state = new_state.copy()
            all_trajectories.append(trajectory)
        return all_trajectories


# Parameters
num_timesteps = 100
initial_state = [0.0, 0.0, 0.0]

# Simulate
agent = UncontrolledAgent(initial_state)
trajectory = [initial_state]
# predicted_trajectories = []

# for _ in range(num_timesteps):
#     new_state = agent.step()
#     trajectory.append(new_state)
#     predicted_trajectories.append(agent.predicted_trajectories.copy())

# trajectory = np.array(trajectory)

predicted_trajectories_samples = {action: [] for action in agent.action_space}

for _ in range(num_timesteps):
    new_state = agent.step()
    trajectory.append(new_state)
    for action in agent.action_space:
        action_trajectories = agent.sample_trajectories(action)
        predicted_trajectories_samples[action].append(action_trajectories)

# Set up the figure, axis, and plot element
fig, ax = plt.subplots()
ax.set_xlim(min(trajectory[:, 0]) - 1, max(trajectory[:, 0]) + 1)
ax.set_ylim(min(trajectory[:, 1]) - 1, max(trajectory[:, 1]) + 1)
line, = ax.plot([], [], 'b-', lw=2)
point, = ax.plot([], [], 'ro', markersize=6)
predicted_lines = {action: ax.plot([], [], 'g', lw=1)[0] for action in agent.action_space}

def init():
    """Initialize the line, point, and predicted lines to empty data."""
    line.set_data([], [])
    point.set_data([], [])
    for action_line in predicted_lines.values():
        action_line.set_data([], [])
    return [line, point] + list(predicted_lines.values())

def update(frame):
    """Update the line, point, and predicted trajectories for each frame of the animation."""
    line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    point.set_data(trajectory[frame, 0], trajectory[frame, 1])
    for action in predicted_trajectories_samples:
        for pred_trajectory in predicted_trajectories_samples[action][frame]:
            pred_trajectory = np.array(pred_trajectory)
            ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'g', lw=1, alpha=0.2)
    return [line, point]

ani = FuncAnimation(fig, update, frames=len(trajectory),
                    init_func=init, blit=True, repeat=False)

plt.show()


