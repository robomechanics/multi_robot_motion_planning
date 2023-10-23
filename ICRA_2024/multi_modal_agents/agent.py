import numpy as np
from visualizer import Visualizer
import matplotlib.pyplot as plt

class UncontrolledAgent:
    def __init__(self, initial_state, dt=0.05, action_duration=20, v_noise_std=0.08, 
                 omega_noise_std=0.08, prediction_horizon=10):
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

        self.traj_cache = {key: [] for key in self.action_space}

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
        for action in self.action_space:
            trajectory = []
            simulated_state = self.state.copy()
            for _ in range(self.prediction_horizon):
                new_state = self.simulate_action(simulated_state, action)
                trajectory.append(new_state)
                simulated_state = new_state.copy()
            self.traj_cache[action].append(trajectory)
        return self.traj_cache

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

if __name__ == "__main__":
    # Number of time steps to simulate
    num_steps = 1

    # Create an instance of the UncontrolledAgent class with initial conditions
    initial_state = [0.0, 0.0, 0.0]  # [x, y, theta]
    agent = UncontrolledAgent(initial_state)

    # Simulate the agent's motion for the specified number of steps
    trajectory = []
    for _ in range(num_steps):
        state = agent.step()
        trajectory.append((state[0], state[1]))
    
    print(agent.traj_cache)

    # Create an AgentVisualizer instance with the precomputed trajectory
    visualizer = Visualizer(trajectory, agent.traj_cache)

    # Show the animation
    plt.show()
