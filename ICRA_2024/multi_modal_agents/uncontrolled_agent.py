import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualizer import Visualizer
class UncontrolledAgent:
  def __init__(self, dt=0.05, T=10, H=1.0, action_duration=3.0):
    self.dt = dt
    self.T = T 
    self.H = H
    self.action_duration = action_duration
    self.action_prob = [0.2, 0.6, 0.2]
    self.actions = [
  (np.random.uniform(0.4, 0.6), np.random.uniform(0.7, 0.8)), 
  (np.random.uniform(0.4, 0.6), np.random.uniform(0.0, 0.0)), 
  (np.random.uniform(0.4, 0.6), np.random.uniform(-0.7, -0.8))]
    self.noise_range = [-0.01, 0.01]
        
  def propagate_state(self, x, y, theta, v, omega):
    noise_x = np.random.uniform(self.noise_range[0], self.noise_range[1])
    noise_y = np.random.uniform(self.noise_range[0], self.noise_range[1])
    
    x += v * np.cos(theta) * self.dt + noise_x
    y += v * np.sin(theta) * self.dt + noise_y
    theta += omega * self.dt
    
    return x, y, theta, noise_x, noise_y

  def simulate_diff_drive(self, x0=0, y0=0, theta0=0):
    trajectories = [[] for _ in self.actions]
    
    x, y, theta = x0, y0, theta0
    current_action_duration = 0
    selected_action = self.actions[np.random.choice(len(self.actions), p=self.action_prob)]
    
    for t in np.arange(0, self.T, self.dt):
        if current_action_duration >= self.action_duration:
            # Sample a new action based on belief
            selected_action = self.actions[np.random.choice(len(self.actions), p=self.action_prob)]
            current_action_duration = 0
        
        for i, (v, omega) in enumerate(self.actions):
            temp_x, temp_y, temp_theta = x, y, theta
            traj = []
            accumulated_noise_x, accumulated_noise_y = 0, 0
            for _ in np.arange(0, self.H, self.dt):
                temp_x, temp_y, temp_theta, noise_x, noise_y = self.propagate_state(temp_x, temp_y, temp_theta, v, omega)
                accumulated_noise_x += noise_x
                accumulated_noise_y += noise_y
                traj.append((temp_x, temp_y, accumulated_noise_x, accumulated_noise_y))
            trajectories[i].append(traj)
        
        # Update robot's actual state using the selected action
        x, y, theta, _, _ = self.propagate_state(x, y, theta, selected_action[0], selected_action[1])
        current_action_duration += self.dt
    
    return trajectories

dt = 0.05
T = 10.0
H = 1.0
n_prediction_steps = int(H/dt)  # Compute the number of prediction steps based on the horizon and time-step

predictor = UncontrolledAgent()
trajectories = predictor.simulate_diff_drive()

# Visualization setup
visualizer = Visualizer(trajectories, predictor.actions, n_prediction_steps)
visualizer.show()