# import numpy as np
# from agent import UncontrolledAgent

# class TrajectoryPredictor:
#     def __init__(self, agent):
#         """
#         Initialize the TrajectoryEstimator.

#         :param agent: The uncontrolled agent.
#         :param num_steps: Number of time steps to simulate.
#         :param initial_state: Initial state of the agent.
#         """
#         self.agent = agent
#         self.prediction_horizon = agent.prediction_horizon
#         self.current_state = agent.state
#         self.trajectories = {action: [] for action in agent.action_space}

#     def simulate_trajectories(self):
#         """
#         Simulate agent trajectories for each action and store them in the 'trajectories' dictionary.
#         """
#         for action in self.agent.action_space:
#             for _ in range(self.prediction_horizon):
#                 state = self.agent.step()
#                 self.trajectories[action].append(state)

#             # Convert the trajectory to a numpy array
#             self.trajectories[action] = np.array(self.trajectories[action])
    
# # Example usage:
# if __name__ == "__main__":
#     num_steps = 50
#     initial_state = [0.0, 0.0, 0.0]
#     agent = UncontrolledAgent(initial_state)
    
#     predictor = TrajectoryPredictor(agent)
#     predictor.simulate_trajectories()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def propagate_state(x, y, theta, v, omega, dt, noise_range=(-0.01, 0.01)):
    noise_x = np.random.uniform(noise_range[0], noise_range[1])
    noise_y = np.random.uniform(noise_range[0], noise_range[1])
    
    x += v * np.cos(theta) * dt + noise_x
    y += v * np.sin(theta) * dt + noise_y
    theta += omega * dt
    
    return x, y, theta, noise_x, noise_y

def simulate_diff_drive(actions, beliefs, dt=0.05, T=10, x0=0, y0=0, theta0=0, H=1.0, action_duration=3.0):
    trajectories = [[] for _ in actions]
    
    x, y, theta = x0, y0, theta0
    current_action_duration = 0
    selected_action = actions[np.random.choice(len(actions), p=beliefs)]
    
    for t in np.arange(0, T, dt):
        if current_action_duration >= action_duration:
            # Sample a new action based on belief
            selected_action = actions[np.random.choice(len(actions), p=beliefs)]
            current_action_duration = 0
        
        for i, (v, omega) in enumerate(actions):
            temp_x, temp_y, temp_theta = x, y, theta
            traj = []
            accumulated_noise_x, accumulated_noise_y = 0, 0
            for _ in np.arange(0, H, dt):
                temp_x, temp_y, temp_theta, noise_x, noise_y = propagate_state(temp_x, temp_y, temp_theta, v, omega, dt)
                accumulated_noise_x += noise_x
                accumulated_noise_y += noise_y
                traj.append((temp_x, temp_y, accumulated_noise_x, accumulated_noise_y))
            trajectories[i].append(traj)
        
        # Update robot's actual state using the selected action
        x, y, theta, _, _ = propagate_state(x, y, theta, selected_action[0], selected_action[1], dt)
        current_action_duration += dt
    
    return trajectories

actions = [
    (np.random.uniform(0.4, 0.6), np.random.uniform(0.7, 0.8)), 
    (np.random.uniform(0.4, 0.6), np.random.uniform(0.0, 0.0)), 
    (np.random.uniform(0.4, 0.6), np.random.uniform(-0.7, -0.8))
]

beliefs = [0.2, 0.6, 0.2]  # Sample beliefs for the actions

dt = 0.05
T = 10.0
H = 1.0  # Prediction horizon, which should be <= T
n_prediction_steps = int(H/dt)  # Compute the number of prediction steps based on the horizon and time-step

trajectories = simulate_diff_drive(actions, beliefs, T=5, action_duration=2.0)  # Action persists for 2 seconds

fig, ax = plt.subplots(figsize=(5, 5))
lines = [ax.plot([], [])[0] for _ in actions]
circles = [[ax.add_patch(plt.Circle((0, 0), 0, color='gray', alpha=0.3)) for _ in range(n_prediction_steps)] for _ in actions]

ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
ax.set_aspect('equal')
ax.grid(True)

def animate(i):
    for j, traj in enumerate(trajectories):
        prediction = traj[i]
        xs, ys, noises_x, noises_y = zip(*prediction)
        lines[j].set_data(xs, ys)
        
        for k, (x, y, noise_x, noise_y) in enumerate(prediction):
            circle = circles[j][k]
            circle.center = (x, y)
            circle.radius = np.sqrt(noise_x**2 + noise_y**2)
            
    return [item for sublist in circles for item in sublist] + lines

ani = FuncAnimation(fig, animate, frames=len(trajectories[0]), interval=200, blit=True)

plt.title("Animated Differential Drive with Future Prediction, Accumulating Noise, and Action Persistence")
plt.show()
