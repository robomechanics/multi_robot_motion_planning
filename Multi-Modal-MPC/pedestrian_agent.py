import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time
from itertools import cycle

class Visualizer:
    def __init__(self, x_positions, y_value, gmm_predictions, road_width=1, interval=100):
        self.x_positions = x_positions
        self.y_value = y_value
        self.gmm_predictions = gmm_predictions
        self.road_width = road_width
        self.interval = interval
        self.circles = []

    def animate(self):
        fig, ax = plt.subplots()
        half_width = self.road_width / 2
        ax.set_xlim(-half_width-1, half_width+1)
        ax.set_ylim(self.y_value - 1, self.y_value + 1)

        # Drawing road boundaries
        ax.axvline(x=-half_width, color='black', linestyle='--')
        ax.axvline(x=half_width, color='black', linestyle='--')

        # Initialize pedestrian point
        point, = ax.plot([], [], 'ro')

        def init():
            point.set_data([], [])

            for circle in self.circles:
                circle.remove()
            self.circles.clear()

            return point,

        def update(frame):
            # Clear previous circles
            for circle in self.circles:
                circle.remove()
            self.circles.clear()

            # Iterate through each mode in the prediction for the current frame
            colors = ['green', 'blue']
            color_cycle = cycle(colors) 
            for mode, mode_pred in self.gmm_predictions[frame].items():
                circle_color = next(color_cycle)  # Get the next color from the cycle
                means = mode_pred['means']
                covariances = mode_pred['covariances']

                # Plot circles for each mean with corresponding covariance as radius
                for mean, covariance in zip(means, covariances):
                    circle = patches.Circle(mean, radius=0.05, fill=True, alpha=0.05, color=circle_color)
                    ax.add_patch(circle)
                    self.circles.append(circle)

            # Update pedestrian position
            point.set_data(self.x_positions[frame], self.y_value)

            return [point] + self.circles

        ani = FuncAnimation(fig, update, frames=len(self.x_positions),
                            init_func=init, blit=True, interval=self.interval)
        plt.show()

class PedestrianSimulator:
    def __init__(self, initial_position, initial_velocity, rationality, sim_time, dt, N, y_pos, vel_variance):
        self.position = initial_position
        self.velocity = initial_velocity
        self.dt = 0.1
        self.road_width = 2
        self.rationality = rationality
        self.state_cache = []
        self.predictions = []
        self.actions = [self.velocity, -self.velocity]
        self.vel_variance = vel_variance
        self.T = sim_time
        self.N = 20
        self.rationality = rationality
        self.action_probabilities = [0.5, 0.5]
        self.switch_time = (1 - rationality) * self.T
        self.initial_action = np.random.choice(self.actions, p=self.action_probabilities)
        self.start_time = time.time()
        self.dt = dt
        self.N = N
        self.y_pos = y_pos

    def step(self):
        elapsed_time = (time.time() - self.start_time) * 10000
        if(elapsed_time < self.switch_time):
            uncertain_action = np.random.normal(self.initial_action, np.sqrt(self.vel_variance), 1)[0]
            self.position += uncertain_action * self.dt
        else:
            uncertain_action = np.random.normal(-self.initial_action, np.sqrt(self.vel_variance), 1)[0]
            self.position += uncertain_action * self.dt
        
        self.state_cache.append((self.position,self.y_pos)) 

        return (self.position,self.y_pos)
    
    def step_from_current(self, current_state, action):
        next_x_pos = current_state[0] + action * self.dt
        next_state = (next_x_pos, current_state[1])

        return next_state
    
    def get_gmm_predictions(self):
        # Single dictionary for the single agent
        agent_prediction = {}

        # Calculate the mean and covariance for each action at each timestep within the prediction horizon
        for mode, action in enumerate(self.actions):
            # Mean and covariance vectors for the entire prediction horizon
            means = []
            covariances = []

            # Populate the means and covariances for each timestep within the prediction horizon
            for _ in np.arange(0, self.T, self.dt):
                means.append(action)  # The mean of v and omega is the action's value
                covariance = np.diag([self.vel_variance**2, self.vel_variance**2])  # Diagonal covariance matrix
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
        for mode, action in enumerate(self.actions):
            # Initialize the covariance matrix for the initial state
            covariance = self.vel_variance

            # Mean and covariance vectors for the entire prediction horizon
            means = []
            covariances = []

            # Process noise matrix, assuming it's constant over time
            Q = np.diag([self.vel_variance**2, 0.0])

            temp_state = current_state

            # Populate the means and covariances for each timestep within the prediction horizon
            for step in range(self.N):
                # Propagate the state without additional noise
                uncertain_action = np.random.normal(action, np.sqrt(self.vel_variance), 1)[0]
                new_state = self.step_from_current(temp_state, uncertain_action)
                means.append(new_state)  # Mean of the state after propagation

                # Predict the new covariance matrix
                covariance = covariance + Q * self.dt  # Simplified prediction step for covariance

                # Store the predicted covariance matrix
                covariances.append(np.array(covariance))

                temp_state = new_state
      
            # Assign the mean and covariance vectors to the corresponding mode
            agent_prediction[mode] = {
                'means': means,  # List of means over the prediction horizon
                'covariances': covariances  # List of covariance matrices over the prediction horizon
            }

        # The predictions for all modes of the single agent are encapsulated in a list
        gmm_predictions = [agent_prediction]

        return gmm_predictions
    
    def simulate_pedestrian(self):
        for _ in range(int(self.T / self.dt)):
            position = self.step()
            pred = self.get_gmm_predictions_from_current(position)
            self.predictions.append(pred[0])

        return self.predictions, self.state_cache

# Simulation parameters
T = 6 
dt = 0.1
rationality = 0.8
N = 20
y_pos = 3
vel_variance = 0.01

# Create an instance of the simulator
uncontrolled_agent = PedestrianSimulator(initial_position=0, initial_velocity=0.1, rationality=rationality, sim_time=T, dt=dt, N=N, y_pos=y_pos, vel_variance=vel_variance)
predictions, state_cache = uncontrolled_agent.simulate_pedestrian()

vis = Visualizer(x_positions=state_cache, y_value=y_pos, gmm_predictions=predictions)
vis.animate()