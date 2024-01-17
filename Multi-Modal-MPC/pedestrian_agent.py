import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

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
            for mode, mode_pred in self.gmm_predictions[frame].items():
                means = mode_pred['means']
                covariances = mode_pred['covariances']

                # Plot circles for each mean with corresponding covariance as radius
                for mean, covariance in zip(means, covariances):
                    circle = patches.Circle((mean, self.y_value), radius=np.sqrt(covariance), fill=True, alpha=0.1)
                    ax.add_patch(circle)
                    self.circles.append(circle)

            # Update pedestrian position
            point.set_data(self.x_positions[frame], self.y_value)

            return [point] + self.circles

        ani = FuncAnimation(fig, update, frames=len(self.x_positions),
                            init_func=init, blit=True, interval=self.interval)
        plt.show()

class PedestrianSimulator:
    def __init__(self, initial_position, initial_velocity):
        self.position = initial_position
        self.velocity = initial_velocity
        self.dt = 0.1
        self.road_width = 2
        self.rationality = 0.5
        self.state_cache = []
        self.actions = [self.velocity, -self.velocity]
        self.init_state_variance = 0.01
        self.T = 10
        self.N = 10

    def step(self):
        self.position += self.velocity * self.dt
        self.state_cache.append(self.position) 

        return self.position
    
    def step_from_current(self, current_state, action):
        next_state = current_state + action * self.dt

        return next_state
    
    def get_gmm_from_current(self, current_state):
         # Single dictionary for the single agent
        agent_prediction = {}

        # Calculate the mean and covariance for each action at each timestep within the prediction horizon
        for mode, action in enumerate(self.actions):
            # Initialize the covariance matrix for the initial state
            covariance = self.init_state_variance

            # Mean and covariance vectors for the entire prediction horizon
            means = []
            covariances = []

            # Process noise matrix, assuming it's constant over time
            Q = self.init_state_variance

            temp_state = current_state

            # Populate the means and covariances for each timestep within the prediction horizon
            for step in range(self.N):
                # Propagate the state without additional noise
                new_state = self.step_from_current(temp_state, action)
                means.append(new_state)  # Mean of the state after propagation

                # Predict the new covariance matrix
                covariance = covariance + Q * self.dt  # Simplified prediction step for covariance

                # Store the predicted covariance matrix
                covariances.append(covariance)

                temp_state = new_state
      
            # Assign the mean and covariance vectors to the corresponding mode
            agent_prediction[mode] = {
                'means': means,  # List of means over the prediction horizon
                'covariances': covariances  # List of covariance matrices over the prediction horizon
            }

        # The predictions for all modes of the single agent are encapsulated in a list
        gmm_predictions = [agent_prediction]

        return gmm_predictions

# Create an instance of the simulator
sim = PedestrianSimulator(initial_position=0, initial_velocity=0.1)

# Simulation parameters
T = 10  # total simulation time
dt = 0.1

# Run the simulation
positions = []
gmm_predictions = []
for time in range(int(T / dt)):
    position = sim.step()
    gmm_pred = sim.get_gmm_from_current(position)
    gmm_predictions.append(gmm_pred[0])
    positions.append(position)

vis = Visualizer(x_positions=positions, y_value=2, gmm_predictions=gmm_predictions)
vis.animate()