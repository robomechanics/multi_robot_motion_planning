import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, x_positions, y_value, road_width=2, interval=100):
        """
        Initializes the visualizer with x-positions, a constant y-value, and animation interval.
        :param x_positions: List of x-positions.
        :param y_value: Constant y-value for all points.
        :param interval: Time interval (in ms) between frames.
        """
        self.x_positions = x_positions
        self.y_value = y_value
        self.interval = interval
        self.road_width = road_width

    def animate(self):
        fig, ax = plt.subplots()
        half_width = self.road_width / 2
        ax.set_xlim(-half_width-1, half_width+1)
        ax.set_ylim(self.y_value - 1, self.y_value + 1)

        # Drawing road boundaries
        ax.axvline(x=-half_width, color='black', linestyle='--')
        ax.axvline(x=half_width, color='black', linestyle='--')

        point, = ax.plot([], [], 'ro')
        def init():
            point.set_data([], [])
            return point,

        def update(frame):
            point.set_data(self.x_positions[frame], self.y_value)
            return point,

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

    def step(self):
        self.position += self.velocity * self.dt
        self.state_cache.append(self.position) 

        return self.position

# Create an instance of the simulator
sim = PedestrianSimulator(initial_position=0, initial_velocity=0.3)

# Simulation parameters
T = 10  # total simulation time
dt = 0.1

# Run the simulation
positions = []
for time in range(int(T / dt)):
    position = sim.step()
    positions.append(position)

vis = Visualizer(x_positions=positions, y_value=2)
vis.animate()