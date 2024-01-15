import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class PedestrianSimulator:
    def __init__(self, initial_position, initial_velocity):
        self.position = initial_position
        self.velocity = initial_velocity
        self.dt = 0.1
        self.road_width = 2
        self.rationality = 0.5

    def step(self):
        self.position += self.velocity * self.dt 

        return self.position

    def animate(self, positions, dt):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'ro')

        def init():
            ax.set_xlim(-self.road_width/2-1, self.road_width/2+1)
            ax.set_ylim(0, 1)

            ax.axvline(x=-self.road_width/2, color='blue', linewidth=2)
            ax.axvline(x=self.road_width/2, color='blue', linewidth=2)  # Adjust linewidth for visibility

            return line, 

        def update(frame):
            line.set_data(positions[frame], 0.5)
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(positions),
                                      init_func=init, blit=True, interval=dt*1000)
        plt.show()

# Create an instance of the simulator
sim = PedestrianSimulator(initial_position=0, initial_velocity=0.3)

# Simulation parameters
total_time = 10  # total simulation time
dt = 0.1

# Run the simulation
positions = []
for _ in np.arange(0, dt, total_time):
    position = sim.step()
    positions.append(position)

# Animate the result
sim.animate(positions, total_time)
