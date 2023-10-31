import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

class Visualizer:
    def __init__(self, trajectories, actions):
        self.trajectories = trajectories
        self.actions = actions
        
        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.lines = [self.ax.plot([], [])[0] for _ in actions]
        self.circles = [[self.ax.add_patch(plt.Circle((0, 0), 0, color='gray', alpha=0.3)) for _ in traj] for traj in trajectories]

        self.ax.set_xlim([-1, 2])
        self.ax.set_ylim([-1, 2])
        self.ax.set_aspect('equal')
        self.ax.grid(True)

    def animate(self, i):
        for j, traj in enumerate(self.trajectories):
            prediction = traj[i]
            xs, ys, noises_x, noises_y = zip(*prediction)
            self.lines[j].set_data(xs, ys)
                
            for k, (x, y, noise_x, noise_y) in enumerate(prediction):
                circle = self.circles[j][k]
                circle.center = (x, y)
                circle.radius = np.sqrt(noise_x**2 + noise_y**2)
                
        return [item for sublist in self.circles for item in sublist] + self.lines

    def show(self):
        ani = FuncAnimation(self.fig, self.animate, frames=len(self.trajectories[0]), interval=200, blit=True)
        plt.show()