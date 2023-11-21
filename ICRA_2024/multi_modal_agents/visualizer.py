import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

class Visualizer:
    def __init__(self, trajectories, actions, switch_times, mode_probabilities):
        self.trajectories = trajectories
        self.actions = actions
        self.switch_times = switch_times
        self.mode_prob = mode_probabilities
        
        # Initialize the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.lines = [self.ax1.plot([], [])[0] for _ in actions]
        self.circles = [[self.ax1.add_patch(plt.Circle((0, 0), 0, color='gray', alpha=0.3)) for _ in traj] for traj in trajectories]

        self.ax1.set_xlim([-5, 5])
        self.ax1.set_ylim([-5, 5])
        self.ax1.set_aspect('equal')

        self.ax2.set_ylim([0, 1])

        self.bars = self.ax2.bar(range(len(self.mode_prob[0])), self.mode_prob[0])

        self.ax1.set_title("Agent Trajectory")
        self.ax2.set_title("Mode Probabilities")
        self.ax2.set_xlabel("Modes")
        self.ax2.set_ylabel("Probability")

    def animate(self, i):
        for j, traj in enumerate(self.trajectories):
            prediction = traj[i]
            xs, ys, noises_x, noises_y = zip(*prediction)
            self.lines[j].set_data(xs, ys)
            
            for k, (x, y, noise_x, noise_y) in enumerate(prediction):
                circle = self.circles[j][k]
                circle.center = (x, y)
                circle.radius = np.sqrt(noise_x**2 + noise_y**2)

            for bar, prob in zip(self.bars, self.mode_prob[i]):
                bar.set_height(prob)

        anim_elements = [item for sublist in self.circles for item in sublist] + self.lines
        anim_elements.extend(self.bars.patches)  # Add bar chart rectangles (patches) to the animation elements   
        
        return anim_elements

    def show(self):
        ani = FuncAnimation(self.fig, self.animate, frames=len(self.trajectories[0]), interval=200, blit=True)
        plt.show()