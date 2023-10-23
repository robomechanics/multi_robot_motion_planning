import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class Visualizer:
    def __init__(self, trajectory, predicted_trajectories):
        self.trajectory = trajectory
        self.predicted_trajectories = predicted_trajectories
        
        # Initialize the figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1.0, 1.0)  # Set initial x-axis limits
        self.ax.set_ylim(-1.0, 1.0)  # Set initial y-axis limits

        # Create an empty plot for the agent's trajectory
        self.trajectory_plot, = self.ax.plot([], [], label='Uncontrolled Agent')

        # Create an empty circle for the agent's current position
        self.agent_circle = Circle((0, 0), 0.03, fill=True)
        self.ax.add_patch(self.agent_circle)

        # Set axis labels and legend
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.legend()

        # Create an animation
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.trajectory), repeat=False)

    def update(self, frame):
        # Update the trajectory plot
        self.trajectory_plot.set_data(*zip(*self.trajectory[:frame+1]))

        # Update the agent's current position
        x, y = self.trajectory[frame]
        self.agent_circle.center = (x, y)

        # Redraw the legend to include predicted trajectories
        self.ax.legend()
