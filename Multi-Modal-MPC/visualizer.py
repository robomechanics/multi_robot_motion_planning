import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np


class Visualizer:
    def __init__(self, fleet_data, actions):
        self.fleet_data = fleet_data
        self.actions = actions
        self.agent_ids = list(fleet_data.keys())

        # Initialize the main figure
        self.fig = plt.figure(figsize=(10, 5))

        # Create a grid for subplots
        gs = self.fig.add_gridspec(len(self.agent_ids), 2)

        # Calculate the max and min values for x and y from all trajectories
        all_x_values = []
        all_y_values = []
        for agent_data in fleet_data.values():
            for traj in agent_data['executed_traj']:
                all_x_values.append(traj[0])
                all_y_values.append(traj[1])

        x_min, x_max = min(all_x_values), max(all_x_values)
        y_min, y_max = min(all_y_values), max(all_y_values)

        # Add some margin to the limits
        x_margin = (x_max - x_min) * 0.5
        y_margin = (y_max - y_min) * 0.5
        x_limits = (x_min - x_margin, x_max + x_margin)
        y_limits = (y_min - y_margin, y_max + y_margin)

        # Initialize the trajectory plot on the left
        self.ax1 = self.fig.add_subplot(gs[:, 0])
        self.ax1.set_xlim(x_limits)
        self.ax1.set_ylim(y_limits)
        self.ax1.set_aspect('equal')
        self.ax1.set_title("Agents Trajectories")

        # Assign a unique color to each agent using a perceptually uniform color map
        colors = plt.cm.get_cmap('viridis', len(self.agent_ids))
        self.agent_colors = {agent_id: colors(i) for i, agent_id in enumerate(self.agent_ids)}

        # Initialize lines for predicted trajectories
        self.lines = {agent_id: [self.ax1.plot([], [], color=self.agent_colors[agent_id])[0] for _ in actions] for agent_id in self.agent_ids}

        # Initialize a single moving circle for each agent
        self.circles = {agent_id: self.ax1.add_patch(Circle((0, 0), 0.1, color=self.agent_colors[agent_id], alpha=0.5)) for agent_id in self.agent_ids}

        # Initialize mode probability plots with matching colors
        self.mode_axes = []
        self.mode_bars = {}
        for idx, agent_id in enumerate(self.agent_ids):
            ax = self.fig.add_subplot(gs[idx, 1])
            self.mode_axes.append(ax)
            mode_prob = self.fleet_data[agent_id]['mode_probabilities']
            bars = ax.bar(range(len(mode_prob[0])), mode_prob[0], color=self.agent_colors[agent_id])
            self.mode_bars[agent_id] = bars

            ax.set_ylim([0, 1])
            ax.set_title(f"Agent {agent_id} Mode Probabilities")
            ax.set_xlabel("Modes")
            ax.set_ylabel("Probability")

        plt.tight_layout()

    def animate(self, i):
        anim_elements = []
        for agent_id in self.agent_ids:
            # Update the circle's position
            current_state = self.fleet_data[agent_id]['executed_traj'][i]
            self.circles[agent_id].center = (current_state[0], current_state[1])
            anim_elements.append(self.circles[agent_id])

            # Update predicted trajectories
            for action_idx, action in enumerate(self.actions):
                predictions = self.fleet_data[agent_id]['predictions'][action_idx]
                if i < len(predictions):
                    prediction = predictions[i]
                    xs, ys = zip(*[(state[0], state[1]) for state in prediction])
                    self.lines[agent_id][action_idx].set_data(xs, ys)
                    anim_elements.append(self.lines[agent_id][action_idx])

            # Update mode probabilities with matching color
            if i < len(self.fleet_data[agent_id]['mode_probabilities']):
                for bar, prob in zip(self.mode_bars[agent_id].patches, self.fleet_data[agent_id]['mode_probabilities'][i]):
                    bar.set_height(prob)
                anim_elements.extend(self.mode_bars[agent_id].patches)

        return anim_elements

    def show(self):
        max_frames = max(len(traj['executed_traj']) for traj in self.fleet_data.values())
        ani = FuncAnimation(self.fig, self.animate, frames=max_frames, interval=200, blit=True)
        plt.show()