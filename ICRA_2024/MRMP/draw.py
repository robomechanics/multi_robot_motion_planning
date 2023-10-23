import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib as mpl
import random 

class Draw_MPC_point_stabilization_v1(object):
    def __init__(self, robot_states: dict, init_state: np.array, target_state: np.array, obs_state, rob_dia, map,
                 export_fig=True):
        self.num_agents = len(robot_states)
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state

        self.static_obs = [] if not obs_state["static"] else obs_state["static"]
        self.dynamic_obs = [] if not obs_state["dynamic"] else obs_state["dynamic"]

        self.map = map

        self.rob_radius = rob_dia / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, 4), ylim=(-1, 5))
        self.fig.set_size_inches(7, 6.5)
        # self.animation_init()
        skip_frames = 3
        num_frames = len(max(robot_states.values(), key=lambda x: len(x)))
        included_frames = range(0, num_frames, skip_frames)  # Generate indices of included frames
        
        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, included_frames,
                                           init_func=self.animation_init, interval=75, repeat=False)

        if export_fig:
            self.ani.save('./v1.gif', writer='imagemagick', fps=30)
        plt.show()

    def generate_colors(self):
        num_robots = min(self.num_agents, len(self.all_colors))
        return self.all_colors[:num_robots]

    def animation_init(self):
        # self.ax.imshow(self.map, cmap='gray_r', origin='lower')

        self.target_circles = []
        self.target_arrs = []
        self.robot_bodies = []
        self.robot_arrs = []

        self.all_colors = ['#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf',  # cyan
        '#1f1f1f',  # black
        '#ff9896',  # light red
        '#98df8a',  # light green
        '#c5b0d5',  # light purple
        '#c49c94',  # light brown
        '#f7b6d2', 
        '#aec7e8',  # light blue
        '#ffbb78']
        
        self.robot_colors = self.generate_colors()

        for i in range(self.num_agents):
            init_state = self.init_state[i]
            target_state = self.target_state[i]

            target_circle = plt.Circle(target_state[:2], self.rob_radius, color='b', fill=False)
            # self.ax.add_artist(target_circle)
            self.target_circles.append(target_circle)

            target_arr = mpatches.Arrow(target_state[0], target_state[1],
                                        self.rob_radius * np.cos(target_state[2]),
                                        self.rob_radius * np.sin(target_state[2]), width=0.1)
            # self.ax.add_patch(target_arr)
            self.target_arrs.append(target_arr)

            color = self.robot_colors[i]
            robot_body = plt.Circle(init_state[:2], self.rob_radius, color=color, fill=False)
            self.ax.add_artist(robot_body)
            self.robot_bodies.append(robot_body)

            robot_arr = mpatches.Arrow(init_state[0], init_state[1],
                                       self.rob_radius * np.cos(init_state[2]),
                                       self.rob_radius * np.sin(init_state[2]), width=0.001, color=color, alpha=0)
            self.ax.add_patch(robot_arr)
            self.robot_arrs.append(robot_arr)

            if self.static_obs:
                for obs in self.static_obs:
                    self.obs_static_body = plt.Circle(obs[0:2], obs[2]/2, color='r', fill=True)
                    self.obs_static_body.set_alpha(0.5)
                    self.obs_artist = self.ax.add_artist(self.obs_static_body)
            else:
                self.static_obs = []
        
        self.ax.axis('off')  
        self.ax.set_frame_on(False)  
        plt.show()
        
        return self.target_circles, self.target_arrs, self.robot_bodies, self.robot_arrs, self.static_obs

    def animation_loop(self, indx):
        # self.ax.cla()  # Clear the axes for each frame
        # self.ax.imshow(self.occupancy_map, cmap='gray_r', origin='lower')
        
        for i in range(self.num_agents):
            robot_states = self.robot_states[i]

            color = self.robot_colors[i]

            if indx < len(robot_states):
                position = robot_states[indx][:2]
                orientation = robot_states[indx][2]

                alpha = min(1.0, indx / len(robot_states))
                self.robot_bodies[i].center = position

                robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                            self.rob_radius * np.sin(orientation), width=0.1, color=color, alpha=alpha)

                # Remove the previous arrow and add the new one
                # if self.robot_arrs[i] is not None:
                #     self.robot_arrs[i].remove()
                self.robot_arrs[i] = robot_arr
                self.ax.add_patch(robot_arr)

        self.fig.canvas.draw()
        self.ax.axis('off')  
        self.ax.set_frame_on(False)  
        plt.show()

        return self.robot_arrs, self.robot_bodies

class Draw_MPC_Obstacle(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, obstacle: np.array,
                 rob_dia=0.3, export_fig=True):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_dia / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.8, 3), ylim=(-0.8, 3.))
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            print('no obstacle given, break')
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('obstacle.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.obstacle_circle = plt.Circle(self.obstacle[:2], self.obstacle[2], color='g', fill=True)
        self.ax.add_artist(self.obstacle_circle)
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr, self.obstacle_circle

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        # self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_MPC_tracking(object):
    def __init__(self, robot_states: list, init_state: np.array, rob_dia=0.3, export_fig=False):
        self.init_state = init_state
        self.robot_states = robot_states
        self.rob_radius = rob_dia
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1.0, 16), ylim=(-0.5, 1.5))
        # self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('tracking.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self, ):
        # draw target line
        self.target_line = plt.plot([0, 12], [1, 1], '-r')
        # draw the initial position of the robot
        self.init_robot_position = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.init_robot_position)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_line, self.init_robot_position, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_FolkLift(object):
    def __init__(self, robot_states: list, initial_state: np.array, export_fig=False):
        self.init_state = initial_state
        self.robot_state_list = robot_states
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1.0, 8.0), ylim=(-0.5, 8.0))

        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_state_list)),
                                                   init_func=self.animation_init, interval=100, repeat=False)
        if export_fig:
            pass
        plt.show()

    def animation_init(self, ):
        x_, y_, angle_ = self.init_state[:3]
        tr = mpl.transforms.Affine2D().rotate_deg_around(x_, y_, angle_)
        t = tr + self.ax.transData
        self.robot_arr = mpatches.Rectangle((x_ - 0.12, y_ - 0.08),
                                             0.24,
                                             0.16,
                                             transform=t,
                                             color='b',
                                             alpha=0.8,
                                             label='DIANA')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr

    def animation_loop(self, indx):
        x_, y_, angle_ = self.robot_state_list[indx][:3]
        angle_ = angle_ * 180 / np.pi
        tr = mpl.transforms.Affine2D().rotate_deg_around(x_, y_, angle_)
        t = tr + self.ax.transData
        self.robot_arr.remove()
        self.robot_arr = mpatches.Rectangle((x_ - 0.12, y_ - 0.08),
                                             0.24,
                                             0.16,
                                             transform=t,
                                             color='b',
                                             alpha=0.8,
                                             label='DIANA')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr