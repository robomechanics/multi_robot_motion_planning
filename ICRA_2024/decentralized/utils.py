import random 
import numpy as np
from node import Node
import math
import numpy as np
import os 
import pickle 
import matplotlib.pyplot as plt
from PIL import Image
from draw import Draw_MPC_point_stabilization_v1

def distance_between_points(p1, p2):
    return np.linalg.norm(p1 - p2)

def get_avg_rob_dist(state_cache):
    total_distance = 0.0
    num_distances = 0

    # Convert the robot_data dictionary to a list of trajectory arrays
    trajectories = list(state_cache.values())

    # Convert 1x3 arrays to 2D arrays
    for i in range(len(trajectories)):
        trajectories[i] = np.array(trajectories[i])

    # Get the number of robots and the number of time steps in the trajectory
    num_robots = len(trajectories)
    
    # Find the minimum number of time steps among all robots' trajectories
    num_timesteps = min(trajectory.shape[0] for trajectory in trajectories)

    # Calculate the distances between all pairs of robots at each time step
    for i in range(num_timesteps):
        for j in range(i + 1, num_timesteps):
            for r1 in range(num_robots):
                for r2 in range(r1 + 1, num_robots):
                    distance = distance_between_points(trajectories[r1][i], trajectories[r2][j])
                    total_distance += distance
                    num_distances += 1

    return total_distance / num_distances


def get_traj_length(state_cache):
    traj_len = 0.0
    for agent_id, traj in state_cache.items():
        for i in range(1, len(traj)):
            x1, y1, theta1 = traj[i - 1]
            x2, y2, theta2 = traj[i]
            
            dx = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            traj_len += dx

    return traj_len

def get_best_node(conflict_tree):
    lowest_cost_node = min(conflict_tree, key=lambda node: node.cost)
    return lowest_cost_node

def generate_map(map_size, num_obstacles):
    num_cells = map_size[0] * map_size[1]    
    occupancy_grid = np.zeros(map_size, dtype=int)
    
    # Place obstacles randomly on the grid
    obstacle_indices = random.sample(range(num_cells), num_obstacles)
    for index in obstacle_indices:
        row = index // map_size[1]
        col = index % map_size[1]
        occupancy_grid[row][col] = 1
    
    return occupancy_grid

def discretize_waypoints(waypoints_dict, dt, N):
    discretized_dict = {}

    for robot_id, waypoints in waypoints_dict.items():
        discretized_waypoints = []
        for i in range(len(waypoints) - 1):
            start_time = waypoints[i]['t']
            end_time = waypoints[i + 1]['t']
            time_diff = end_time - start_time
            num_steps = int(time_diff / dt)

            if num_steps == 0:
                discretized_waypoints.append(waypoints[i])
            else:
                for step in range(num_steps):
                    fraction = float(step + 1) / float(num_steps + 1)
                    x = waypoints[i]['x'] + fraction * (waypoints[i + 1]['x'] - waypoints[i]['x'])
                    y = waypoints[i]['y'] + fraction * (waypoints[i + 1]['y'] - waypoints[i]['y'])
                    t = start_time + (step + 1) * dt
                    discretized_waypoints.append({'t': t, 'x': x, 'y': y})

        discretized_waypoints.append(waypoints[-1])  # Add the last waypoint as is
        discretized_dict[robot_id] = discretized_waypoints

    return discretized_dict


def load_metrics(trial_file_path):
    with open(trial_file_path, 'rb') as file:
        metrics = pickle.load(file)
    return metrics

def generate_grouped_bar_plot(data, x_labels, ylabel, title, legend_labels):
    num_bars = len(data)
    num_sub_bars = len(data[0])
    width = 0.2
    x_positions = np.arange(num_bars)
    
    plt.figure(figsize=(10, 6))
    
    for i in range(num_sub_bars):
        sub_data = [entry[i] for entry in data]
        sub_x_positions = x_positions + (i - num_sub_bars/2) * width
        plt.bar(sub_x_positions, sub_data, width=width, label=legend_labels[i])
    
    plt.xlabel("Number of Robots")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_positions, x_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_average_metrics(base_folder="results"):
    robot_metrics = {}
    algorithm_names = []
    
    for run_folder in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, run_folder)):
            parts = run_folder.split("_")
            for part in parts:
                if part.isdigit():
                    num_robots = int(part)  # Assuming the numeric part represents the number of robots
                    break
            else:
                continue
            
            algorithm_name = "_".join(parts[:-1])  # Exclude the number of robots
            if num_robots not in robot_metrics:
                robot_metrics[num_robots] = {
                    "avg_comp_time": [],
                    "makespan": [],
                    "success_rate": []
                }

            algorithm_names.append(parts[0])

            for trial_file in os.listdir(os.path.join(base_folder, run_folder)):
                if trial_file.endswith(".pkl"):
                    trial_file_path = os.path.join(base_folder, run_folder, trial_file)
                    metrics = load_metrics(trial_file_path)
                    robot_metrics[num_robots]["avg_comp_time"].append(np.mean(metrics["avg_comp_time"]))
                    robot_metrics[num_robots]["makespan"].append(np.mean(metrics["makespan"]))

    sorted_robot_metrics = {key: robot_metrics[key] for key in sorted(robot_metrics)}
    
    x_labels = [str(robots) for robots in sorted_robot_metrics.keys()]
    data_avg_comp_time = [sorted_robot_metrics[robots]["avg_comp_time"] for robots in sorted_robot_metrics]
    data_makespan = [sorted_robot_metrics[robots]["makespan"] for robots in sorted_robot_metrics]
    # Uncomment the line below to include success rate
    # data_success_rate = [sorted_robot_metrics[robots]["success_rate"] for robots in sorted_robot_metrics]
    
    # algorithm_names = [f"Algorithm {i+1}" for i in range(len(data_avg_comp_time))]
    generate_grouped_bar_plot(data_avg_comp_time, x_labels, "Average Computation Time (seconds)", "Average Computation Time", algorithm_names)
    generate_grouped_bar_plot(data_makespan, x_labels, "Average Makespan (seconds)", "Average Makespan", algorithm_names)

def visualize_logged_run(foldername):
    file_path = os.path.join("results", foldername, "trial_0.pkl")
    with open(file_path, "rb") as file:
        metrics = pickle.load(file)
        state_cache = metrics["state_cache"]
        initial_state = metrics["initial_state"]
        final_state = metrics["final_state"]
        map = metrics["map"]

        draw_result = Draw_MPC_point_stabilization_v1(
            rob_dia=0.3, init_state=initial_state, target_state=final_state, robot_states=state_cache, obs_state={"static": [], "dynamic": []}, map=map)

def print_metrics_summary(foldername):
    file_path = os.path.join("results", foldername, "trial_1.pkl")

    # Load the metrics data from the .pkl file
    with open(file_path, "rb") as file:
        metrics = pickle.load(file)
        # Returns the collected metrics data
      
        avg_computation_time = metrics["avg_comp_time"]
        max_computation_time = metrics["max_comp_time"]
        traj_length = metrics["traj_length"]
        makespan = metrics["makespan"]
        avg_rob_dist = metrics["avg_rob_dist"]
        success = metrics["success"]
        c_avg = metrics["c_avg"]
        state_cache = metrics["state_cache"]

        if(success):
            print("Avg Comp Time:")
            print(avg_computation_time)
            print("Max Comp time:")
            print(max_computation_time)
            print("Traj Length:")
            print(traj_length)
            print("Makespan:")
            print(makespan)
            print("Avg Rob Distance:")
            print(avg_rob_dist)
            print("C_avg:")
            print(c_avg)
            print("Success:")
            print(bool(success))
            print("State cache")
            print(state_cache)
            print("===================")
        else:
            print(state_cache)
            print(bool(success))
            print("===================")

def shift_to_positive(initial_states, final_states):
    # Convert the lists to NumPy arrays for easier manipulation
    initial_states = np.array(initial_states)
    final_states = np.array(final_states)
    
    # Find minimum x and y values across both initial and final states
    min_x = min(np.min(initial_states[:, 0]), np.min(final_states[:, 0]))
    min_y = min(np.min(initial_states[:, 1]), np.min(final_states[:, 1]))
    
    # Calculate shift amounts for x and y
    shift_x = max(0, -min_x)
    shift_y = max(0, -min_y)
    
    # Apply the shift to both initial and final states
    shifted_initial_states = initial_states + [shift_x, shift_y, 0]
    shifted_final_states = final_states + [shift_x, shift_y, 0]
    
    return shifted_initial_states, shifted_final_states

def save_gif_frame_as_png(gif_filename, frame_index):
    try:
        gif = Image.open(gif_filename)

        # Get the specific frame
        gif.seek(frame_index)

        # Convert the frame to RGBA mode (necessary for saving as PNG)
        frame = gif.convert("RGBA")

        # Generate the output PNG filename based on the input GIF filename
        output_filename = os.path.splitext(gif_filename)[0] + f"_frame_{frame_index}.png"

        # Save the frame as a PNG file
        frame.save(output_filename, "PNG")

        print(f"Frame {frame_index} saved as {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

import matplotlib.pyplot as plt

def visualize_scenario(waypoints_dict, occupancy_grid, initial_states, final_states):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the occupancy grid
    ax.imshow(occupancy_grid, cmap='gray', origin='lower')

    # Assign colors for agents
    num_agents = len(waypoints_dict)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))

    # Plot waypoints for each agent
    for i, (agent_id, waypoints) in enumerate(waypoints_dict.items()):
        color = colors[i]
        x_vals = [waypoint['x'] for waypoint in waypoints]
        y_vals = [waypoint['y'] for waypoint in waypoints]
        ax.plot(x_vals, y_vals, marker='o', label=f'Agent {agent_id}', color=color)

    # Plot initial and final positions
    for i, (initial, final) in enumerate(zip(initial_states, final_states)):
        color = colors[i % num_agents]
        ax.plot(initial[0], initial[1], marker='o', color=color, markersize=8)
        ax.plot(final[0], final[1], marker='x', color=color, markersize=8)

    # Set labels and legend
    plt.show()