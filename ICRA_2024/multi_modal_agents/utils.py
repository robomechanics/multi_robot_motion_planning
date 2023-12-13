import random 
import numpy as np
import math
import numpy as np
import os 
import pickle 
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from draw import Draw_MPC_point_stabilization_v1
from collections import OrderedDict
import seaborn as sns
from collections import defaultdict

sns.set_palette("Set1")

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

def create_bar_plots(data_structure):
    input_sizes = list(data_structure.keys())
    algorithms = ['CB-MPC', 'PR-MPC', 'D-MPC']
    metrics = ['avg_comp_time', 'max_comp_time', 'makespan', 'success']
    titles = ['Average Computation Time Per Robot', "Max Computation Time Per Fleet", 'Makespan', 'Success Rate']
    y_labels = ['Average computation time per robot (sec)', 'Max computation time per fleet (sec)', 'Makespan (sec)', 'Success rate']

    for idx, metric in enumerate(metrics):
        plt.figure(figsize=(7,5))
        grouped_data = {algorithm: [] for algorithm in algorithms}
        error_data = {algorithm: [] for algorithm in algorithms}
        labels = []
        
        for algorithm in algorithms:
            for input_size in input_sizes:
                values = data_structure[input_size][algorithm][metric]
                if metric in ['avg_comp_time', 'makespan', 'max_comp_time']:
                    values = [value for value in values if value != 0.0]
                    avg_metric = np.mean(values)
                    std_metric = np.std(values) / np.sqrt(len(values))
                    grouped_data[algorithm].append(avg_metric)
                    error_data[algorithm].append(std_metric)
                elif metric == 'success':
                    success_rate = np.mean(values)
                    grouped_data[algorithm].append(success_rate)
                    error_data[algorithm].append(0)  # No error bar for success rate
            labels.append(algorithm)

        bar_width = 0.2
        x_positions = np.arange(len(input_sizes))
        
        for i, algorithm in enumerate(algorithms):
            plt.bar(x_positions + i * bar_width, grouped_data[algorithm], bar_width, yerr=error_data[algorithm], label=algorithm, capsize=5)

        plt.xlabel('Number of Robots', fontsize=15)
        plt.ylabel(y_labels[idx], fontsize=15)
        plt.title(titles[idx], fontsize=15)
        plt.xticks(x_positions + bar_width * (len(algorithms) - 1) / 2, input_sizes)
        plt.legend(fontsize=14)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tick_params(axis='both', which='both', top=False, right=False)

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
            algorithm_name = algorithm_name.split('_')[0]
            if num_robots not in robot_metrics:
                robot_metrics[num_robots] = {}
            
            if algorithm_name not in robot_metrics[num_robots]:
                robot_metrics[num_robots][algorithm_name] = {
                    "avg_comp_time": [],
                    "max_comp_time":[],
                    "makespan": [],
                    "success": [], 
                }

            algorithm_names.append(parts[0])
            for trial_file in os.listdir(os.path.join(base_folder, run_folder)):
                if trial_file.endswith(".pkl"):
                    trial_file_path = os.path.join(base_folder, run_folder, trial_file)
                    metrics = load_metrics(trial_file_path)
                    robot_metrics[num_robots][algorithm_name]["avg_comp_time"].append(metrics["avg_comp_time"])
                    robot_metrics[num_robots][algorithm_name]["max_comp_time"].append(metrics["max_comp_time"])
                    robot_metrics[num_robots][algorithm_name]["makespan"].append(metrics["makespan"])
                    robot_metrics[num_robots][algorithm_name]["success"].append(metrics["success"])

    sorted_robot_metrics = {key: robot_metrics[key] for key in sorted(robot_metrics)}
    x_labels = [str(robots) for robots in sorted_robot_metrics.keys()]
    algorithm_names = ["CB-MPC", "PR-MPC", "D-MPC"]

    sorted_robot_metrics = OrderedDict(sorted(robot_metrics.items(), key=lambda x: x[0]))
    create_bar_plots(sorted_robot_metrics)

def visualize_logged_run(foldername, trial_num):
    filename = "trial_" + str(trial_num) + ".pkl"
    file_path = os.path.join("open", foldername, filename)
    with open(file_path, "rb") as file:
        metrics = pickle.load(file)
        state_cache = metrics["state_cache"]
        initial_state = metrics["initial_state"]
        final_state = metrics["final_state"]
        map = metrics["map"]

        draw_result = Draw_MPC_point_stabilization_v1(
            rob_dia=0.3, init_state=initial_state, target_state=final_state, robot_states=state_cache, obs_state={"static": [], "dynamic": []}, map=map)

def print_metrics_summary(foldername, trial_num):
    filename = "trial_" + str(trial_num) + ".pkl"
    file_path = os.path.join("results", foldername, filename)

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
        execution_collision = metrics["execution_collision"]
        max_time_reached = metrics["max_time_reached"]

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
        else:
            print("Success:")
            print(bool(success))
            print("Collision:")
            print(execution_collision)
            print("Max time reached:")
            print(max_time_reached)
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

def get_obstacle_coordinates(occupancy_grid, current_position):
    obstacle_centers = []
    current_row = current_position[0]
    current_col = current_position[1]
        
    for row_idx, row in enumerate(occupancy_grid):
        for col_idx, cell in enumerate(row):
            if cell == 1:  # Assuming 1 represents an obstacle
                center_x = col_idx + 0.5  # Adding 0.5 to get the center
                center_y = row_idx + 0.5  # Adding 0.5 to get the center

                distance = ((center_x - current_col)**2 + (center_y - current_row)**2) ** 0.05
                if distance <= 4:
                    obstacle_centers.append((center_x, center_y))
    
    return obstacle_centers

def plot_success_rate(df):
    # Check if 'df' is a DataFrame and if 'algorithm' and 'noise_level' columns exist
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input is not a valid pandas DataFrame.")
    if 'algorithm' not in df.columns or 'noise_level' not in df.columns or 'success' not in df.columns:
        raise ValueError("DataFrame must contain 'algorithm', 'noise_level', and 'success' columns.")

    # Sort algorithms with MM-MPC first
    algorithms = sorted(df['algorithm'].unique(), key=lambda x: (x != 'MM-MPC', x))
    noise_levels = sorted(df['noise_level'].unique())
    bar_width = 0.2
    opacity = 0.8

    plt.figure(figsize=(12, 6))

    for i, algorithm in enumerate(algorithms):
        success_rates = df[df['algorithm'] == algorithm].groupby('noise_level')['success'].mean().reindex(noise_levels)
        bar_positions = [x + (i - len(noise_levels) / 2) * bar_width for x in noise_levels]
        plt.bar(bar_positions, success_rates, bar_width, alpha=opacity, label=algorithm)

    plt.xlabel('Noise Level')
    plt.ylabel('Success Rate')
    plt.title('Average Success Rate by Noise Level and Algorithm')
    plt.xticks(range(len(noise_levels)), noise_levels)
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_simulation_results(df):
    plot_success_rate(df)