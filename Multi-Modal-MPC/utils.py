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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter 

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

def summarize_algorithm_comparison_results(folder_path):
    results = {}
    errors = {}  # Dictionary to store errors

    for subfolder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subfolder)):
            parts = subfolder.split('_')
            algorithm = parts[0]
            noise_level = parts[2]

            # Lists to store individual values for calculating standard deviation
            infeasible_ratios = []
            task_completion_times = []
            avg_comp_times = []
            max_comp_times = []
            control_mag_avgs = []

            num_trials = 0
            success = 0
            for file in os.listdir(os.path.join(folder_path, subfolder)):
                if file.endswith('.pkl'):
                    with open(os.path.join(folder_path, subfolder, file), 'rb') as f:
                        data = pickle.load(f)
                        success += data['success']
                        num_timesteps = data['num_timesteps']
                        infeasible_count = data['infeasible_count']
                        avg_comp_time = np.mean(data['avg_comp_time'])
                        max_comp_time = np.max(data['max_comp_time'])
                        control_mag_avg = np.linalg.norm(data['control_cache'][0][1])

                        # Store individual values
                        infeasible_ratios.append(infeasible_count / num_timesteps)
                        task_completion_times.append(num_timesteps * 0.2)
                        avg_comp_times.append(avg_comp_time)
                        max_comp_times.append(max_comp_time)
                        control_mag_avgs.append(control_mag_avg)

                        num_trials += 1

            # Calculate averages
            infeasible_ratio = np.mean(infeasible_ratios)
            task_completion_time = np.mean(task_completion_times)
            avg_comp_time = np.mean(avg_comp_times)
            max_comp_time = np.mean(max_comp_times)
            control_mag_avg = np.mean(control_mag_avgs)
            success_rate = success / num_trials

            # Calculate standard deviations
            infeasible_ratio_error = np.std(infeasible_ratios)
            task_completion_time_error = np.std(task_completion_times)
            avg_comp_time_error = np.std(avg_comp_times)
            max_comp_time_error = np.std(max_comp_times)
            control_mag_avg_error = np.std(control_mag_avgs)

            # Initialize dictionary structures if needed
            if noise_level not in results:
                results[noise_level] = {}
                errors[noise_level] = {}
            if algorithm not in results[noise_level]:
                results[noise_level][algorithm] = {}
                errors[noise_level][algorithm] = {}

            # Store metrics and errors in the dictionaries
            results[noise_level][algorithm]['infeasible_ratio'] = infeasible_ratio
            results[noise_level][algorithm]['task_completion_time'] = task_completion_time
            results[noise_level][algorithm]['avg_comp_time'] = avg_comp_time
            results[noise_level][algorithm]['max_comp_time'] = max_comp_time
            results[noise_level][algorithm]['control_mag_avg'] = control_mag_avg
            results[noise_level][algorithm]['success_rate'] = success_rate

            errors[noise_level][algorithm]['infeasible_ratio'] = infeasible_ratio_error
            errors[noise_level][algorithm]['task_completion_time'] = task_completion_time_error
            errors[noise_level][algorithm]['avg_comp_time'] = avg_comp_time_error
            errors[noise_level][algorithm]['max_comp_time'] = max_comp_time_error
            errors[noise_level][algorithm]['control_mag_avg'] = control_mag_avg_error

    return results, errors

def summarize_ablation_comparison_results(folder_path):
    results = {}
    for subfolder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subfolder)):
            parts = subfolder.split('_')
            algorithm = parts[0]
            noise_level = parts[2]
            branch_time = parts[4]

            infeasible_count = 0
            num_timesteps = 0
            comp_times = []  # List to store all avg_comp_time values
            max_comp_times = []  # List to store all max_comp_time values
            control_mag_avg = 0
            task_completion_times = []  # List to store all task_completion_time values
            num_trials = 0

            for file in os.listdir(os.path.join(folder_path, subfolder)):
                if file.endswith('.pkl'):
                    with open(os.path.join(folder_path, subfolder, file), 'rb') as f:
                        data = pickle.load(f)
                        num_timesteps += data['num_timesteps']
                        infeasible_count += data['infeasible_count']
                        comp_times.append(np.mean(data['avg_comp_time']))  # Add to list
                        max_comp_times.append(np.mean(data['max_comp_time']))  # Add to list
                        control_mag_avg += np.linalg.norm(data['control_cache'][0][1])
                        # Calculate and store task completion time for this trial
                        task_completion_times.append((data['num_timesteps'] * 0.2))
                        num_trials += 1

            # Calculate averages and standard deviations
            infeasible_ratio = infeasible_count / num_timesteps if num_timesteps > 0 else 0
            avg_comp_time = np.mean(comp_times)
            std_avg_comp_time = np.std(comp_times)
            max_comp_time = np.mean(max_comp_times)
            std_max_comp_time = np.std(max_comp_times)
            avg_task_completion_time = np.mean(task_completion_times)
            std_task_completion_time = np.std(task_completion_times)
            control_mag_avg = control_mag_avg / num_trials

            # Initialize dictionary structure if needed
            if noise_level not in results:
                results[noise_level] = {}
            if branch_time not in results[noise_level]:
                results[noise_level][branch_time] = {}
            if algorithm not in results[noise_level][branch_time]:
                results[noise_level][branch_time][algorithm] = {}

            # Store metrics and their standard deviations in the dictionary
            results[noise_level][branch_time][algorithm]['infeasible_ratio'] = infeasible_ratio
            results[noise_level][branch_time][algorithm]['task_completion_time'] = avg_task_completion_time
            results[noise_level][branch_time][algorithm]['std_task_completion_time'] = std_task_completion_time
            results[noise_level][branch_time][algorithm]['avg_comp_time'] = avg_comp_time
            results[noise_level][branch_time][algorithm]['std_avg_comp_time'] = std_avg_comp_time
            results[noise_level][branch_time][algorithm]['max_comp_time'] = max_comp_time
            results[noise_level][branch_time][algorithm]['std_max_comp_time'] = std_max_comp_time
            results[noise_level][branch_time][algorithm]['control_mag_avg'] = control_mag_avg

    return results

def compute_average_norm(feedback_gain_map):
    """
    Computes the average Frobenius norm of an "average" matrix derived from all matrices 
    across all modes in the provided map.
    
    Parameters:
    - matrix_map: A dictionary where each key is an integer and each value is a list of NumPy arrays.
    
    Returns:
    - The Frobenius norm of the element-wise average of all matrices across all modes.
    """
    # Aggregate all matrices into a single list
    all_matrices = [matrix for matrices in feedback_gain_map.values() for matrix in matrices]
    
    # Compute the element-wise average of these matrices
    average_matrix = np.mean(all_matrices, axis=0)
    
    # Compute the Frobenius norm of this "average" matrix
    average_norm = np.linalg.norm(average_matrix, 'fro')
    
    return average_norm

def plot_algorithm_comparison_results(results, errors):
    algorithm_order = ["MM-MPC", "MLE-MPC", "Branch-MPC", "Robust-MPC"]
    data_types = ['infeasible_ratio', 'task_completion_time', 'avg_comp_time', 'max_comp_time', 'control_mag_avg', 'success_rate']
    colors = ['#2E86AB', '#D7263D', '#44A08D', '#F2C14E', '#F29F05', '#A23B72', '#75B1A9']
    # colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    titles = ["Infeasible Solve Ratio", "Task Completion Time", "Average Computation Time", "Max Computation Time", "Average Velocity", "Success_rate"]

    noise_levels = sorted({float(noise_level) for noise_level in results.keys()})
    n_groups = len(noise_levels)
    n_algorithms = len(algorithm_order)
    bar_width = 0.5 / n_algorithms
    offset = (1 - 0.5) / 2

    for i, data_type in enumerate(data_types):
        fig, ax = plt.subplots(figsize=(8, 8))
        index = np.arange(n_groups)

        plt.rcParams['font.size'] = 18  # Change default font size

        for j, algorithm in enumerate(algorithm_order):
            performance = []
            error_values = []  # Store the standard deviations or errors here
            for noise_level in noise_levels:
                data_key = str(noise_level)
                if algorithm in results[data_key]:
                    performance.append(results[data_key][algorithm].get(data_type, 0))
                    error_values.append(errors[data_key][algorithm].get(data_type, 0))  # Get the corresponding error
                else:
                    performance.append(0)
                    error_values.append(0)

            bar_positions = index + offset + j * bar_width
            plt.bar(bar_positions, performance, bar_width, label=algorithm, color=colors[j % len(colors)], yerr=error_values, capsize=5)
            plt.xticks(fontsize=16) 
            plt.yticks(fontsize=16) 

        plt.xlabel('Noise Level', fontsize=20)
        plt.ylabel(data_type.replace('_', ' ').title(), fontsize=20)
        plt.title(titles[i], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(index + 0.4, [str(n) for n in noise_levels])
        plt.ylim(0,0.25)
        # plt.legend(frameon=False, facecolor='none')

        plt.tight_layout()
        plt.show()

def plot_ablation_comparison_results(results):
    metrics = ['task_completion_time', 'infeasible_ratio']
    colors = ['#2E86AB', '#D7263D', '#44A08D', '#F29F05', '#A23B72', '#75B1A9']
    algorithms = set(alg for noise_level in results.values() for branch_time in noise_level.values() for alg in branch_time)
    
    bar_width = 0.1  # Width of the bars in the bar chart
    for algorithm in algorithms:
        print(algorithm)
        for metric in metrics:
            plt.figure(figsize=(8, 8))
            plt.rcParams['font.size'] = 16  # Change default font size
            
            data_for_plot = {}
            error_data = {}  # New dictionary to store error data
            for noise_level in sorted(results, key=lambda x: float(x)):
                for branch_time in results[noise_level]:
                    if algorithm in results[noise_level][branch_time]:
                        if branch_time not in data_for_plot:
                            data_for_plot[branch_time] = []
                            error_data[branch_time] = []  # Initialize list for error data
                        value = results[noise_level][branch_time][algorithm][metric]
                        std_value = results[noise_level][branch_time][algorithm].get(f'std_{metric}', 0)  # Get std deviation
                        data_for_plot[branch_time].append((float(noise_level), value))
                        error_data[branch_time].append((float(noise_level), std_value))  # Append std deviation
            
            for branch_time in data_for_plot:
                data_for_plot[branch_time].sort(key=lambda x: x[0])
                error_data[branch_time].sort(key=lambda x: x[0])  # Sort error data
            
            unique_branch_times = sorted(data_for_plot.keys(), key=lambda x: float(x))
            num_branch_times = len(unique_branch_times)
            
            for idx, branch_time in enumerate(unique_branch_times):
                values = data_for_plot[branch_time]
                errors = error_data[branch_time]  # Extract error values
                noise_levels, metric_values = zip(*values)
                _, std_values = zip(*errors)  # Only need the std deviation values
                
                offsets = np.arange(len(noise_levels))
                center_adjustment = (bar_width * num_branch_times) / 2 - (bar_width / 2)
                positions = offsets - center_adjustment + idx * bar_width
                
                plt.bar(positions, metric_values, width=bar_width, color=colors[idx % len(colors)], label=f'BT = {branch_time}', yerr=std_values, capsize=5)  # Add error bars
            
            plt.title(f"{metric.replace('_', ' ').title()} at Different Branching Times", fontsize=18)
            plt.xlabel('Noise Level', fontsize=18)
            plt.xticks(offsets, labels=[str(nl) for nl in noise_levels], fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=18)
            plt.legend()

            # Remove right and top borders
            ax = plt.gca()  # Get current axes
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.show()

def animate_trial(folder_name, trial_number):
    # Construct the file path
    file_path = f"mm_results/{folder_name}/trial_{trial_number}.pkl"
        
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract control_cache, state_cache, uncontrolled_traj, and predictions
    control_cache = data['control_cache'][0]
    state_cache = data['state_cache'][0]
    uncontrolled_traj = data['uncontrolled_fleet_data'][0]['executed_traj']
    predictions = data['uncontrolled_fleet_data'][0]['predictions']

    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle('Execution Results for Robust-MPC')
    
    # Initial setup for state plot
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.axvline(x=-1, color='k', linestyle='--', linewidth=2)
    ax1.axvline(x=1, color='k', linestyle='--', linewidth=2)
    ax1.set_xlabel('X') 
    ax1.set_ylabel('Y')
    controlled_circle = plt.Circle((0, 0), 0.3, fill=True, color='blue', label='Controlled Agent')
    uncontrolled_circle = plt.Circle((0, 0), 0.3, fill=True, color='red', label='Uncontrolled Agent')
    ax1.add_patch(controlled_circle)
    ax1.add_patch(uncontrolled_circle)
    
    # Initialize a line to trace the circle's path
    trajectory, = ax1.plot([], [], 'r-', linewidth=2)
    trajectory_data = {'x': [], 'y': []}

    # Predictions visualization setup
    prediction_lines = []
    for mode in range(len(predictions)):
        line, = ax1.plot([], [], 'k', linewidth=2)
        prediction_lines.append(line)

    # Initial setup for control plot
    control_line, = ax2.plot([], [], 'k-')  # Initialize an empty line for control values
    control_data = {'frames': [], 'values': []}  # Initialize the dictionaries to store control plot data

    ax2.set_xlim(0, len(state_cache))  # Assuming frames equal the length of state_cache
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Velocity [m/s]")

    def update(frame):
        # Update state plot for controlled agent
        x, y, _ = state_cache[frame]  # Assuming state_cache contains (x, y, theta) tuples
        controlled_circle.center = (x, y)
        
        # Update the trajectory data and line
        trajectory_data['x'].append(x)
        trajectory_data['y'].append(y)
        trajectory.set_data(trajectory_data['x'], trajectory_data['y'])

        # Update state plot for uncontrolled agent
        ux, uy, _ = uncontrolled_traj[frame]
        uncontrolled_circle.center = (ux, uy)

        # Update predictions
        for mode, line in enumerate(prediction_lines):
            mode_predictions = predictions[mode][frame]  # Get predictions for the current frame and mode
            px, py = zip(*[(pred[0], pred[1]) for pred in mode_predictions])  # Extract x, y coordinates
            line.set_data(px, py)

        # Update control plot data lists
        control_data['frames'].append(frame)  # Append the current frame number
        control_data['values'].append(control_cache[frame][0][1])  # Append the current control value
        
        # Update control plot
        control_line.set_data(control_data['frames'], control_data['values'])  # Update the control plot with accumulated data
        ax2.set_ylim(-1.5, 1.5)
        
        return [controlled_circle, uncontrolled_circle, trajectory, control_line] + prediction_lines
   
    ani = FuncAnimation(fig, update, frames=len(state_cache), interval=100, blit=True, repeat=False)
    
    plt.legend(loc="upper left")
    plt.show()

    # Specify the writer and options (FFmpeg is a common choice, but you might need to adjust based on your system)
    
    # Specify the filename and path where you want to save the animation
    # animation_file_name = f"trial_{trial_number}_animation.gif"
    
    # # Save the animation
    # writer = PillowWriter(fps=30)  # Adjust fps as needed
    # ani.save(animation_file_name, writer=writer)   
   
    plt.show()

def plot_key_timesteps(folder_name, trial_number, key_timesteps):
    # Construct the file path
    file_path = f"mm_results/{folder_name}/trial_{trial_number}.pkl"
    
    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract necessary data
    state_cache = data['state_cache'][0]
    uncontrolled_traj = data['uncontrolled_fleet_data'][0]['executed_traj']
    # predictions = data['uncontrolled_fleet_data'][0]['predictions']

    # Initialize lists to store trajectory points
    controlled_traj_x = []
    controlled_traj_y = []
    uncontrolled_traj_x = []
    uncontrolled_traj_y = []

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8,8))

    plt.title('Robust-MPC', fontsize=18)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axvline(x=-1, color='k', linestyle='--', linewidth=2)
    ax.axvline(x=1, color='k', linestyle='--', linewidth=2)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')

    # Plot each key timestep
    for i, frame in enumerate(key_timesteps):
        # Calculate transparency based on the timestep's position in the list
        alpha = (i + 1) / len(key_timesteps)

        # Controlled agent (Robot)
        x, y, _ = state_cache[frame]
        controlled_circle = plt.Circle((x, y), 0.3, fill=True, color='blue', alpha=alpha, label='Controlled Agent' if i == 0 else "")
        ax.add_patch(controlled_circle)
        controlled_traj_x.append(x)
        controlled_traj_y.append(y)

        # Uncontrolled agent (Pedestrian)
        ux, uy, _ = uncontrolled_traj[frame]
        uncontrolled_circle = plt.Circle((ux, uy), 0.3, fill=True, color='red', alpha=alpha, label='Uncontrolled Agent' if i == 0 else "")
        ax.add_patch(uncontrolled_circle)
        uncontrolled_traj_x.append(ux)
        uncontrolled_traj_y.append(uy)

        # Plot trajectories
        ax.plot(controlled_traj_x, controlled_traj_y, color='blue', alpha=0.5, linewidth=2)
        ax.plot(uncontrolled_traj_x, uncontrolled_traj_y, color='red', alpha=0.5, linewidth=2)

        ax.axis('Off')

          # Plot lane boundaries
        ax.axvline(x=-1, color='k', linestyle='--', linewidth=2)
        ax.axvline(x=1, color='k', linestyle='--', linewidth=2)

    plt.show()