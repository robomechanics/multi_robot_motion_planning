import numpy as np
import matplotlib.pyplot as plt
import math
from discrete_mpc import Discrete_MPC
from utils.create_obstacles import create_obstacles
from integration import rk_four
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    mpc = Discrete_MPC()
    curr_state = np.array([-2.0, 2.0, np.pi/2])
    goal_state = np.array([2.0, 0.0, np.pi/2])
    num_agent = 1
    robot_state_history = np.empty((num_agent, mpc.NUMBER_OF_TIMESTEPS, 3))
    best_rollout_history = []
    rollout_history = []
    robot_state = curr_state

    obstacles = create_obstacles(mpc.SIM_TIME, mpc.NUMBER_OF_TIMESTEPS + mpc.HORIZON_LENGTH)
    obstacles_arr = np.array(obstacles)
    
    state_rollouts, control_rollouts = mpc.perform_constant_vel_rollout(curr_state)
    best_state_rollout, best_control_rollout = mpc.get_best_rollout(state_rollouts, control_rollouts, obstacles_arr[0:2, 0:mpc.HORIZON_LENGTH, :], goal_state)
    best_rollout_history.append(best_state_rollout)
    
    for i in range(num_agent):
        for j in range(mpc.NUMBER_OF_TIMESTEPS):
            # predict the obstacles' position in future
            obstacle_predictions = obstacles_arr[0:2, j:j+mpc.HORIZON_LENGTH, :]
            state_rollouts, control_rollouts = mpc.perform_constant_vel_rollout(robot_state)
            rollout_history.append(state_rollouts)
            best_state_rollout, best_control_rollout = mpc.get_best_rollout(state_rollouts, control_rollouts, obstacles_arr[0:2, j:j+mpc.HORIZON_LENGTH, :], goal_state)
            best_rollout_history.append(best_state_rollout)
            next_state = rk_four(mpc.model.f, robot_state, best_control_rollout[:,0], mpc.TIMESTEP)

            robot_state = next_state
            robot_state_history[i, j, :] = next_state
        
        # with open(filename, "wb") as f:
        #     pickle.dump(data_log, f)

    print("Finished running experiment")
    
# Get the data for animation
x = best_state_rollout[0, :]
y = best_state_rollout[1, :]
robot_state_history = robot_state_history[0, :, :]
obstacle_predictions = np.array(obstacle_predictions)

# Create the figure and axis
fig, ax = plt.subplots()

# Initialize the line objects
line, = ax.plot([], [], linewidth=3.0)
scatter = ax.scatter([], [])

# Set the axis limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Animation update function
def update(frame):
    line.set_data([], [])
    plt.cla()  # Clear the current axis
    plt.clf()  # Clear the entire figure

    # Plot the best state rollout
    # line.set_data(x[:frame + 1], y[:frame + 1])
    
    # Plot the extracted elements
    plt.plot(robot_state_history[:frame + 1, 0], robot_state_history[:frame + 1, 1], color='blue')
    
    # Scatter plot of obstacle predictions
    for id in range(obstacles_arr.shape[2]):
        plt.scatter(obstacles_arr[0, :frame + 1, id], obstacles_arr[1, :frame + 1, id], color='red')

    rollout = rollout_history[frame]
    plt.plot(rollout[0, :, :], rollout[1, :, :], color='orange')

    best_rollout = best_rollout_history[frame]
    plt.plot(best_rollout[0, :], best_rollout[1, :], color='green')

    plt.xlim(-5, 5)  # Replace x_min and x_max with desired values
    plt.ylim(-5, 5) 

    return line, scatter

# Create the animation
animation = FuncAnimation(fig, update, frames=len(robot_state_history), interval=200, blit=True)
# animation.save('animation.mp4')

# Show the animation
plt.show()
