from d_mpc import D_MPC
from vanilla_mpc import MPC
from pr_mpc import PR_MPC
from cb_mpc import CB_MPC
from joint_mpc import Joint_MPC
from task_generator import Task_Generator
import numpy as np
from cbs import Environment
from cbs import CBS
from utils import *
import matplotlib.pyplot as plt
from draw import Draw_MPC_point_stabilization_v1

if __name__ == "__main__":
    initial_states = [[-2.0, 0.0, 0.0], [2.0, 0.0, -np.pi], [-1.0, -2.0, -np.pi/2], [-1.0, 2.0, -np.pi/2], [1.0, -2.0, np.pi/2], [1.0, 2.0, -np.pi/2], [0.0, -2.0, np.pi/2], [0.0, 2.0, np.pi/2], [-2.0, -1.0, 0.0], [2.0, -1.0, np.pi], [-2.0, 1.0, 0.0], [2.0, 1.0, np.pi]]
    final_states = [[2.0, 0.0, 0.0], [-2.0, 0.0, -np.pi], [-1.0, 2.0, -np.pi/2], [-1.0, -2.0, -np.pi/2], [1.0, 2.0, -np.pi/2], [1.0, -2.0, np.pi/2], [0.0, 2.0, np.pi/2], [0.0, -2.0, np.pi/2], [2.0, -1.0, 0.0], [-2.0, -1.0, np.pi], [2.0, 1.0, 0.0], [-2.0, 1.0, np.pi]]

    # initial_states = [[1.0, 1.0, np.pi/2], [1.0, 3.0, -np.pi/2], [1.0, 0.0, np.pi/2], [1.0, 4.0, -np.pi/2], [1.0, 0.5, np.pi/2]]
    # final_states = [[1.0, 3.0, np.pi/2], [1.0, 1.0, -np.pi/2], [1.0, 4.0, np.pi/2], [1.0, 0.0, -np.pi/2], [1.0, 2.5, np.pi/2]]

    cost_func_params = {
        'Q':  np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]]),
        'R': np.array([[12.5, 0.0], [0.0, 0.05]]),
        'P': np.array([[12.5, 0.0], [0.0, 12.5]]),
        'Qc': 8,
        'kappa': 3 
    }
    mpc_params = {
        'num_agents': 12,
        'dt': 0.05,
        'N' : 60,
        'rob_dia': 0.3,
        'v_lim': 1.0,
        'omega_lim': 1.0,
        'total_sim_timestep': 500,
        'obs_sim_timestep': 100,
        'epsilon_o': 0.05,
        'epsilon_r': 0.05,
        'safety_margin': 0.05,
        'goal_tolerence': 0.2
    }
    # obs_traj = np.array(create_dynamic_obstacles(mpc_params['obs_sim_timestep'], int(mpc_params['obs_sim_timestep']/mpc_params['dt'])))
    obs_traj = []
    # static_obs = [[-1, 2, 1.0], [1, 2, 1.0], [0, 3, 0.5], [0, 1, 0.5]]
    static_obs = [[0.5, 2, 0.4], [1.5, 2, 0.4]]

    obs = {"static": static_obs, "dynamic": obs_traj}

    # save_gif_frame_as_png("cluttered_animation.gif", 57)

    map_size = (15,15)
    obstacle_density = 0.0
    map = generate_map(map_size, 0)

    # task_gen = Task_Generator(mpc_params["num_agents"], map, 2*mpc_params["rob_dia"])
    # initial_states, final_states = task_gen.generate_tasks()

    # visualize_scenario(map, initial_states, final_states)

    # call CBS to generate reference
    initial_states, final_states = shift_to_positive(initial_states, final_states)

    # env = Environment(map, map_size, initial_states[:mpc_params['num_agents']], final_states[:mpc_params['num_agents']])
    # cbs = CBS(env)
    # solution = cbs.search(20)
    
    # ref = discretize_waypoints(solution, mpc_params["dt"], mpc_params["N"])
    
    # if not solution:
    #     print("CBS Solution not found")

    # visualize_average_metrics()

    scenario = "open_12_robot"
    trial = 1

    # visualize_average_metrics()

    # print_metrics_summary("CB-MPC_open_12_robot", 1)
    # visualize_logged_run("CB-MPC_open_12_robot", 1)

    mpc = MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, map=map)
    mpc.simulate()

    # mpc = CB_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, map=map)
    # mpc.simulate()

    # mpc = Joint_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial)
    # mpc.simulate()


