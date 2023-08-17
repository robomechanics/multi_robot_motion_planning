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

    cost_func_params = {
        'Q':  np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]]),
        'R': np.array([[7.5, 0.0], [0.0, 0.05]]),
        'P': np.array([[10.0, 0.0], [0.0, 10.0]]),
        'Qc': 8,
        'kappa': 3 
    }
    mpc_params = {
        'num_agents': 2,
        'dt': 0.05,
        'N' : 40,
        'rob_dia': 0.3,
        'v_lim': 1.0,
        'omega_lim': 1.0,
        'total_sim_timestep': 400,
        'obs_sim_timestep': 100,
        'epsilon_o': 0.05,
        'epsilon_r': 0.05,
        'safety_margin': 0.05,
        'goal_tolerence': 0.2
    }
    # obs_traj = np.array(create_dynamic_obstacles(mpc_params['obs_sim_timestep'], int(mpc_params['obs_sim_timestep']/mpc_params['dt'])))
    obs_traj = []
    # static_obs = [[-1, 2, 1.0], [1, 2, 1.0], [0, 3, 0.5], [0, 1, 0.5]]
    static_obs = []

    obs = {"static": static_obs, "dynamic": obs_traj}

    # print_metrics_summary("MPC_cluttered_2")
    # visualize_logged_run("CB-MPC_cluttered_16")
    
    num_trials = 1
    num_agents = [20]
    algorithms = ["CB-MPC", "PR-MPC", "D-MPC"]
    for num_agent in num_agents:
        scenario = "cluttered_" + str(num_agent)
        mpc_params["num_agents"] = num_agent

        map_size = (12,12)
        num_obstacles = 10
        map = generate_map(map_size, num_obstacles)

        task_gen = Task_Generator(num_agent, map, mpc_params["rob_dia"])
        initial_states, final_states = task_gen.generate_tasks()
        print("Generated tasks")
        
        # env = Environment(map, map_size, initial_states, final_states)
        # cbs = CBS(env)
        # solution = cbs.search()
        
        # ref = discretize_waypoints(solution, mpc_params["dt"], mpc_params["N"])
        # visualize_scenario(solution, map, initial_states, final_states)

        # if not solution:
        #     print("CBS Solution not found")

        for trial in range(num_trials):
            # mpc = DMPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, ref)
            # mpc.simulate()
            # print("Finished MPC")

            mpc = CB_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, map)
            mpc.simulate()
            print("Finished CB-MPC")

            # mpc = PR_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, ref)
            # mpc.simulate()
            # print("Finished PR-MPC")