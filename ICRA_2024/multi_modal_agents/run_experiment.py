from mpc import MPC
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from uncontrolled_agent import UncontrolledAgent

if __name__ == "__main__":
    initial_states = [[0.0, 0.0, 0.0]]
    final_states = [[5.0, 0.0, 0.0]]

    cost_func_params = {
        'Q':  np.array([[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 2.5]]),
        'R': np.array([[7.5, 0.0], [0.0, 3.5]]),
        'P': np.array([[12.5, 0.0], [0.0, 12.5]]),
        'Qc': 8,
        'kappa': 3 
    }
    mpc_params = {
        'num_agents': 1,
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

    obs_traj = []
    static_obs = [[2.0, 0.5, 0.6]]

    obs = {"static": static_obs, "dynamic": obs_traj}

    map_size = (15,15)
    obstacle_density = 0.0
    map = generate_map(map_size, 0)

    scenario = "test_1_robot"
    trial = 1

    uncontrolled_agent = UncontrolledAgent()
    uncontrolled_traj = uncontrolled_agent.simulate_diff_drive()

    mpc = MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj, map=map)
    mpc.simulate()


