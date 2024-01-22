# from mpc import MPC
from mm_mpc import MM_MPC
from mpc import MPC
from branch_mpc import Branch_MPC
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from pedestrian_agent import PedestrianSimulator

if __name__ == "__main__":
    initial_states = [[0.0, 2.0, np.pi/2]]
    final_states = [[0.0, 4.0, np.pi/2]]

    cost_func_params = {
        'Q': np.array([[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 2.5]]),
        'R': np.array([[5.5, 0.0], [0.0, .5]]),
        'P': np.array([[12.5, 0.0], [0.0, 12.5]]),
        'Qc': 8,
        'kappa': 3 
    }
    mpc_params = {
        'num_agents': 1,
        'dt': 0.1,
        'N' : 20,
        'rob_dia': 0.3,
        'v_lim': 1.0,
        'omega_lim': 1.0,
        'total_sim_timestep': 200,
        'obs_sim_timestep': 100,
        'epsilon_o': 0.05,
        'epsilon_r': 0.05,
        'safety_margin': 0.05,
        'goal_tolerence': 0.2, 
        'linearized_ca': True
    }

    obs_traj = []
    static_obs = []

    obs = {"static": static_obs, "dynamic": obs_traj}

    num_trials = 1
    algs = ["MPC"]
    vel_var_levels = [0.05, 0.4]
    rationality = 0.5
    T = 6
    y_pos = 3

    scenario = "test"
    trial = 1

    uncontrolled_agent = PedestrianSimulator(initial_position=0, initial_velocity=0.1, rationality=rationality, sim_time=T, dt=mpc_params['dt'], N=mpc_params['N'], y_pos=y_pos, vel_variance=0.01)
    predictions, uncontrolled_traj = uncontrolled_agent.simulate_pedestrian()

    mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj)
    mpc.simulate()
    
    # for trial in range(num_trials):
    #     for var_level in vel_var_levels:
    #         uncontrolled_agent = PedestrianSimulator(initial_position=0, initial_velocity=0.1, rationality=rationality, sim_time=T, dt=mpc_params['dt'], N=mpc_params['N'], y_pos=y_pos, vel_variance=var_level)
    #         predictions, uncontrolled_traj = uncontrolled_agent.simulate_pedestrian()
            
    #         for alg in algs:
    #             scenario = alg + "_" + "n_" + str(var_level)
    #             if alg == "MM-MPC":
    #                 mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj, feedback=True)
    #                 mpc.simulate()
    #             if alg == "Branch-MPC":
    #                 mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj)
    #                 mpc.simulate()
    #             else:
    #                 mpc = MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj)
    #                 mpc.simulate()
                