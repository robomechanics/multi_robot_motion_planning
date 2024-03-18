from mm_mpc import MM_MPC
from mm_cbs import MM_CBS
from branch_mpc import Branch_MPC
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from uncontrolled_agent import UncontrolledAgent
from path_planner import calc_spline_course

if __name__ == "__main__":
    # initial_states = [[0.0, 0.0, -np.pi/2]]
    # final_states = [[0.0, 3.0, np.pi/2]]

    cost_func_params = {
        'Q': np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 2.0]]),
        'R': np.array([[2.0, 0.0], [0.0, 0.5]]),
        'P': np.array([[15.0, 0.0], [0.0, 15.0]]),
        'Qc': 8,
        'kappa': 3 
    }
    mpc_params = {
        'num_agents': 1,
        'dt': 0.2,
        'N' : 12,
        'rob_dia': 0.3,
        'v_lim': 1.0,
        'omega_lim': 1.0,
        'total_sim_timestep': 100,
        'obs_sim_timestep': 100,
        'epsilon_o': 0.05,
        'epsilon_r': 0.05,
        'safety_margin': 0.05,
        'goal_tolerence': 0.25, 
        'linearized_ca': True
    }

    obs_traj = []
    static_obs = []

    obs = {"static": static_obs, "dynamic": obs_traj}

    map_size = (15,15)
    obstacle_density = 0.0
    # map = generate_map(map_size, 0)

    num_trials = 30
    # algs = ["MM-MPC", "MLE-MPC", "Branch-MPC", "Robust-MPC"]
    algs = ["MM-MPC"]
    branch_times = [2, 6, 12]
    noise_levels = [0.2]

    # results, errors = summarize_algorithm_comparison_results("mm_results_arch")
    # plot_algorithm_comparison_results(results, errors)
    
    # results = summarize_ablation_comparison_results("mm_results")
    # plot_ablation_comparison_results(results)

    # animate_trial("MM-MPC_n_0.5_b_2", 1)
    # plot_key_timesteps("Robust-MPC_n_0.1", 8, [1,5,10,15,20,25,30])

    for noise_level in noise_levels:
        for bt in branch_times:
            for trial in range(num_trials):
                initial_states = [[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), np.pi/2]]
                final_states = [[random.uniform(-0.1, 0.1), random.uniform(4.5, 5.0), np.pi/2]]
                
                x_unc = random.uniform(-0.1, 0.1) 
                y_unc = random.uniform(2.5, 3.0) 
                
                uncontrolled_fleet = UncontrolledAgent(init_state=[(x_unc, y_unc, 0.0)], dt=mpc_params['dt'], H=mpc_params['dt']*mpc_params['N'], action_variance=noise_level)
                uncontrolled_fleet_data = uncontrolled_fleet.simulate_diff_drive()
                
                for alg in algs:                
                    # for bt in branch_times:          
                    rx, ry, ryaw, rk, s = calc_spline_course([initial_states[0][0], final_states[0][0]], [initial_states[0][1], final_states[0][1]])
                    ref = [[x, y, yaw] for x, y, yaw in zip(rx, ry, ryaw)]
                    
                    for alg in algs:
                        scenario = alg + "_" + "n_" + str(noise_level) + "_b_" + str(bt)
                        if alg == "MM-MPC":
                            mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=True, robust_horizon=bt, ref=ref)
                            mpc.simulate()
                        elif alg == "Branch-MPC":
                            mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=False, robust_horizon=2, ref=ref)
                            mpc.simulate()
                        elif alg == "MLE-MPC":
                            mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=True, robust_horizon=mpc_params['N'], ref=ref, mle=True)
                            mpc.simulate()
                        else:
                            mpc = MM_MPC(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=False, robust_horizon=mpc_params['N'], ref=ref)
                            mpc.simulate()