from intersection_sim import Simulator
from intersection2d_sim import Simulator
from pedestrian_agent import Agent, Agent2D
# from mm_mpc_inter import MM_MPC_TI
from mm_mpc_inter2d import MM_MPC_TI
from uncontrolled_agent import UncontrolledAgent

from utils import *
from path_planner import calc_spline_course

import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as tf
from celluloid import Camera
from IPython.display import HTML
from matplotlib.animation import FFMpegWriter   # ← import added
import pdb

initial_states = [[0.0, 2.0]]
final_states = [[100.0, 4.0]]

cost_func_params = {
    'Q': np.array([[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 2.5]]),
    'R': np.array([[.75, 0.0], [0.0, .5]]),
    'P': np.array([[12.5, 0.0], [0.0, 12.5]]),
    'Qc': 8,
    'kappa': 3 
}
mpc_params = {
    'num_agents': 1,
    'dt': 0.2,
    'N' : 12,
    'rob_dia': 0.3,
    'v_lim': 8.0,
    'omega_lim': 1.0,
    'total_sim_timestep': 200,
    'obs_sim_timestep': 100,
    'epsilon_o': 0.05,
    'epsilon_r': 0.05,
    'safety_margin': 0.05,
    'goal_tolerence': 0.2, 
    'linearized_ca': False
}

obs_traj = []
static_obs = []

obs = {"static": static_obs, "dynamic": obs_traj}


num_trials = 1
# algs = ["SM-MPC"]#, "MM-MPC", "Branch-MPC", "Robust-MPC"]
algs = ["MM-MPC", 'SM-MPC', 'Branch-MPC', 'Robust-MPC'][::-1]
# algs = ["SM-MPC"]
# algs = ["Robust-MPC"]
# algs = ["Branch-MPC"]
branch_times = [2]
clusters     = [2]
noise_levels = [0.01]
make_plots =False
if make_plots:
    results = summarize_algorithm_comparison_results("mm_results")
    # import pdb; pdb.set_trace()
    plot_algorithm_comparison_results(results)
    # "pass"
else:
    for noise_level in noise_levels[:1]:
        # for bt in branch_times[:1]:
        bt =2
        uncontrolled_fleet = UncontrolledAgent(init_state=[(0, 0, -np.pi/2)], dt=mpc_params['dt'], H=mpc_params['dt']*mpc_params['N'], action_variance=0.2)
        uncontrolled_fleet_data = uncontrolled_fleet.simulate_diff_drive()
        for cl in clusters:
            for trial in range(num_trials):
                ev_start_state = np.array([45, 5.7 + random.uniform(-0.2,0.2), 0])
                tv_scenario = random.choice([1,4])
                tv_start_state = random.choice([[0, 2+ random.uniform(-0.5,0.5)], [6, 0.1]])
                ped_start_state = random.choice([[0, 2+ random.uniform(-0.5,0.5)], [6, 0.1]])
                ped_decision_point = 8.+random.uniform(-0.5,0.5)
                for alg in algs:

                    if cl > clusters[0] and alg!="SM-MPC":
                        continue
                
                    ev_noise_std=[0.00001,0.00001]
                    ev=Agent2D(role='EV', cl=3, state=ev_start_state, dt = mpc_params['dt'], noise_std=ev_noise_std)
                    tv_noise_std=[noise_level]*2
                    agents=[Agent(role='TV', cl=tv_scenario, dt = mpc_params['dt'], state=np.array(tv_start_state), noise_std=tv_noise_std) for i in range(1)]
                    agents.append(Agent(role='ped', cl=6, dt = mpc_params['dt'], state=np.array(ped_start_state), noise_std=tv_noise_std, s_dec = ped_decision_point))
                    # # agents.append(Agent(role='ped', cl=9, state=np.array([0., 2.+ random.uniform(-0.1,0.1)]), noise_std=tv_noise_std, s_dec = 12+random.uniform(-0.5,0.5)))
                    # agents=[Agent(role='TV', cl=4, dt = mpc_params['dt'], state=np.array([6, 0.1]), noise_std=tv_noise_std) for i in range(1)]
                    # agents= []
                    # agents.append(Agent(role='ped', cl=7, dt = mpc_params['dt'], state=np.array([-1., 5+ random.uniform(-0.1,0.1)]), noise_std=tv_noise_std, s_dec = 6+random.uniform(-0.5,0.5)))
                    
                    tv_n_stds=[v.noise_std for v in agents]
                    agents.append(ev)
                    Sim=Simulator(agents, T_FINAL=120, K_max=cl)
                    
                    Sim.set_MPC_N(mpc_params['N'])
                    if alg!= "SM-MPC":
                        scenario = alg + "_" + "n_" + str(noise_level) + "_b_" + str(bt)+'_v3'
                    else:
                        scenario = alg + "_" + "n_" + str(noise_level) + "_c_" + str(cl)+'_v3'
                    

                    if alg in  ["MM-MPC", "SM-MPC"]:

                        mpc = MM_MPC_TI(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=True, robust_horizon=bt, ref=None)
                        # mpc = MM_MPC_TI(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj)
                        
                    elif alg == "Branch-MPC":
                        mpc = MM_MPC_TI(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=False, robust_horizon=bt, ref=None)
                    else:
                        mpc = MM_MPC_TI(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=False, robust_horizon=Sim.N, ref=None)
                        
                        
                    mpc.simulate(Sim)
                    
                    print(f"Finished algorithm {alg}, trial {trial}, noise level {noise_level}")
                    if trial == 0:
                        fig, ax= plt.subplots()
                        camera = Camera(fig)
                        for  i in range(Sim.t):
                            Sim.draw_intersection(ax, i)
                            camera.snap()
                        animation = camera.animate(repeat = True, repeat_delay = 100)
                        writer = FFMpegWriter(
                        fps=15,                    # frames per second
                        metadata=dict(artist='You'),
                        bitrate=1800)

                        # 3. Save to MP4
                        
                        animation.save(f'videos/intersection_{scenario}.mp4', writer=writer)

                        print(f"Saved animation for {scenario}")