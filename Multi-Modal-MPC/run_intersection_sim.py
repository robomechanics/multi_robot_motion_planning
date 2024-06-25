from intersection_sim import Simulator
from pedestrian_agent import Agent
from mm_mpc_inter import MM_MPC_TI
from uncontrolled_agent import UncontrolledAgent

import numpy as np



ev_noise_std=[0.001,0.01]
ev=Agent(role='EV', cl=3, noise_std=ev_noise_std)
tv_noise_std=[0.01, 0.1]
agents=[Agent(role='TV', cl=2, state=np.array([0, 7.]), noise_std=tv_noise_std) for i in range(1)]
agents.append(Agent(role='ped', cl=7, state=np.array([0., 4.]), noise_std=tv_noise_std, s_dec = 20))
agents.append(Agent(role='ped', cl=9, state=np.array([0., 2.]), noise_std=tv_noise_std, s_dec = 10))

tv_n_stds=[v.noise_std for v in agents]
agents.append(ev)
Sim=Simulator(agents)



Sim.set_MPC_N(10)

initial_states = [[0.0, 2.0]]
final_states = [[100.0, 4.0]]

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
    'N' : 10,
    'rob_dia': 0.3,
    'v_lim': 10.0,
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
uncontrolled_fleet = UncontrolledAgent(init_state=[(0, 0, -np.pi/2)], dt=mpc_params['dt'], H=mpc_params['dt']*mpc_params['N'], action_variance=0.1)
uncontrolled_fleet_data = uncontrolled_fleet.simulate_diff_drive()

mpc = MM_MPC_TI(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_fleet, uncontrolled_fleet_data, map=map, feedback=True, robust_horizon=2, ref=None)
# mpc = MM_MPC_TI(initial_states, final_states, cost_func_params, obs, mpc_params, scenario, trial, uncontrolled_agent, uncontrolled_traj)
mpc.simulate(Sim)


