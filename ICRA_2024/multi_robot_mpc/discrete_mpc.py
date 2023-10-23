"""
Collision avoidance using Nonlinear Model-Predictive Control

author: Ashwin Bose (atb033@github.com)
"""

from utils.multi_robot_plot_mpc import plot_robot_and_obstacles
from utils.sim import *
import numpy as np
import time
from generate_maps import Map
import cbs
import pickle
import time
from integration import rk_four
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpc_base import MPC_Base
from draw import Draw_MPC_point_stabilization_v1

class Discrete_MPC(MPC_Base):

    def perform_constant_vel_rollout(self, init_state):
        lin_vel = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ang_vel = [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8]
        num_rollouts = len(lin_vel) * len(ang_vel)
        state_rollouts = np.zeros((3, self.N, num_rollouts)) 
        control_rollouts = np.zeros((2, self.N, num_rollouts))
        for idx, (v, w) in enumerate(itertools.product(lin_vel, ang_vel)):
            curr_state = init_state
            for i in range(self.N):
                u = self.model.uni2diff(np.array([v, w]))
                control_rollouts[:, i, idx] = u
                next_state = rk_four(self.model.f, curr_state, u, self.dt)
                state_rollouts[:, i, idx] = next_state

                curr_state = next_state
        
        return state_rollouts, control_rollouts

    def get_best_rollout(self, state_rollouts, control_rollouts, obstacle_positions, goal_state):
        rollout_costs = []
        num_rollouts = state_rollouts.shape[2]
       
        for idx1 in range(num_rollouts):
            goal_cost = np.linalg.norm(state_rollouts[0:2, self.N - 1, idx1] - goal_state[0:2])
            collision_cost = self.total_collision_cost(state_rollouts[0:2, :, idx1], obstacle_positions)
            total_cost = goal_cost + 5 * collision_cost

            rollout_costs.append(total_cost)

        min_index = rollout_costs.index(min(rollout_costs))

        return state_rollouts[:, :, min_index], control_rollouts[:, :, min_index]        

    def simulate(self):
        ##### DATA LOGGING
        data_log = {}
        computation_time = []
        solution_cost = 0.0
        control_std = []
        rollout_history = []
        best_rollout_history = []
   
        # automate task generation
        # starts = np.array([[0.0, 0.0, 0.0]])
        # goals = np.array([[0.0, 2.0, 0.0]])
        
        # genmap = Map(num_agent, map_size, num_obstacle)
        # # genmap = Map("circle.yaml")
        # agents = genmap.agents
        # num_obs = len(genmap.obstacles)
        # num_agents = genmap.num_agents
        # static_obstacles = genmap.obstacles
        # starts = np.empty((num_agents,2))
        # goals = np.empty((num_agents,2))
        robot_state_history = np.empty((self.num_agent, self.total_sim_timestep + self.N, 3))
        state_cache = []
        # for i in range(num_agents):
        #     starts[i] = [agents[i]['start'][0],agents[i]['start'][1]]
        #     goals[i] = [agents[i]['goal'][0],agents[i]['goal'][1]]

        # print('Starts are', starts)
        # print('Goals are', goals)

        # starts of all agents
        curr_state = self.initial_state
        goal_state = self.final_state

        for i in range(self.num_agent):
            for j in range(self.total_sim_timestep):
                start_time = time.time()

                state_rollouts, control_rollouts = self.perform_constant_vel_rollout(curr_state)
                rollout_history.append(state_rollouts)
                best_state_rollout, best_control_rollout = self.get_best_rollout(state_rollouts, control_rollouts, self.obs_traj[0:2, j:j+self.N, :], goal_state)
                best_rollout_history.append(best_state_rollout)
                next_state = rk_four(self.model.f, curr_state, best_control_rollout[:,0], self.dt)

                curr_state = next_state
                robot_state_history[i, j, :] = next_state
                state_cache.append(next_state)

                end_time = time.time()
                
                ##### DATA LOGGING
                computation_time.append(end_time - start_time)

            ######## DATA LOGGING
        # if(ref):
        #         # filename = "circle_nmpc.pickle"
        #     filename = "discrete_nmpc_" + "trial_" + str(trial) + "_num_agent_" + str(num_agent) + "_num_obs_" + str(num_obstacle) + "_map_size_" + str(map_size) + "_scenario_" + str(scenario) + ".pickle"
        # else:
        #         # filename = "circle_nmpc_no_ref.pickle"
        #     filename = "no_ref_" + "trial_" + str(trial) + "_num_agent_" + str(num_agent) + "_num_obs_" + str(num_obstacle) + "_map_size_" + str(map_size) + "_scenario_" + str(scenario) + ".pickle"

        # data_log["computation_time"] = np.mean(computation_time)
        # for i in range(num_agent):
        #     for j in range(self.NUMBER_OF_TIMESTEPS-1):
        #         dx = robot_state_history[i, j+1, 0] - robot_state_history[i, j, 0]
        #         dy = robot_state_history[i, j+1, 1] - robot_state_history[i, j, 1]
        #             # solution_cost += np.sqrt(pow(dx,2) + pow(dy,2))
        #     # data_log["solution_cost"] = solution_cost
        # data_log["control_std"] = np.std(control_std)
        # data_log["control_avg"] = np.mean(control_std)
        
        # with open(filename, "wb") as f:
        #     pickle.dump(data_log, f)

        print("Finished running experiment")
        draw_result = Draw_MPC_point_stabilization_v1(rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=state_cache, obs_state=self.obs_traj)
        # plot_robot_and_obstacles(
        # robot_state_history, obstacles_arr, self.rob_dia, self.NUMBER_OF_TIMESTEPS, self.SIM_TIME, filename)

    def compute_xref(self, start, goal, number_of_steps, timestep):
        dir_vec = (goal - start)
        norm = np.linalg.norm(dir_vec)
        if norm < 0.1:
            new_goal = start
        else:
            dir_vec = dir_vec / norm
            new_goal = start + dir_vec * self.v_lim * timestep * number_of_steps
        return np.linspace(start, new_goal, number_of_steps).reshape((2*number_of_steps))


    def collision_cost(self, x0, x1):
        """
        Cost of collision between two robot_state
        """
        d = np.linalg.norm(x0 - x1)
        Qc = self.cost_func_params['Qc']
        kappa = self.cost_func_params['kappa']
        cost = Qc / (1 + np.exp(kappa * (d - 2 * self.rob_dia)))
        return cost

    def total_collision_cost(self, robot, obstacles):
        total_cost = 0
        num_obstacles = self.obs_traj.shape[2]
        for i in range(10):
            for j in range(num_obstacles):
                obs = obstacles[:, i, j]
                rob = robot[:,i]
                total_cost += self.collision_cost(rob, obs)
        return total_cost