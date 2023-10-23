import casadi as ca
import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1
from mpc_base import MPC_Base
import random
from utils import *
import math

class PR_MPC(MPC_Base):    
    def run_single_mpc(self, agent_id, current_state, inter_rob_constraints):
        # casadi parameters
        opti = ca.Opti()

        opt_states = opti.variable(self.N + 1, 3)
        opt_x = opt_states[:,0]
        opt_y = opt_states[:,1]

        opt_controls = opti.variable(self.N, 2)
        v = opt_controls[:,0]
        omega = opt_controls[:, 1]
        
        opt_epsilon_o = opti.variable(self.N+1, 1)
        opt_epsilon_r = opti.variable(self.N+1, 1)

        # parameters
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3)

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for j in range(self.N):
            x_next = opt_states[j, :] + self.f(opt_states[j, :], opt_controls[j, :]).T*self.dt
            opti.subject_to(opt_states[j+1, :] == x_next)
            opti.subject_to(opti.bounded(0, opt_epsilon_o[j], ca.inf))
            opti.subject_to(opti.bounded(0, opt_epsilon_r[j], ca.inf))

        # define the cost function
        robot_cost = 0  # cost
        collision_cost = 0
        total_cost = 0
            
        Q = self.cost_func_params['Q'] # terminal cost
        R = self.cost_func_params['R'] # control cost
        P = self.cost_func_params['P'] # reference cost

        if self.ref:
            ref_traj = self.extract_trajectory_segment(current_state)
            ref_arr = np.array([[waypoint['x'], waypoint['y']] for waypoint in ref_traj[agent_id]])

            for k in range(self.N):
                robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
                                        ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) +  ca.mtimes([(opt_states[k, :2]-ref_arr[k,:].reshape(1,2)), P, (opt_states[k, :2]-ref_arr[k,:].reshape(1,2)).T]) + 100000 * opt_epsilon_o[k] + 100000 * opt_epsilon_r[k]
        else:
            for k in range(self.N):
                robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
                                        ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) + 1000000 * opt_epsilon_o[k] + 1000000 * opt_epsilon_r[k]

            # for l in range(self.num_agent):
            #     if l == agent_id:
            #         continue
            #     this_rob = self.current_state[agent_id]
            #     other_rob = self.current_state[l]
            #     distance = math.sqrt((this_rob[0] - other_rob[0])**2 + (this_rob[1] - other_rob[1])**2)
            #     if distance < 0.5:
            #         collision_cost += distance

        total_cost = robot_cost + 100 * collision_cost
        opti.minimize(total_cost)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(-12.0, opt_x, 12.0))
        opti.subject_to(opti.bounded(-12.0, opt_y, 12.0))
        opti.subject_to(opti.bounded(-self.v_lim, v, self.v_lim))
        opti.subject_to(opti.bounded(-self.omega_lim, omega, self.omega_lim))        

        # static obstacle constraint
        # for obs in self.static_obs:
        #     obs_x = obs[0]
        #     obs_y = obs[1]
        #     obs_dia = obs[2]
        #     for l in range(self.N+1):
        #         rob_obs_constraints_ = ca.sqrt((opt_states[l, 0]-obs_x)**2+(opt_states[l, 1]-obs_y)**2)-self.rob_dia/2.0 - obs_dia/2.0 - self.safety_margin/2.0 + opt_epsilon_o[l]
        #         opti.subject_to(self.opti.bounded(0.0, rob_obs_constraints_, ca.inf))
        if self.map is not None:
            for obs in self.obs["static"]:
                obs_x = obs[0]
                obs_y = obs[1]
                obs_dia = obs[2]
                for l in range(self.N+1):
                    rob_obs_constraints_ = ca.sqrt((opt_states[l, 0]-obs_x)**2+(opt_states[l, 1]-obs_y)**2) - obs_dia + opt_epsilon_o[l]
                    opti.subject_to(opti.bounded(0.0, rob_obs_constraints_, ca.inf))

        num_rob_constraints = 0.0
        if inter_rob_constraints:
            for other_rob in inter_rob_constraints:         
                other_rob_traj = self.prediction_cache[other_rob]

                rob_rob_constraints_ = ca.sqrt((opt_states[:,0]-other_rob_traj[:,0])**2 + (opt_states[:,1]-other_rob_traj[:,1])**2) - self.rob_dia - self.safety_margin + opt_epsilon_r

                opti.subject_to(opti.bounded(0.0, rob_rob_constraints_, ca.inf))
                num_rob_constraints += self.N
        self.c_avg.append(num_rob_constraints)

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
                            'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
                            'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}

        opti.solver('ipopt', opts_setting)
        opti.set_value(opt_xs, self.final_state[agent_id])
            
        # start MPC
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)
                    
        # solve the optimization problem
        t_ = time.time()
        sol = opti.solve()
        solve_time = time.time() - t_
        # print("Agent " + str(agent_id) + " Solve Time: " + str(solve_time))

        # obtain the control input
        u_res = sol.value(opt_controls)
        next_states_pred = sol.value(opt_states)
        eps_o = sol.value(opt_epsilon_o)
        eps_r = sol.value(opt_epsilon_r)

        self.prev_states[agent_id] = next_states_pred
        self.prev_controls[agent_id] = u_res
        self.prev_epsilon_o[agent_id] = eps_o 
        self.prev_epsilon_r[agent_id] = eps_r
  
        return u_res, next_states_pred
    
    def assign_random_priorities(self):
            agent_list = list(range(self.num_agent))
            random.shuffle(agent_list)

            return agent_list
    
    def get_constraints_for_robot(self, agent_id, agent_list):
        constraint_map = {i: [] for i in range(self.num_agent)}
        # agent list goes from highest priority to lowest
        agent_priority = next((i for i, x in enumerate(agent_list) if x == agent_id), -1)
        # Go through all higher priority agents
        for i in range(agent_priority):
            constraint_map[agent_id].append(i)
        
        return constraint_map 

    def simulate(self):
        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N)) for agent_id in range(self.num_agent)}
        
        # agent_list = self.assign_random_priorities()
        avg_comp_time = []
        agent_list = list(range(self.num_agent))
        while(not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
            time_1 = time.time()
            print(self.num_timestep)

            # initial MPC solve
            for agent_id in agent_list:
                current_state = np.array(self.current_state[agent_id])
                
                constraints = []
                if(agent_id != 0):
                    for lower_priority_robot in range(agent_id):
                        constraints.append(lower_priority_robot)

                u, next_states_pred = self.run_single_mpc(agent_id, current_state, constraints)
                next_state, u0, next_states = self.shift_movement(current_state, u, next_states_pred, self.f_np)

                self.prediction_cache[agent_id] = next_states_pred
                self.control_cache[agent_id] = u
                self.current_state[agent_id] = next_state
                self.state_cache[agent_id].append(next_state)
            
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)
            self.num_timestep = self.num_timestep + 1

        avg_comp_time = 0.0
        if self.is_solution_valid(self.state_cache):
            print("Executed solution is GOOD!")
            avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            self.max_comp_time = max(self.avg_comp_time)
            self.c_avg = (sum(self.c_avg) / len(self.c_avg)) / self.num_agent
            self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
            self.avg_rob_dist = get_avg_rob_dist(self.state_cache)
            self.success = True
        else:
            self.success = False
        
        run_description = "PR-MPC_" + self.scenario 
        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.map, self.initial_state, self.final_state, avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()

        # draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)