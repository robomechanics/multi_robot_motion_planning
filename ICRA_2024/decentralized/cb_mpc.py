import casadi as ca
import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1
from mpc_base import MPC_Base
import multiprocessing as mp
import math
from node import Node
from utils import *
import matplotlib.pyplot as plt

class CB_MPC(MPC_Base):

    def find_collisions(self, node):
        agent_predictions = node.state_solution
        conflict_list = []
        for i in range(self.num_agent):
            for j in range(i+1, self.num_agent):
                if i == j:
                    continue
                agent_1_traj = agent_predictions[i]
                agent_2_traj = agent_predictions[j]

                for index, (wp_1, wp_2) in enumerate(zip(agent_1_traj, agent_2_traj)):
                    distance = math.sqrt((wp_1[0] - wp_2[0])**2 + (wp_1[1] - wp_2[1])**2)
                    if distance < self.rob_dia:
                        # print("Collision detected between " + str(i) + " and " + str(j) + " at index " + str(index))
                        conflict_list.append((i, j, index))
                        break
        
        return conflict_list
    
    def get_agent_cost(self, agent_id):
        path_length = 0
        cost_to_go = 0
        
        final_state = self.final_state[agent_id]
        current_traj = self.state_cache[agent_id]
        for i in range(len(current_traj) - 1):
            x1, y1, _ = current_traj[i]
            x2, y2, _ = current_traj[i + 1]
                
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            path_length += distance

        distance_to_goal = math.sqrt((current_traj[-1][0] - final_state[0])**2 + (current_traj[-1][1] - final_state[1])**2)
        cost_to_go += distance_to_goal
            
        return path_length + cost_to_go

    def run_single_mpc(self, agent_id, current_state, inter_rob_constraints):
        # casadi parameters
        opti = ca.Opti()

        opt_states = opti.variable(self.N + 1, 3)
        opt_x = opt_states[:,0]
        opt_y = opt_states[:,1]

        opt_controls = opti.variable(self.N, 2)
        v = opt_controls[:,0]
        omega = opt_controls[:,1]
        
        opt_epsilon_o = opti.variable(self.N+1, 1)
        opt_epsilon_r = opti.variable(self.N+1, 1)
        
        # parameters
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3)
        # self.opt_epsilon_r.append(self.opti.variable(self.N+1, 1))

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
            
        Q = self.cost_func_params['Q']
        R = self.cost_func_params['R']
        P = self.cost_func_params['P']
     
        for k in range(self.N):
            if self.ref:
                ref_seg = self.extract_trajectory_segment(current_state)
                ref = np.array([[d['x'], d['y']] for d in ref_seg[agent_id]])
                curr_ref = ref[k,:].reshape(1,2)
                robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
                                        ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) + ca.mtimes([(opt_states[k, :2]-curr_ref), P, (opt_states[k, :2]-curr_ref).T]) + 100000 * opt_epsilon_o[k] + 100000 * opt_epsilon_r[k]
            else: 
                robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
                                        ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) + 100000 * opt_epsilon_o[k] + 100000 * opt_epsilon_r[k]

            
        # for l in range(self.num_agent):
        #     if l == agent_id:
        #         continue
        #     this_rob = self.current_state[agent_id]
        #     other_rob = self.current_state[l]
        #     distance = math.sqrt((this_rob[0] - other_rob[0])**2 + (this_rob[1] - other_rob[1])**2)
        #     if distance < 0.5:
        #         collision_cost += distance

        total_cost = robot_cost + 1000 * collision_cost
        opti.minimize(total_cost)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(-12.0, opt_x, 12.0))
        opti.subject_to(opti.bounded(-12.0, opt_y, 12.0))
        opti.subject_to(opti.bounded(-self.v_lim, v, self.v_lim))
        opti.subject_to(opti.bounded(-self.omega_lim, omega, self.omega_lim))        

        # static obstacle constraint
        # if self.map is not None:
        #     obstacles = get_obstacle_coordinates(self.map, current_state)
        #     for obs in obstacles:
        #         obs_x = obs[0]
        #         obs_y = obs[1]
        #         for l in range(self.N+1):
        #             rob_obs_constraints_ = ca.sqrt((opt_states[l, 0]-obs_x)**2+(opt_states[l, 1]-obs_y)**2)-1.1 + opt_epsilon_o[l]
        #             opti.subject_to(opti.bounded(0.0, rob_obs_constraints_, ca.inf))
        
        # Add inter robot constraints
        if inter_rob_constraints:
            for constraint in inter_rob_constraints:
                other_rob = constraint[0]
                collision_index = constraint[1]
                if agent_id == other_rob:
                    continue
                        
                other_rob = self.prediction_cache[other_rob]
                
                rob_rob_constraints_ = ca.sqrt((opt_states[collision_index:-1,0]-other_rob[collision_index:-1,0])**2 + (opt_states[collision_index:-1,1]-other_rob[collision_index:-1,1])**2) - self.rob_dia - self.safety_margin + opt_epsilon_r[collision_index:-1]

                opti.subject_to(opti.bounded(0.0, rob_rob_constraints_, ca.inf))

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
                            'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
                            'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}

        opti.solver('ipopt', opts_setting)
        opti.set_value(opt_xs, self.final_state[agent_id])
        
        # start MPC
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)

        opti.set_initial(opt_states, self.prev_states[agent_id])
        opti.set_initial(opt_controls, self.prev_controls[agent_id])
        opti.set_initial(opt_epsilon_o, self.prev_epsilon_o[agent_id])
        opti.set_initial(opt_epsilon_r, self.prev_epsilon_r[agent_id])

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
    
    def simulate(self):
        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N)) for agent_id in range(self.num_agent)}
        
        while(not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
            time_1 = time.time()
            # print(self.num_timestep)
            # initial MPC solve
            pool = mp.Pool()
    
            # Apply MPC solve to each agent in parallel
            results = pool.starmap(self.run_single_mpc, [(agent_id, np.array(self.current_state[agent_id]), []) for agent_id in range(self.num_agent)])
    
            pool.close()
            pool.join()
    
            # Process the results and update the current state
            for agent_id, result in enumerate(results):
                u, next_states_pred = result
                current_state = np.array(self.current_state[agent_id])
                # next_state, u0, next_states = self.shift_movement(current_state, u, next_states_pred, self.f_np)

                self.prediction_cache[agent_id] = next_states_pred
                self.control_cache[agent_id] = u
                self.current_state[agent_id] = current_state
                self.state_cache[agent_id].append(current_state)

            self.num_timestep += 1

            # Handling conflict resolution logic
            conflict_tree = []
            root = Node()
            root.update_solution(self.control_cache, self.prediction_cache)
            root.update_cost(self.final_state)
            conflict_tree.append(root)

            # loop until conflict tree is empty
            num_rob_constraints = 0.0
            while conflict_tree:
                print(len(conflict_tree))
                p = get_best_node(conflict_tree)
                # conflict_tree.remove(p)
                conflict_list = self.find_collisions(p)

                # apply controls if there are no collisions and break out of the conflict resolution loop
                if not conflict_list:
                    for agent_id in range(self.num_agent):
                        current_state = np.array(self.current_state[agent_id])
                        # u = self.control_cache[agent_id]
                        u = p.control_solution[agent_id]
                        # next_states_pred = self.prediction_cache[agent_id]
                        next_states_pred = p.state_solution[agent_id]

                        next_state, u0, next_states = self.shift_movement(current_state, u, next_states_pred, self.f_np)
                        self.current_state[agent_id] = next_state
                        self.state_cache[agent_id].append(next_state)
                    self.c_avg.append(num_rob_constraints)
                    break
                else:
                    # get first conflict (rob i, rob j, collision index)
                    conflict = conflict_list[0]
                    conflict_list.pop(0)
                    
                    for i in range(2):
                        new_node = Node(constraints=p.constraints, state_solution=p.state_solution, control_solution=p.control_solution)

                        agent_id = 0
                        new_constraint = (conflict[0], conflict[2])
                        agent_id = conflict[1]

                        if(i == 1):
                            new_constraint = (conflict[1], conflict[2])
                            agent_id = conflict[0]

                        new_node.add_constraint(new_constraint)
                        # print(new_node.constraints)
                        u, next_states_pred = self.run_single_mpc(agent_id, self.current_state[agent_id], new_node.constraints)

                        num_rob_constraints += self.N - conflict[2]

                        new_node.update_solution(u, next_states_pred, agent_id)
                        new_node.update_cost(self.final_state)

                        conflict_tree.append(new_node)
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)
            
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
        
        run_description = "CB-MPC_" + self.scenario 
        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.map, self.initial_state, self.final_state, avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # Draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)