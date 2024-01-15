import casadi as ca
import numpy as np
import time
# from draw import Draw_MPC_point_stabilization_v1
from mpc_base import MPC_Base
import multiprocessing as mp
from utils import *
import scipy.special as sp

class MPC(MPC_Base):    
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
            opti.subject_to(opti.bounded(0, opt_epsilon_o[j], 0.001))
            opti.subject_to(opti.bounded(0, opt_epsilon_r[j], 0.001))

        # define the cost function
        robot_cost = 0  # cost
        collision_cost = 0
        total_cost = 0
            
        Q = self.cost_func_params['Q']
        R = self.cost_func_params['R']
        P = self.cost_func_params['P']

        # ref_seg = self.extract_trajectory_segment(current_state)
        # ref = np.array([[d['x'], d['y']] for d in ref_seg[agent_id]])

        for k in range(self.N):
            # if self.ref:
            #     curr_ref = ref[k,:].reshape(1,2)
            #     robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
            #                             ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) + ca.mtimes([(opt_states[k, :2]-curr_ref), P, (opt_states[k, :2]-curr_ref).T]) + 100000 * opt_epsilon_o[k]
            # else: 
            robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
                                        ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) +  100000 * opt_epsilon_r[k]
            
        # robot_cost = robot_cost + ca.mtimes([(opt_states[self.N, :]-opt_xs.T), P, (opt_states[self.N, :]-opt_xs.T).T])

        total_cost = robot_cost + collision_cost
        opti.minimize(total_cost)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(-10.0, opt_x, 10.0))
        opti.subject_to(opti.bounded(-2.0, opt_y, 2.0))
        opti.subject_to(opti.bounded(-self.v_lim, v, self.v_lim))
        opti.subject_to(opti.bounded(-self.omega_lim, omega, self.omega_lim))        

        # static obstacle constraint
        for obs in self.static_obs:
            obs_x = obs[0]
            obs_y = obs[1]
            obs_dia = obs[2]
     
            for l in range(self.N):
                rob_obs_constraints_ = ca.sqrt((opt_states[l, 0]-obs_x)**2+(opt_states[l, 1]-obs_y)**2)-obs_dia/2 - self.rob_dia/2 - self.safety_margin + opt_epsilon_o[l]
                opti.subject_to(rob_obs_constraints_ >= 0)

        ##### Get chance constraints from the given GMM prediction
        ## aij = (pi - pj) / ||pi - pj|| and bij = ri + rj 
        ## aij^T(pi - pj) - bij >= erf^-1(1 - 2delta)sqrt(2*aij^T(sigma_i + sigma_j)aij)
        current_uncontrolled_state = self.uncontrolled_traj[self.num_timestep]
        gmm_predictions = self.uncontrolled_agent.get_gmm_predictions_from_current(current_uncontrolled_state)

        for k, agent_prediction in enumerate(gmm_predictions):
            for j, prediction in agent_prediction.items():
                covariances = prediction['covariances']
                for t in range(1,self.N):
                    if self.linearized_ca:
                        tv_pos = ca.DM(prediction['means'][t-1][:2])
                        if type(self.prev_states[agent_id])==type([]):
                            ref_pos = self.prev_states[agent_id][j][t,:2].T
                        else:
                            ref_pos = ca.DM(current_state)[:2]
                        
                        rob_proj = tv_pos+2*self.rob_dia*(ref_pos-tv_pos)/ca.norm_2(ref_pos-tv_pos)
                        nom_dist = (rob_proj-tv_pos).T@(opt_states[t,:2].T-rob_proj)

                        rv_dist  = sp.erfinv(1-2*self.delta)*(rob_proj-tv_pos).T@covariances[t][:2,:2]*2

                        opti.subject_to(rv_dist@rv_dist.T<=(opt_epsilon_r[t-1]+nom_dist)**2)
                        opti.subject_to(nom_dist>=-opt_epsilon_r[t-1])

                    else:
                        tv_pos = ca.DM(prediction['means'][t-1][:2])
                        rob_pos = ca.vec(opt_states[t,:])

                        rob_rob_constraint = ca.sqrt((rob_pos[0]-tv_pos[0])**2 + (rob_pos[1]-tv_pos[1])**2) - 2* self.rob_dia + opt_epsilon_r[t]
                        opti.subject_to(rob_rob_constraint >= 0)

        # for agent_prediction in gmm_predictions:
        #     for mode, prediction in agent_prediction.items():
        #         means = prediction['means']
        #         covariances = prediction['covariances']
                
        #         for timestep, (mean, covariance) in enumerate(zip(means, covariances)):
        #             pi = opt_states[timestep,:]
        #             pj = np.array(mean[:2])

        #             rob_rob_constraint = ca.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2) - 2* self.rob_dia + opt_epsilon_r[timestep]
        #             # aij = (pi[:,:2]-pj) / ca.norm_2(pi[:,:2]-pj)
        #             # bij = self.rob_dia*2
        #             # rob_rob_constraint = aij@(pi[:,:2]-pj).T - bij
        #             opti.subject_to(rob_rob_constraint >= 0)

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
                            'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
                            'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}
    
        opti.solver('ipopt', opts_setting)
        opti.set_value(opt_xs, self.final_state[agent_id])
            
        # start MPC
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)

        # set optimizing target withe init guess
        opti.set_initial(opt_controls, self.prev_controls[agent_id])  # (N, 2)
        opti.set_initial(opt_states, self.prev_states[agent_id])  # (N+1, 3)
        opti.set_initial(opt_epsilon_o, self.prev_epsilon_o[agent_id])

        u_res = None
        next_states_pred = None

        try:       
            # solve the optimization problem
            t_ = time.time()
            sol = opti.solve()
            solve_time = time.time() - t_
            print("Agent " + str(agent_id) + " Solve Time: " + str(solve_time))

            # obtain the control input
            u_res = sol.value(opt_controls)
            next_states_pred = sol.value(opt_states)
            eps_o = sol.value(opt_epsilon_o)

            self.prev_states[agent_id] = next_states_pred
            self.prev_controls[agent_id] = u_res
            self.prev_epsilon_o[agent_id] = eps_o 

        except RuntimeError as e:
            print("Infeasible solve")
    
        return u_res, next_states_pred
    
    def simulate(self):
        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N)) for agent_id in range(self.num_agent)}
        
        # self.setup_visualization()

        # parallelized implementation
        while (not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
            time_1 = time.time()
            print(self.num_timestep)

            # Create a multiprocessing pool
            pool = mp.Pool()
    
            # Apply MPC solve to each agent in parallel
            results = pool.starmap(self.run_single_mpc, [(agent_id, np.array(self.current_state[agent_id]), []) for agent_id in range(self.num_agent)])
    
            pool.close()
            pool.join()

            current_uncontrolled_state = self.uncontrolled_traj[self.num_timestep]
            gmm_predictions = self.uncontrolled_agent.get_gmm_predictions_from_current(current_uncontrolled_state)

            # self.plot_gmm_means_and_state(self.current_state[0], self.prediction_cache[0], gmm_predictions[0])
            
            # Process the results and update the current state
            for agent_id, result in enumerate(results):
                u, next_states_pred = result
                if u is None:
                    self.infeasible_count += 1
                    self.infeasible = True
                    break
                else:
                    current_state = np.array(self.current_state[agent_id])
                    next_state, u0, next_states = self.shift_movement(current_state, u, next_states_pred, self.f_np)

                    self.prediction_cache[agent_id] = next_states_pred.T
                    self.control_cache[agent_id] = u
                    self.current_state[agent_id] = next_state
                    self.state_cache[agent_id].append(next_state)

            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)

        print(self.infeasible_count)
        print("##############")
        if self.is_solution_valid(self.state_cache):
            print("Executed solution is GOOD!")
            self.max_comp_time = max(self.avg_comp_time)
            self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
            self.success = True
        else:
            self.success = False
        
        run_description = self.scenario 

        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached, self.infeasible_count, self.num_timestep)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)
        