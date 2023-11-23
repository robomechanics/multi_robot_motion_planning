import casadi as ca
import numpy as np
import time
# from draw import Draw_MPC_point_stabilization_v1
from mpc_base import MPC_Base
import multiprocessing as mp
from utils import *
import scipy.special as sp

class Branch_MPC(MPC_Base):    
    def run_single_mpc(self, agent_id, current_state, inter_rob_constraints):
        # casadi parameters
        opti = ca.Opti()

        opt_states = [opti.variable(self.N+1, 3) for k in range(self.num_modes)]            
        opt_x = ca.horzcat(*[opt_states[k][:, 0] for k in range(self.num_modes)])
        opt_y = ca.horzcat(*[opt_states[k][:, 1] for k in range(self.num_modes)])

        opt_controls = [opti.variable(self.N, 2) for k in range(self.num_modes)]
        v = ca.horzcat(*[opt_controls[k][:, 0] for k in range(self.num_modes)])
        omega = ca.horzcat(*[opt_controls[k][:, 1] for k in range(self.num_modes)])
        
        opt_epsilon_o = [opti.variable(self.N+1, 1) for k in range(self.num_modes)]
        opt_epsilon_r = [opti.variable(self.N+1, 1) for k in range(self.num_modes)]

        self.prev_states = {agent_id: [np.zeros((self.N+1, 3)) for _ in range(self.num_modes)] for agent_id in range(self.num_agent)}
        self.prev_controls = {agent_id: [np.zeros((self.N, 2)) for _ in range(self.num_modes)] for agent_id in range(self.num_agent)}
        self.prev_epsilon_o = {agent_id: [np.zeros((self.N+1, 1)) for _ in range(self.num_modes)] for agent_id in range(self.num_agent)}
        self.prev_epsilon_r = {agent_id: [np.zeros((self.N+1, 1)) for _ in range(self.num_modes)] for agent_id in range(self.num_agent)}

        # parameters
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3)

        # init_condition
        for m in range(self.num_modes):
            opti.subject_to(opt_states[m][0,:] == opt_x0.T)

        for n in range(self.N):
            for m in range(self.num_modes):
                ### Needs to fix this
                x_next = opt_states[m][n,:] + self.f(opt_states[m][n,:], opt_controls[m][n,:]).T * self.dt
                opti.subject_to(opt_states[m][n+1,:] == x_next)
                opti.subject_to(opti.bounded(0, opt_epsilon_o[m][n,:], ca.inf))
                opti.subject_to(opti.bounded(0, opt_epsilon_r[m][n,:], ca.inf))

        for n in range(self.robust_horizon):
            for i in range(self.num_modes):
                for j in range(i + 1, self.num_modes):
                    opti.subject_to(opt_states[i][n, :] == opt_states[j][n, :])
                    opti.subject_to(opt_controls[i][n, :] == opt_controls[j][n, :])

        # define the cost function
        robot_cost = 0  # cost
        collision_cost = 0
        total_cost = 0
            
        Q = self.cost_func_params['Q']
        R = self.cost_func_params['R']
        P = self.cost_func_params['P']

        # ref_seg = self.extract_trajectory_segment(current_state)
        # ref = np.array([[d['x'], d['y']] for d in ref_seg[agent_id]])

        mode_prob = self.mode_prob[self.num_timestep]
        for k in range(self.N):
            for m in range(self.num_modes):
            # if self.ref:
            #     curr_ref = ref[k,:].reshape(1,2)
            #     robot_cost = robot_cost + ca.mtimes([(opt_states[k, :]-opt_xs.T), Q, (opt_states[k, :]-opt_xs.T).T]
            #                             ) + ca.mtimes([opt_controls[k, :], R, opt_controls[k, :].T]) + ca.mtimes([(opt_states[k, :2]-curr_ref), P, (opt_states[k, :2]-curr_ref).T]) + 100000 * opt_epsilon_o[k]
            # else: 
                mode_weight = mode_prob[m]
                robot_cost = robot_cost + mode_weight * (ca.mtimes([(opt_states[m][k, :]-opt_xs.T), Q, (opt_states[m][k, :]-opt_xs.T).T]
                                            ) + ca.mtimes([opt_controls[m][k, :], R, opt_controls[m][k, :].T])) +  100000 * opt_epsilon_r[m][k]
            
        total_cost = robot_cost + collision_cost
        opti.minimize(total_cost)
 
        # boundrary and control conditions
        opti.subject_to(opti.bounded(-10.0, opt_x, 10.0))
        opti.subject_to(opti.bounded(-10.0, opt_y, 10.0))
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

        for agent_prediction in gmm_predictions:
            for mode, prediction in agent_prediction.items():
                means = prediction['means']
                covariances = prediction['covariances']
                
                for timestep, (mean, covariance) in enumerate(zip(means, covariances)):
                    pi = ca.vec(opt_states[mode][timestep,:])
                    pj = ca.vec(np.array(mean[:2]))

                    rob_rob_constraint = ca.sqrt((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2) - 2* self.rob_dia + opt_epsilon_r[mode][timestep]\
                                        -sp.erf(1-2*self.delta)*ca.sqrt(2*(pi-pj).T@covariance[:2,:2]@(pi-pj))
                    opti.subject_to(rob_rob_constraint >= 0)

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
                            'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
                            'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}
    
        opti.solver('ipopt', opts_setting)
        opti.set_value(opt_xs, self.final_state[agent_id])
            
        # start MPC
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)

        # set optimizing target withe init guess
        for mode in range(self.num_modes):
            # Set the initial value for each mode of each agent
            opti.set_initial(opt_states[mode], self.prev_states[agent_id][mode])
            opti.set_initial(opt_controls[mode], self.prev_controls[agent_id][mode])  # (N, 2)
            opti.set_initial(opt_epsilon_o[mode], self.prev_epsilon_o[agent_id][mode])  # (N+1, 3)
            opti.set_initial(opt_epsilon_r[mode], self.prev_epsilon_r[agent_id][mode])
                    
        # solve the optimization problem
        t_ = time.time()
        sol = opti.solve()
        solve_time = time.time() - t_
        print("Agent " + str(agent_id) + " Solve Time: " + str(solve_time))

        # obtain the control input
        u_res = [opti.value(opt_controls[k]) for k in range(self.num_modes)]
        next_states_pred = [opti.value(opt_states[k]) for k in range(self.num_modes)]

        eps_o = [opti.value(opt_epsilon_o[k]) for k in range(self.num_modes)]
        
        self.prev_states[agent_id] = next_states_pred
        self.prev_controls[agent_id] = u_res
        self.prev_epsilon_o[agent_id] = eps_o 
  
        return u_res, next_states_pred
    
    def simulate(self):
        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1, self.num_modes)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N, self.num_modes)) for agent_id in range(self.num_agent)}
        
        self.setup_visualization()

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

            self.plot_gmm_means_and_state(self.current_state[0], self.prediction_cache[0], gmm_predictions[0])
            
            # Process the results and update the current state
            for agent_id, result in enumerate(results):
                u, next_states_pred = result
                current_state = np.array(self.current_state[agent_id])
                next_state, u0, next_states = self.shift_movement(current_state, u[0], next_states_pred[0], self.f_np)

                self.prediction_cache[agent_id] = next_states_pred
                self.control_cache[agent_id] = u
                self.current_state[agent_id] = next_state
                self.state_cache[agent_id].append(next_state)

            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)

        if self.is_solution_valid(self.state_cache):
            print("Executed solution is GOOD!")
            self.max_comp_time = max(self.avg_comp_time)
            self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
            self.success = True
        else:
            self.success = False
        
        run_description = "MPC_" + self.scenario 

        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # draw function
        draw_result = Draw_MPC_point_stabilization_v1(
            rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)
        