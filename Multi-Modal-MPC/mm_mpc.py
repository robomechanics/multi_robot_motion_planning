import casadi as ca
import numpy as np
import time
from mpc_base import MPC_Base
import multiprocessing as mp
from utils import *
import scipy.special as sp
from scipy.stats import multivariate_normal
import pdb

class MM_MPC(MPC_Base):            
    def _get_robot_ATV_dynamics(self, current_state, x_lin=None, u_lin=None):
        """
        Constructs system matrices such for the robot,
        X_t= A_pred@current_state + B_pred@U_t + C_pred + E_pred@W_t
        where
        X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
        U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence)
        W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T,  (EV process noise sequence)
        x_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 
        n_x, n_u = self.model.n_x, self.model.n_u
        self.model.make_dynamics_jac(self.dt) 

        A=[ca.MX(n_x, n_x) for _ in range(self.N+1)]
        B=[ca.MX(n_x, n_u) for _ in range(self.N+1)]
        C=[ca.MX(n_x, 1) for _ in range(self.N+1)]

        A_block=ca.MX(n_x*(self.N+1), n_x)
        B_block=ca.MX(n_x*(self.N+1), n_u*self.N)
        C_block=ca.MX(n_x*(self.N+1), 1)
        E_block=ca.MX(n_x*(self.N+1), n_x*self.N)

        A_block[0:n_x, 0:n_x]=ca.DM.eye(n_x)

        E=0.001*ca.DM.eye(n_x)  # Coefficients of Linearization error

        if x_lin is None:
            x_lin=ca.DM(self.N+1,3)
            u_lin=ca.horzcat(.2*ca.DM.ones(self.N,1), ca.DM(self.N,1))
            x_lin[0,:]=current_state.reshape((1,-1))
            for t in range(self.N):
                x_lin[t+1,:]=self.model.fCd(x_lin[t,:], u_lin[t,:])

        for t in range(self.N):
            A[t]=self.model.fAd(x_lin[t,:], u_lin[t,:])
            B[t]=self.model.fBd(x_lin[t,:], u_lin[t,:])
            C[t]=x_lin[t+1,:].T-A[t]@x_lin[t,:].T-B[t]@u_lin[t,:].T

            A_block[(t+1)*n_x:(t+2)*n_x, :]=A[t]@A_block[t*n_x:(t+1)*n_x, :]

            B_block[(t+1)*n_x:(t+2)*n_x,:]=A[t]@B_block[t*n_x:(t+1)*n_x,:]
            B_block[(t+1)*n_x:(t+2)*n_x,t*n_u:(t+1)*n_u]=B[t]
            
            C_block[(t+1)*n_x:(t+2)*n_x,:]=A[t]@C_block[t*n_x:(t+1)*n_x,:] + C[t]
            E_block[(t+1)*n_x:(t+2)*n_x,:]=A[t]@E_block[t*n_x:(t+1)*n_x,:]
            E_block[(t+1)*n_x:(t+2)*n_x,t*n_x:(t+1)*n_x]=E

        return A_block, B_block, C_block, E_block
    
    def _get_obs_ATV_dynamics(self, mean_inputs, covar_inputs, mean_traj):
        """
        Parameters:
        mean_inputs  : mean speed and omega along horizon
        covar_inputs : speed and omega covariances along horizon
        mean_traj    : mean trajectory of obstacle along horizon

        Returns:
        System matrices: T_obs, c_obs, E_obs

        The matrices provide stacked trajectory predictions:
        O_t=T_obs@o_{t|t}+c_obs+E_obs@N_t
        where
        O_t=[o_{t|t}, o_{t+1|t},...,o_{t+N|t}].T, (obstacle state predictions)
        N_t=[n_{t|t}, n_{t+1|t},...,n_{t+N-1|t}].T,  (obstacle process noise sequence)
        o_{i|t}= state prediction of obstacle at time step i, given current time t

        Diff_drive robot dynamics:

        o_{t+1}=[[1, 0], [0,1]]@o_t+ dt*[cos\theta; sin\theta]v+n_t

        o_{t+2}=o_{t|t}+dt*[cos\theta_t; sin\theta_t]v_t+ n_t + dt*[cos\theta_(t+1); sin\theta_(t+1)]v_(t+1)  + n_(t+1)

        o_{t+N}= o_{t|t} + dt*[cos\theta_t; sin\theta_t]v_t+ .... + dt*[cos\theta_(t+N-1); sin\theta_(t+N-1)]v_(t+N-1) + n_t +... n_{t+N-1}
        """ 
        T_obs=ca.DM(2*(self.N+1), 2)
        c_obs=ca.DM(2*(self.N+1), 1) 
        E_obs=ca.DM(2*(self.N+1),self.N)

        for t in range(self.N+1):
            T_obs[t*2:(t+1)*2,:]=ca.DM.eye(2)
            if t>0:
                
                theta = mean_traj[min(t, self.N-1)][2]
                B=self.dt*ca.DM([np.cos(theta), np.sin(theta)])

                v=mean_inputs[min(t,self.N-1)][0]
                c_obs[t*2:(t+1)*2,:]=c_obs[(t-1)*2:t*2,:]+B@v

                E=B@covar_inputs[t-1][0,0]**(0.5)
                E_obs[t*2:(t+1)*2,:]=E_obs[(t-1)*2:t*2,:]    
                E_obs[t*2:(t+1)*2,(t-1)*1:t*1]=E

        return T_obs, c_obs, E_obs

    def run_single_mpc(self, agent_id, current_state, inter_rob_constraints, linearized_ca = False):
        # casadi parameters
        opti = ca.Opti('conic')

        # current_state_obs_vector = [self.uncontrolled_fleet_data[obs]['executed_traj'][self.num_timestep] for obs in range(len(self.uncontrolled_fleet_data))]
        # gmm_predictions_vector = self.uncontrolled_fleet.get_gmm_predictions_from_current(current_state_obs_vector)
        # noise_chars = self.uncontrolled_fleet.get_gmm_predictions()
        # mode_prob = self.uncontrolled_fleet_data[0]['mode_probabilities'][self.num_timestep]

        filtered_predictions_vector = []
        filtered_noise_chars = []

        if self.mle:
            for agent_idx, (agent_predictions, agent_noise) in enumerate(zip(self.gmm_predictions, self.noise_chars)):
                # Assuming mode_prob[agent_idx] gives us the most likely mode index for the agent
                most_likely_mode = self.mode_prob[agent_idx]
                
                # Ensure most_likely_mode is an integer, as expected
                most_likely_mode = int(most_likely_mode)
                
                # Filter the predictions and noise characteristics to keep only those related to the most likely mode
                most_likely_mode_prediction = {most_likely_mode: agent_predictions[most_likely_mode]}
                most_likely_mode_noise = {most_likely_mode: agent_noise[most_likely_mode]}
                
                # Append these filtered predictions and noise characteristics to the new lists
                filtered_predictions_vector.append(most_likely_mode_prediction)
                filtered_noise_chars.append(most_likely_mode_noise)
        else:
            filtered_predictions_vector = self.gmm_predictions
            filtered_noise_chars = self.noise_chars

        ####
        # EV feedforward + TV state feedback policies from https://arxiv.org/abs/2109.09792
        # U_stack[j] = opt_controls + sum_i=1^{N_obs} K_stack[i][j]@(O_stack[i][j] - E[O_stack[i][j]])
        #### 
        # nominal feedforward controls
        rob_horizon_u = opti.variable(self.robust_horizon, 2)
        
        opt_controls = [ca.vertcat(rob_horizon_u, opti.variable(self.N-self.robust_horizon,2)) for _ in range(self.num_modes)]
        opt_epsilon_r = [opti.variable(self.N, 1) for _ in range(self.num_modes)]

        A_rob, B_rob, C_rob, E_rob = [], [], [], []
        opt_states, opt_x, opt_y, v, omega = [], [], [], [], []
        for j in range(self.num_modes):
            # if np.linalg.norm(self.prev_states[agent_id][j])>1e-2:
            if type(self.prev_states[agent_id])== type([]):
                A, B, C, E = self._get_robot_ATV_dynamics(current_state, self.prev_states[agent_id][j], self.prev_controls[agent_id][j])       
            else:
                A, B, C, E = self._get_robot_ATV_dynamics(current_state)
                
            A_rob.append(A); B_rob.append(B); C_rob.append(C); E_rob.append(E)

            # nominal state predictions
            opt_states.append(ca.vec(A@ca.DM(current_state)+B@ca.vec(opt_controls[j].T)+C).reshape((-1,self.N+1)).T)

            opt_x.append(opt_states[-1][:,0])
            opt_y.append(opt_states[-1][:,1])
            v.append(opt_controls[j][:,0])
            omega.append(opt_controls[j][:,1])

            # opti.subject_to(opti.bounded(0, opt_epsilon_r[j][:], 0.001))

        # opt_epsilon_o = opti.variable(self.N+1, 1)
        
        # parameters
        # opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3)
        # self.opt_epsilon_r.append(self.opti.variable(self.N+1, 1))

        # define the cost function
        robot_cost = 0  # cost
        collision_cost = 0
        total_cost = 0
            
        Q = self.cost_func_params['Q']
        R = self.cost_func_params['R']
        P = self.cost_func_params['P']
        
        ref = np.array(self.extract_trajectory_segment(self.current_state[0])[0])

        for j in range(self.num_modes):
            for k in range(self.N):
                mode_weight = 1
                if not self.mle:
                    mode_weight = self.mode_prob[j]
                # if k > self.robust_horizon:
                # robot_cost = robot_cost + mode_weight*(ca.mtimes([(opt_states[j][k, :]-opt_xs.T), Q, (opt_states[j][k, :]-opt_xs.T).T] 
                #             )+ ca.mtimes([opt_controls[j][k, :], R, opt_controls[j][k, :].T]) + 100000 * opt_epsilon_r[j][k]) #+ 100000 * opt_epsilon_o[k]
                robot_cost = robot_cost + mode_weight*(ca.mtimes([(opt_states[j][k, :]-opt_xs.T), Q, (opt_states[j][k, :]-opt_xs.T).T] 
                    )+ ca.mtimes([opt_controls[j][k, :], R, opt_controls[j][k, :].T])) #+ 100000 * opt_epsilon_r[j][k]) 
                # else:
                #     new_ref = ref[k, :].reshape((3,1))
                #     robot_cost = robot_cost + mode_weight*(ca.mtimes([(opt_states[j][k, :]- new_ref.T), Q, (opt_states[j][k, :]-new_ref.T).T] 
                #                 )+ ca.mtimes([opt_controls[j][k, :], R, opt_controls[j][k, :].T])) #+ 100000 * opt_epsilon_r[j][k]) 
            
                # for obs in self.static_obs:
                #     obs_x = obs[0]
                #     obs_y = obs[1]
                #     obs_dia = obs[2]
                    
                #     rob_obs_constraints_ = ca.sqrt((opt_states[k, 0]-obs_x)**2+(opt_states[k, 1]-obs_y)**2)-obs_dia/2 - self.rob_dia/2 - self.safety_margin #+ opt_epsilon_o[l]
                #     opti.subject_to(rob_obs_constraints_ >= 0)
            
            # boundrary and control conditions
            opti.subject_to(opti.bounded(-0.5, opt_x[j], 0.5))
            opti.subject_to(opti.bounded(-5, opt_y[j], 5))
            opti.subject_to(opti.bounded(-self.v_lim, v[j], self.v_lim))
            opti.subject_to(opti.bounded(-self.omega_lim, omega[j], self.omega_lim))
            # static obstacle constraint
            
        total_cost = robot_cost + collision_cost
        
        ##### Get chance constraints from the given GMM prediction
        ## aij = (pi - pj) / ||pi - pj|| and bij = ri + rj 
        ## aij^T(pi - pj) - bij >= erf^-1(1 - 2delta)sqrt(2*aij^T(sigma_i + sigma_j)aij)    
        pol_gains = []
        T_obs, c_obs, E_obs=[], [], []

        if self.feedback:
            K_rob_horizon = [opti.variable(2,2) for t in range(self.robust_horizon-1)]
        else:
            K_rob_horizon = [ca.DM(2,2) for t in range(self.robust_horizon-1)]
        
        for agent_prediction, agent_noise in zip(filtered_predictions_vector, filtered_noise_chars):
            T_obs_k, c_obs_k, E_obs_k=[], [], []
            pol_gains_k=[]

            for mode, prediction in agent_prediction.items():
                mean_traj = prediction['means']
                covariances = prediction['covariances']
                mean_inputs  = agent_noise[mode]['means']
                covar_inputs = agent_noise[mode]['covariances']

                if self.feedback:
                    K = K_rob_horizon+[opti.variable(2,2) for t in range(self.N-self.robust_horizon)]
                else:
                    K = K_rob_horizon+[ca.DM(2,2) for t in range(self.N-self.robust_horizon)]
                
                K_stack=ca.diagcat(opti.variable(2,2),*[K[t] for t in range(self.N-1)]) 

                obs_xy_cov = ca.diagcat(*[ covariances[i][:2,:2] for i in range(self.N)])
     
                total_cost+= 3*ca.trace((K_stack@obs_xy_cov@obs_xy_cov.T@K_stack.T))

                pol_gains_k.append(K_stack)
        
                T_o, c_o, E_o= self._get_obs_ATV_dynamics(mean_inputs, covar_inputs, mean_traj)

                T_obs_k.append(T_o)
                c_obs_k.append(c_o)
                E_obs_k.append(E_o)

            pol_gains.append(pol_gains_k)
            T_obs.append(T_obs_k)
            c_obs.append(c_obs_k)
            E_obs.append(E_obs_k)

        for k, agent_prediction in enumerate(filtered_predictions_vector):
            for j, prediction in agent_prediction.items():
                for t in range(1,self.N):
                    if self.linearized_ca:
                        ## Prob(||(tv_pos + tv_noise - opt_state- noise_correction)||_2^2 >= 4*self.rob^2_dia) >= 1-epsilon
                        ## g(o,p) = || o -p |||^2,   l(o,p) = g(o_0, p_0) +  dg_p(p-p_0) + dg_o (o-o_0)

                        ## Linearization procedure: Project current_state (of robot) onto a sphere of radius rob_dia, and centered at tv_pos.

                        #### Linearized constraint :  
                        ##   rob_dia^2 +  2*(tv_pos - curr_state)@(opt_state+noise_correction - curr_state) - 2*(tv_pos - curr_state)@(tv_pos+tv_noise - tv_pos) > = rob_dia^2
                        ##  ==>   (tv_pos - curr_state)@(opt_state+noise_correction - curr_state - tv_noise) > =0

                        #### New chance constraint : 
                        ## Prob ( (tv_pos - curr_state)@(opt_state+noise_correction - curr_state - tv_noise) > =0 ) >= 1-eps
                        ##  ==>  Prob((tv_pos - curr_state)@(noise_correction -tv_noise) >= -(tv_pos - curr_state)@(opt_state-curr_state)  ) >=1-eps
                        ##  ==>  sp.erfinv(1-eps)*||(tv_pos-curr_state)@(noise_covar + tv_covar)|| >=  -(tv_pos - curr_state)@(opt_state-curr_state)
                        
                        ## opt_state[k, :]+noise_correction[k] = A_rob[k, :]@curr_state + B_rob[k,:]@opt_controls+  + C_rob +     B_rob[k,:]@K_stack[k][j]@(O_stack[j]- E[O_stack[j]]) + E_rob@W_t
                        ## noise_correction[k]   =  B_rob[k,:]@K_stack[k][j]@E_obs[k][j]@N_t + E_rob@W_t
                        
                        tv_pos   = ca.DM(prediction['means'][t-1][:2])
                        if type(self.prev_states[agent_id])==type([]):
                            ref_pos = self.prev_states[agent_id][j][t,:2].T
                        else:
                            ref_pos = ca.DM(current_state)[:2]
                        
                        # ref_pos = ref[t,:2]

                        rob_proj = tv_pos+2*self.rob_dia*(ref_pos-tv_pos)/ca.norm_2(ref_pos-tv_pos)
                        
                        rv_dist  = sp.erfinv(1-2*self.delta)*(rob_proj-tv_pos).T@(2*ca.horzcat(E_rob[j][t*3:(t+1)*3-1,:],*[B_rob[j][t*3:(t+1)*3-1,:]@pol_gains[l][j]@E_obs[l][j][:-2,:]-int(l==k)*E_obs[k][j][t*2:(t+1)*2,:] for l in range(self.n_obs)]))
                        
                        nom_dist = (rob_proj-tv_pos).T@(opt_states[j][t,:2].T-rob_proj)

                        # opti.subject_to(rv_dist@rv_dist.T<=(opt_epsilon_r[j][t-1]+nom_dist)**2)
                        opti.subject_to(rv_dist@rv_dist.T<=(nom_dist)**2)
                        # opti.subject_to(nom_dist>=-opt_epsilon_r[j][t-1])
                        opti.subject_to(nom_dist>=0)

                    # else:
                    #     tv_pos   = ca.DM(prediction['means'][t-1][:2])
                    #     ##### Get chance constraints from the given GMM prediction
                    #     ## aij =(pi + Eini - pj- Ejnj)  and bij = ri + rj 
                    #     ##     P[ aij^T@aij <= bij**2 ]< eps  
                    #     ## <==>P[([ni;nj].T M[ni;nj] - Tr([I -I].T@[I -I]) <= -((pi-pj)^T@(pi-pj)+Tr(M)-bij**2)] < eps     
                    #     ##  ==> Var([ni;nj].T M[ni;nj]) < =eps*{Var([ni;nj].T M[ni;nj]) +  ( (pi-pj)^T@(pi-pj) + Tr(M)  -bij**2)**2)          
                    #     ###    Last inequality by Cantelli's : https://en.wikipedia.org/wiki/Cantelli%27s_inequality) 
                    #     pi = ca.vec(opt_states[j][t,:2])
                    #     pj = ca.vec(tv_pos)

                    #     joint_rv = ca.horzcat(E_rob[j][t*3:(t+1)*3-1,:],*[B_rob[j][t*3:(t+1)*3-1,:]@pol_gains[l][j]@E_obs[l][j][:-2,:]-int(l==k)*E_obs[k][j][t*2:(t+1)*2,:] for l in range(n_obs)])
                    #     joint_cov = joint_rv.T@joint_rv

                    #     tr_M_ = ca.trace(joint_cov)
                    #     lmbd_ = (pi-pj).T@(pi-pj) + tr_M_ - 4*self.rob_dia**2
                    #     Var_  = 2*ca.trace(joint_cov@joint_cov)

                    #     rob_rob_constraint = self.delta*(Var_ + lmbd_**2) - Var_
                    #     opti.subject_to(rob_rob_constraint >= -opt_epsilon_r[j][t-1])                        
                    #     opti.subject_to(rob_rob_constraint >= 0)

        # opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
        #                     'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
        #                     'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}
        # # opts_setting = {'ipopt.print_level': 0, 'print_time': 0,}
        opti.minimize(total_cost)

        # opti.solver('ipopt', opts_setting)
        opti.solver('osqp')
        opti.set_value(opt_xs, self.final_state[agent_id])

        # # set optimizing target withe init guess
        for j in range(self.num_modes):
            if type(self.prev_controls[agent_id])!=type([]):
                opti.set_initial(opt_controls[j], self.prev_controls[agent_id])  # (N, 2)
            else:
                opti.set_initial(opt_controls[j], self.prev_controls[agent_id][j])

        u_res = None
        next_states_pred = None

        try:     
            # solve the optimization problem
            t_ = time.time()
            sol = opti.solve()
            solve_time = time.time() - t_
            print("Agent " + str(agent_id) + " Solve Time: " + str(solve_time))
            
            for mode in range(self.num_modes):
                self.feedback_gains[mode] = sol.value(pol_gains[0][mode]).toarray()
                self.feedback_gains_cache[mode].append(sol.value(pol_gains[0][mode]).toarray())

            # obtain the control input
            u_res = [sol.value(opt_controls[j]) for j in range(self.num_modes)]
            # next_states_pred = sol.value(opt_states)
            next_states_pred = [[ca.DM(current_state).T] for j in range(self.num_modes)]

            for j in range(self.num_modes):
                for t in range(u_res[j].shape[0]):
                    next_states_pred[j].append(self.model.fCd(next_states_pred[j][-1], u_res[j][t,:]).T)
                next_states_pred[j] = ca.vertcat(*next_states_pred[j])
            # eps_o = sol.value(opt_epsilon_o)
            
            self.prev_states[agent_id] = next_states_pred
            self.prev_controls[agent_id] = u_res
            self.prev_pol = pol_gains
            
            # self.prev_epsilon_o[agent_id] = eps_o 
            self.rob_affine_model['A']=sol.value(A)
            self.rob_affine_model['B']=sol.value(B)
            self.rob_affine_model['C']=sol.value(C)
            self.rob_affine_model['E']=sol.value(E) 
        
        except RuntimeError as e:
            print("Infeasible solve")
  
        return u_res, next_states_pred
    
    def simulate(self):
        self.setup_visualization()
        self.setup_visualization_heatmap()
        
        # parallelized implementation
        while (not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
            time_1 = time.time()
            print(self.num_timestep)

            uncontrolled_traj = [self.uncontrolled_fleet_data[obs_idx]['executed_traj'] for obs_idx in range(self.n_obs)]
            self.current_uncontrolled_state = [uncontrolled_traj[obs_idx][self.num_timestep] for obs_idx in range(self.n_obs)]
            self.gmm_predictions = self.uncontrolled_fleet.get_gmm_predictions_from_current(self.current_uncontrolled_state)
            self.noise_chars = self.uncontrolled_fleet.get_gmm_predictions()
            self.mode_prob = [prob for obs_idx in range(self.n_obs) for prob in self.uncontrolled_fleet_data[obs_idx]['mode_probabilities'][self.num_timestep]]
            
            self.obs_affine_model ={obs_idx: {mode: {'T': None, 'c':None, 'E':None, 'covars':None} for mode in range(len(self.gmm_predictions[obs_idx]))} for obs_idx in range(self.n_obs)}
            self.rob_affine_model ={'A': None, 'B':None, 'C':None, 'E':None}
            for obs_idx, (agent_prediction, agent_noise) in enumerate(zip(self.gmm_predictions, self.noise_chars)):
                for mode, prediction in agent_prediction.items():
                    mean_traj = prediction['means']
                    covariances = prediction['covariances']
                    mean_inputs  = agent_noise[mode]['means']
                    covar_inputs = agent_noise[mode]['covariances']
                    
                    obs_xy_cov = ca.diagcat(*[ covariances[i][:2,:2] for i in range(self.N)])
                    T_o, c_o, E_o= self._get_obs_ATV_dynamics(mean_inputs, covar_inputs, mean_traj)
                    
                    self.obs_affine_model[obs_idx][mode].update({'T': T_o, 'c':c_o, 'E':E_o, 'covars':obs_xy_cov})

            # Create a multiprocessing pool
            pool = mp.Pool()
    
            # Apply MPC solve to each agent in parallel
            results = [self.run_single_mpc(0, np.array(self.current_state[0]), [])]
                
            pool.close()
            pool.join()

            self.plot_gmm_means_and_state(self.current_state[0], self.prediction_cache[0], self.gmm_predictions, self.mode_prob, ref=self.ref)
            self.plot_feedback_gains()

            # Process the results and update the current state
            for agent_id, result in enumerate(results):
                u, next_states_pred = result
                if u is None:
                    self.infeasible_count += 1
                    self.infeasible = True
                    u = [np.zeros((self.N, 2))]
                    current_state = np.array(self.current_state[agent_id])
                    next_state = self.shift_movement(current_state, u[0], self.f_np)

                    self.prediction_cache[agent_id] = next_states_pred
                    self.control_cache[agent_id].append(u[0])
                    self.current_state[agent_id] = next_state
                    self.state_cache[agent_id].append(next_state)
                else:
                    current_state = np.array(self.current_state[agent_id])
                    next_state = self.shift_movement(current_state, u[0], self.f_np)

                    self.prediction_cache[agent_id] = next_states_pred
                    self.control_cache[agent_id].append(u[0])
                    self.current_state[agent_id] = next_state
                    self.state_cache[agent_id].append(next_state)
                    # print("Agent state: ", next_state, " Agent control: ", u[0,:])
            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)

        if self.is_solution_valid(self.state_cache):
            print("Executed solution is GOOD!")
            self.max_comp_time = max(self.avg_comp_time)
            self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            # self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
            self.success = True
            # self.feedback_gain_avg = compute_average_norm(self.feedback_gains_cache)
        else:
            self.success = False
        
        run_description = self.scenario 

        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.control_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached, self.infeasible_count, self.feedback_gain_avg, self.uncontrolled_fleet_data, self.num_timestep)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)
        