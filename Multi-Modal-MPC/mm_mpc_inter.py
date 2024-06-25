import casadi as ca
import numpy as np
import time
from mpc_base import MPC_Base
import multiprocessing as mp
from utils import *
import scipy.special as sp
from scipy.stats import multivariate_normal
import pdb

class MM_MPC_TI(MPC_Base):            
    def _get_robot_ATV_dynamics(self):
        """
        Constructs system matrices such for the robot,
        X_t= A_pred@current_state + B_pred@U_t + C_pred + E_pred@W_t
        where
        X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
        U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence)
        W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T,  (EV process noise sequence)
        x_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 
        A = ca.DM([[1, self.dt],[0,1]])
        B = ca.DM([[0.5*self.dt**2], [self.dt]])

        E = 0.001*ca.DM.eye(2)
                
        A_pred=ca.DM(2*(self.N+1), 2)
        B_pred=ca.DM(2*(self.N+1),self.N)
        E_pred=ca.DM(2*(self.N+1),self.N*2)
        
        A_pred[:2,:]=ca.DM.eye(2)
        
        for t in range(1,self.N+1):
                A_pred[t*2:(t+1)*2,:]=A@A_pred[(t-1)*2:t*2,:]
                
                B_pred[t*2:(t+1)*2,:]=A@B_pred[(t-1)*2:t*2,:]
                B_pred[t*2:(t+1)*2,t-1]=B
                
                E_pred[t*2:(t+1)*2,:]=A@E_pred[(t-1)*2:t*2,:]
                E_pred[t*2:(t+1)*2,(t-1)*2:t*2]=E
                
        
        return A_pred,B_pred,E_pred
    
    def _get_obs_ATV_dynamics(self, u_tvs, tv_n_std):
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
        T_tv=ca.DM(2*(self.N+1), 2)
        TB_tv=ca.DM(2*(self.N+1), self.N)
        c_tv=ca.DM(2*(self.N+1), 1)
        E_tv=ca.DM(2*(self.N+1),self.N*2)

        E=ca.diag(tv_n_std)
        
        A = ca.DM([[1, self.dt],[0,1]])
        B = ca.DM([[0.5*self.dt**2], [self.dt]])
       
        for t in range(self.N+1):
            if t==0:
                T_tv[:2,:]=ca.DM.eye(2)
            else:
                T_tv[t*2:(t+1)*2,:]=A@T_tv[(t-1)*2:t*2,:]
                TB_tv[t*2:(t+1)*2,:]=A@TB_tv[(t-1)*2:t*2,:]
                TB_tv[t*2:(t+1)*2,t-1:t]=B
                E_tv[t*2:(t+1)*2,:]=A@E_tv[(t-1)*2:t*2,:]    
                E_tv[t*2:(t+1)*2,(t-1)*2:t*2]=E

        c_tv=TB_tv@u_tvs.T             

        return T_tv, c_tv, E_tv

    def run_single_mpc(self, agent_id, update_dict):
        # casadi parameters
        # opti = ca.Opti('conic')
        opti = ca.Opti()

        current_state = update_dict['x0']
        current_state_obs_vector = update_dict['o0']
        gmm_predictions_vector = update_dict['o_glob']
        mm_input_vector        = update_dict['u_tvs']
        noise_chars = update_dict['noise_std']
       

        ####
        # EV feedforward + TV state feedback policies from https://arxiv.org/abs/2109.09792
        # U_stack[j] = opt_controls + sum_i=1^{N_obs} K_stack[i][j]@(O_stack[i][j] - E[O_stack[i][j]])
        #### 
        # nominal feedforward controls
        u_ff = opti.variable(self.N, 1)
        
       
        # opt_controls_h = {obs_idx: {j :ca.vertcat(ca.DM(self.robust_horizon,1), 
        #                             opti.variable(self.N-self.robust_horizon,1)) for j in range(len(gmm_predictions_vector[obs_idx])) } for obs_idx in range(len)}
     
        rob_horizon_u = opti.variable(self.robust_horizon, 1)
        
        opt_controls = [ca.vertcat(rob_horizon_u, opti.variable(self.N-self.robust_horizon,1)) for _ in range(self.num_modes)]
        opt_epsilon_r = [opti.variable(self.N, 1) for _ in range(self.num_modes)]

        A_rob, B_rob, E_rob = [], [], []
        opt_states, opt_x, opt_y, v,a  = [], [], [], [], []
        for j in range(self.num_modes):
            # if np.linalg.norm(self.prev_states[agent_id][j])>1e-2:
            
            A, B, E = self._get_robot_ATV_dynamics()       
    
                
            A_rob.append(A); B_rob.append(B); E_rob.append(E)

            # nominal state predictions
            opt_states.append(ca.vec(A@ca.DM(current_state)+B@ca.vec(opt_controls[j].T)).reshape((-1,self.N+1)).T)

            # import pdb; pdb.set_trace()
            x_pos = update_dict['x_pos']
            d_pos = update_dict['dpos']
           
            x_traj = ca.vec(x_pos)+ ca.vertcat(ca.DM(2,1),ca.diagcat(*d_pos)@opt_states[-1][1:,0])
            opt_x.append(x_traj.reshape((2,-1))[0,:])
            opt_y.append(x_traj.reshape((2,-1))[1,:])
            v.append(opt_states[-1][:,1])
            a.append(opt_controls[j])
           

            # opti.subject_to(opti.bounded(0, opt_epsilon_r[j][:], 0.001))

        # opt_epsilon_o = opti.variable(self.N+1, 1)
        
        # parameters
        # opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(2)
        # self.opt_epsilon_r.append(self.opti.variable(self.N+1, 1))

        # define the cost function
        robot_cost = 0  # cost
    
        total_cost = 0
            
        Q = self.cost_func_params['Q'][:2,:2]
        R = self.cost_func_params['R'][:1,:1]
        P = self.cost_func_params['P'][:2,:2]
        
    

        mode_prob = [1/self.num_modes for j in range(self.num_modes)]
        for j in range(self.num_modes):
            for k in range(self.N):
                mode_weight = mode_prob[j]
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
            # opti.subject_to(opti.bounded(-1.0, opt_x[j], 1.0))
            # opti.subject_to(opti.bounded(-5, opt_y[j], 5))
            opti.subject_to(opti.bounded(-0.1, v[j], self.v_lim))
            opti.subject_to(opti.bounded(-3, a[j], 3))
      
       
            
        total_cost = robot_cost 
        
        ##### Get chance constraints from the given GMM prediction
        ## aij = (pi - pj) / ||pi - pj|| and bij = ri + rj 
        ## aij^T(pi - pj) - bij >= erf^-1(1 - 2delta)sqrt(2*aij^T(sigma_i + sigma_j)aij)    
        pol_gains = []
        T_obs, c_obs, E_obs=[], [], []

        if self.feedback:
            K_rob_horizon = [opti.variable(1,2) for t in range(self.robust_horizon-1)]
        else:
            K_rob_horizon = [ca.DM(1,2) for t in range(self.robust_horizon-1)]
        
        for k, agent_prediction_mm_u_tv in enumerate(zip(gmm_predictions_vector, mm_input_vector)):
            agent_prediction, mm_u_tv = agent_prediction_mm_u_tv
            T_obs_k, c_obs_k, E_obs_k=[], [], []
            pol_gains_k=[]

            for mode, prediction in enumerate(agent_prediction):
                u_tv = mm_u_tv[mode]
           
                covariances = ca.diag(noise_chars[k])
                

                if self.feedback:
                    K = K_rob_horizon+[opti.variable(1,2) for t in range(self.N-self.robust_horizon)]
                else:
                    K = K_rob_horizon+[ca.DM(1,2) for t in range(self.N-self.robust_horizon)]
                
                K_stack=ca.diagcat(ca.DM(1,2),*[K[t] for t in range(self.N-1)]) 

                
                obs_xy_cov = ca.diagcat(*[ covariances[:2,:2] for i in range(self.N)])
     
                total_cost+= ca.trace((K_stack@obs_xy_cov@obs_xy_cov.T@K_stack.T))

                pol_gains_k.append(K_stack)
        
                T_o, c_o, E_o= self._get_obs_ATV_dynamics(u_tv, noise_chars[k])

                T_obs_k.append(T_o)
                c_obs_k.append(c_o)
                E_obs_k.append(E_o)

            pol_gains.append(pol_gains_k)
            T_obs.append(T_obs_k)
            c_obs.append(c_obs_k)
            E_obs.append(E_obs_k)

        for k, agent_prediction in enumerate(gmm_predictions_vector):
            for j, prediction in enumerate(agent_prediction):
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
                        
                        oa_ref=prediction[:,t]
                        
                        x_pos = update_dict['x_pos']
                        Qs = update_dict['Qs']
                        dpos = update_dict['dpos']
                        dpos_tvs = update_dict['droutes']
                        z_lin = update_dict['z_lin']
                       
                        # oa_ref+=(self.x_pos[:,t]-self.pos_tvs[k][m][:,t])/((self.x_pos[:,t]-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,t]-self.pos_tvs[k][m][:,t]))**(0.5)
                        oa_ref+=(x_pos[:,0]-oa_ref)/((x_pos[:,0]-oa_ref).T@Qs[k][j][t-1]@(x_pos[:,0]-oa_ref))**(0.5)
                        # Coefficient of random variables in affine chance constraint
                       
                        rv_dist=sp.erfinv(1-2*self.delta)*((oa_ref-prediction[:,t]).T@Qs[k][j][t-1]@(ca.horzcat(dpos[t-1]@(E[2*t,:]),*[dpos[t-1]@B[2*t,:]@pol_gains[l][j]@E_obs[l][j][:2*self.N,:]-int(l==k)*dpos_tvs[k][j][t-1]@E_obs[k][j][2*t,:] for l in range(self.n_obs)])))
                        
                        # constant term in affine chance constraint
                        # y=(oa_ref-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,t]-oa_ref+self.dpos[t-1]*(A[2*t,:]@self.z_curr+B[2*t,:]@h[j]-self.z_lin[0,t]))
                        try:
                            nom_dist=(oa_ref-prediction[:,t]).T@Qs[k][j][t-1]@(x_pos[:,t]-oa_ref+dpos[t-1]*(A[2*t,:]@ca.DM(current_state)+B[2*t,:]@ca.vec(opt_controls[j].T)-z_lin[0,t]))
                        except:
                            import pdb; pdb.set_trace()
                        # tv_pos   = ca.DM(prediction['means'][t-1][:2])
                        # if type(self.prev_states[agent_id])==type([]):
                        #     ref_pos = self.prev_states[agent_id][j][t,:2].T
                        # else:
                        #     ref_pos = ca.DM(current_state)[:2]
                        
                        # # ref_pos = ref[t,:2]

                        # rob_proj = tv_pos+2*self.rob_dia*(ref_pos-tv_pos)/ca.norm_2(ref_pos-tv_pos)
                        
                        # rv_dist  = sp.erfinv(1-2*self.delta)*(rob_proj-tv_pos).T@(2*ca.horzcat(E_rob[j][t*3:(t+1)*3-1,:],*[B_rob[j][t*3:(t+1)*3-1,:]@pol_gains[l][j]@E_obs[l][j][:-2,:]-int(l==k)*E_obs[k][j][t*2:(t+1)*2,:] for l in range(self.n_obs)]))
                        
                        # nom_dist = (rob_proj-tv_pos).T@(opt_states[j][t,:2].T-rob_proj)

                        # opti.subject_to(rv_dist@rv_dist.T<=(opt_epsilon_r[j][t-1]+nom_dist)**2)
                        # opti.subject_to(nom_dist>=-opt_epsilon_r[j][t-1])
                        
                        opti.subject_to(rv_dist@rv_dist.T<=(nom_dist)**2)
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
        opts_setting = {'ipopt.print_level': 0, 'print_time': 0,}
        opti.minimize(total_cost)

        opti.solver('ipopt', opts_setting)
        # opti.solver('osqp')
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
                self.feedback_gains[0][mode] = sol.value(pol_gains[0][mode]).toarray()
                self.feedback_gains_cache[0][mode].append(sol.value(pol_gains[0][mode]).toarray())

            # obtain the control input
            u_res = [sol.value(opt_controls[j]) for j in range(self.num_modes)]
            # next_states_pred = sol.value(opt_states)
            next_states_pred = [[ca.DM(current_state).T] for j in range(self.num_modes)]

            for j in range(self.num_modes):
                # for t in range(u_res[j].shape[0]):
                #     next_states_pred[j].append()
                # next_states_pred[j] = ca.vertcat(*next_states_pred[j])
                next_states_pred[j] = sol.value(opt_states[j])
           
            # eps_o = sol.value(opt_epsilon_o)
            
            self.prev_states[agent_id] = next_states_pred
            self.prev_controls[agent_id] = u_res
            self.prev_pol = pol_gains
            
            # self.prev_epsilon_o[agent_id] = eps_o 
        
        except RuntimeError as e:
            print("Infeasible solve")
  
        return u_res, next_states_pred
    
    def simulate(self, Sim):
        # self.setup_visualization()
        # self.setup_visualization_heatmap()
        
        # parallelized implementation
        # while (not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
        while Sim.t<self.total_sim_timestep and not Sim.done():
            time_1 = time.time()
            print(self.num_timestep)
    
            # Create a multiprocessing pool
            # pool = mp.Pool()
    
            # Apply MPC solve to each agent in parallel
            if type(self.prev_controls[0]) ==type([]):
                u_ws = self.prev_controls[0][0]
            else:
                u_ws = self.prev_controls[0]
            results = [self.run_single_mpc(0, Sim.get_update_dict(u_ws))]
            
            # results = pool.starmap(self.run_single_mpc, [(agent_id, np.array(self.current_state[agent_id]), []) for agent_id in range(self.num_agent)])
    
            # pool.close()
            # pool.join()

            # current_state_obs_vector = [self.uncontrolled_fleet_data[obs]['executed_traj'][self.num_timestep] for obs in range(len(self.uncontrolled_fleet_data))]
            # gmm_predictions = self.uncontrolled_fleet.get_gmm_predictions_from_current(current_state_obs_vector)

            # mode_prob = self.uncontrolled_fleet_data[0]['mode_probabilities'][self.num_timestep] 
      
            # self.plot_gmm_means_and_state(self.current_state[0], self.prediction_cache[0], gmm_predictions, mode_prob, ref=self.ref)
            # self.plot_feedback_gains()

            # Process the results and update the current state
            for agent_id, result in enumerate(results):
                u, next_states_pred = result
                if u is None:
                    self.infeasible_count += 1
                    self.infeasible = True
                    u = [np.zeros((self.N, 1))]
                    # current_state = np.array(self.current_state[agent_id])
                    # next_state = self.shift_movement(current_state, u[0], self.f_np)

                    # self.prediction_cache[agent_id] = next_states_pred
                    # self.control_cache[agent_id].append(u[0])
                    # self.current_state[agent_id] = next_state
                    # self.state_cache[agent_id].append(next_state)
                    Sim.step(0.)
                else:
                    # current_state = np.array(self.current_state[agent_id])
                    # next_state = self.shift_movement(current_state, u[0], self.f_np)

                    # self.prediction_cache[agent_id] = next_states_pred
                    # self.control_cache[agent_id].append(u[0])
                    # self.current_state[agent_id] = next_state
                    # self.state_cache[agent_id].append(next_state)
                    Sim.step(u[0])
                    print("Agent state: ", Sim.ev.traj[:,Sim.t], " Agent control: ", u[0])
            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)

        # if self.is_solution_valid(self.state_cache):
        #     print("Executed solution is GOOD!")
        #     self.max_comp_time = max(self.avg_comp_time)
        #     self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
        #     # self.traj_length = get_traj_length(self.state_cache)
        #     self.makespan = self.num_timestep * self.dt
        #     self.success = True
        #     self.feedback_gain_avg = compute_average_norm(self.feedback_gains_cache)
        # else:
        #     self.success = False
        
        # run_description = self.scenario 

        # self.logger.log_metrics(run_description, self.trial, self.state_cache, self.control_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached, self.infeasible_count, self.feedback_gain_avg, self.uncontrolled_fleet_data, self.num_timestep)
        # self.logger.print_metrics_summary()
        # self.logger.save_metrics_data()
        
        # draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)
        