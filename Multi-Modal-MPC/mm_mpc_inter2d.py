import casadi as ca
import numpy as np
import time
from mpc_base import MPC_Base
import multiprocessing as mp
from utils import *
import scipy.special as sp
from scipy.stats import multivariate_normal
import pdb
from functools import reduce




class MM_MPC_TI(MPC_Base):            
    def _get_robot_ATV_dynamics(self, x_lin):
        """
        Constructs system matrices such for the robot,
        X_t= A_pred@current_state + B_pred@U_t + C_pred + E_pred@W_t
        where
        X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
        U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence)
        W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T,  (EV process noise sequence)
        x_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 
        
        v_sched = lambda v_x : 0.001 if v_x < 2 else 0.1
  

        E = 0.001*ca.DM.eye(3)
                
        A_pred=ca.DM(3*(self.N+1), 3)
        B_pred=ca.DM(3*(self.N+1),2*self.N)
        E_pred=ca.DM(3*(self.N+1),self.N*3)
        
        A_pred[:3,:]=ca.DM.eye(3)
        
        for t in range(1,self.N+1):
            A = np.array([[1.0, self.dt, 0],[0.,  1., 0.], [0., 0., 1.]])
            B = np.array([[0.5*self.dt**2, 0],[self.dt, 0], [0, self.dt*v_sched(x_lin[1,t])]])
            A_pred[t*3:(t+1)*3,:]=A@A_pred[(t-1)*3:t*3,:]
            
            B_pred[t*3:(t+1)*3,:]=A@B_pred[(t-1)*3:t*3,:]
            B_pred[t*3:(t+1)*3,2*(t-1):2*t]=B
            
            E_pred[t*3:(t+1)*3,:]=A@E_pred[(t-1)*3:t*3,:]
            E_pred[t*3:(t+1)*3,(t-1)*3:t*3]=E
                
        
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
                E_tv[t*2:(t+1)*2,(t-1)*2:t*2]=E#**(1-1/2*(1-self.feedback))

        c_tv=TB_tv@u_tvs.T             

        return T_tv, c_tv, E_tv

    def run_single_mpc(self, agent_id, update_dict):
        # casadi parameters
        # if self.linearized_ca:
        opti = ca.Opti('conic')
        # else:
        # opti = ca.Opti()
        # opts_setting = {'ipopt.max_iter': 100000, 'ipopt.print_level': 0, 'print_time': 0,
        #                     'ipopt.acceptable_tol': 1e-5, 'ipopt.acceptable_obj_change_tol': 1e-4, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9, 'ipopt.max_cpu_time' : 10.0,
        #                     'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}
        # opts_setting = {'ipopt.max_iter':500,  'ipopt.print_level': 0, 'print_time': 0, 'ipopt.warm_start_init_point': 'yes', 'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_obj_change_tol': 1e-3, 'ipopt.max_cpu_time' : 30.0}
        

        # opti.solver('ipopt', opts_setting)
        
        opts = {
            # "PreQLinearize": 1,
            "gurobi.PreSparsify": 1,
            # "gurobi.NumericFocus": 1,
            # "gurobi.Method": -1,
            # "gurobi.Threads": 8,
            "BarConvTol": 1e-3,
            "verbose": True,  # Optional: print solver output
        }
        opti.solver('gurobi', {}, {})
        # opti.solver('proxqp', {}, {'verbose':False})

        current_state = update_dict['x0']
        x_lin         = update_dict['z_lin']
        current_state_obs_vector = update_dict['o0']
        gmm_predictions_vector = update_dict['o_glob']
        mm_input_vector        = update_dict['u_tvs']
        obs_dims               = update_dict['sizes']
        noise_chars = update_dict['noise_std']
        mode_probabilities = update_dict.get('mode_probabilities', None)
        n_obs = len(gmm_predictions_vector)
        from itertools import product
        mode_combos =list(product(*[list(range(self.num_modes)) for _ in range(n_obs)]))
        mode_map = lambda x : mode_combos[x[0]][x[1]]
        
        Revs = update_dict['Revs']
        
        if 'clusters' in update_dict and 'SM-MPC' in self.scenario:
            clusters = update_dict['clusters']
            num_modes = len(clusters)
            scene_modes = max(num_modes,1)
            print(f'Doing SM-MPC with clusters: {clusters}')
        else:
            clusters = None
            num_modes = self.num_modes
            scene_modes = num_modes**n_obs
            
        print(f'Scene modes: {scene_modes}')
        def _get_mm_bias(j):
            if clusters is  None:       
                return reduce(lambda x,y : x+y, [opt_bias_mm[k][mode_map((j,k))] for k in range(n_obs)])
            else:
                return reduce(lambda x,y : x+y, [opt_bias_mm[k][j] for k in range(n_obs)])
        
        rob_u = opti.variable(self.N, 2)
        num_dec_var = 2*self.N
        num_constr = 0
        # h[j] = h_b + sum_{obst}h[k][j]  for MM, J= scene_mode, for SM, j = cluster index
        if num_modes:
            opt_bias_mm  = [[opti.variable(self.N-self.robust_horizon,2) for _ in range(num_modes)] for _ in range(n_obs)]
        else:
            opt_bias_mm  = [[ca.DM(self.N-self.robust_horizon,2)] for _ in range(n_obs)]

        opt_controls = [rob_u+ca.vertcat(ca.DM(self.robust_horizon,2), _get_mm_bias(j)) for j in range(scene_modes)]
        
        num_dec_var+= n_obs*(self.N-self.robust_horizon)*2 
        # print(f"Decision variables after declaring bias terms: {num_dec_var}")
              
          
        A_rob, B_rob, E_rob = [], [], []
        slack = opti.variable(1)
        opt_states, opt_x, opt_y, v, a, ey  = [], [], [], [], [], []
        num_dec_var+=1
        A, B, E = self._get_robot_ATV_dynamics(x_lin)    
        for j in range(scene_modes):
            # if np.linalg.norm(self.prev_states[agent_id][j])>1e-2:
            
            A_rob.append(A); B_rob.append(B); E_rob.append(E)

            # nominal state predictions
            opt_states.append(ca.vec(A@ca.DM(current_state)+B@ca.vec(opt_controls[j].T)).reshape((-1,self.N+1)).T)

            # import pdb; pdb.set_trace()
            ev_global_pos = update_dict['x_pos']
            jac_ev_pos = update_dict['dpos']
            ev_glob_pos_traj = ca.vec(ev_global_pos)+ \
                ca.vertcat(ca.DM(2,1),ca.vec(ca.diagcat(*jac_ev_pos)@ca.vec(opt_states[-1][1:,0::2])))
            opt_x.append(ev_glob_pos_traj.reshape((2,-1))[0,:])
            opt_y.append(ev_glob_pos_traj.reshape((2,-1))[1,:])
            v.append(opt_states[-1][:,1])
            a.append(opt_controls[j])
            ey.append(opt_states[-1][:,-1])
           
        
        # parameters
        # opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3)
        # self.opt_epsilon_r.append(self.opti.variable(self.N+1, 1))
       
        # define the cost function
        robot_cost = 0  # cost
    
        total_cost = 0
            
        Q = self.cost_func_params['Q'][:2,:2]
        R = self.cost_func_params['R'][:2,:2]
        P = self.cost_func_params['P'][:2,:2]
        
    

        # Use updated mode probabilities if available, otherwise fall back to uniform
        if mode_probabilities is not None:
            # Calculate joint probabilities for each scene mode
            mode_prob = []
            for j in range(scene_modes):
                if clusters is None or not clusters:
                    # For MM-MPC: multiply individual mode probabilities
                    joint_prob = 1.0
                    for k in range(n_obs):
                        mode_idx = mode_map((j, k))
                        if mode_idx < len(mode_probabilities[k]):
                            joint_prob *= mode_probabilities[k][mode_idx]
                        else:
                            joint_prob *= 1.0 / self.num_modes  # fallback
                    mode_prob.append(joint_prob)
                else:
                    # For SM-MPC: use cluster probabilities
                    mode_prob.append(1.0 / len(clusters))  # uniform within clusters
        else:
            # Fallback to uniform probabilities if not available
            mode_prob = [(1/self.num_modes)**n_obs for j in range(scene_modes)]
        
        print(f"MPC using mode probabilities: {mode_prob}")
        
        for j in range(scene_modes):
            # robot_cost += 10000*(opt_states[j][-1,0] - 110)**2
            for k in range(self.N):
                mode_weight = mode_prob[mode_map((j,0))]
                # if k > self.robust_horizon:
                # robot_cost = robot_cost + mode_weight*(ca.mtimes([(opt_states[j][k, :]-opt_xs.T), Q, (opt_states[j][k, :]-opt_xs.T).T] 
                #             )+ ca.mtimes([opt_controls[j][k, :], R, opt_controls[j][k, :].T]) + 100000 * opt_epsilon_r[j][k]) #+ 100000 * opt_epsilon_o[k]
                robot_cost = robot_cost + mode_weight*(-2*opt_states[j][k,0]
                    + 10*ca.mtimes([opt_controls[j][k, :], R, opt_controls[j][k, :].T]) ) #+ 100000 * opt_epsilon_r[j][k]) 
                if k>0:
                    robot_cost+= 10000*mode_weight*(opt_controls[j][k-1,:]-opt_controls[j][k,:])@(opt_controls[j][k-1,:]-opt_controls[j][k,:]).T
                    robot_cost+= 1000*mode_weight*(opt_states[j][k-1,2] - opt_states[j][k,2])**2
                else:
                    robot_cost+= 10000*mode_weight*(ca.DM(1,2)-opt_controls[j][k,:])@(opt_controls[j][k-1,:]-opt_controls[j][k,:]).T
                    robot_cost+= 1000*mode_weight*( current_state[2]- opt_states[j][k,2])**2
                    
                opti.subject_to(opti.bounded(-1, v[j], 6))#self.v_lim))
            opti.subject_to(opti.bounded(-100, a[j], 3))
            opti.subject_to(opti.bounded(-2.5, ey[j], 2.5))
            # opti.subject_to(opti.bounded(-5, opt_controls[j][:self.N-1,0]-opt_controls[j][1:self.N,0], 5))
            
            num_constr+= 3*self.N*2
      
        opti.subject_to(opti.bounded(0,slack,.2))
        num_constr+=2
        total_cost = robot_cost + 1000*slack**2
        
        ##### Get chance constraints from the given GMM prediction
        ## aij = (pi - pj) / ||pi - pj|| and bij = ri + rj 
        ## aij^T(pi - pj) - bij >= erf^-1(1 - 2delta)sqrt(2*aij^T(sigma_i + sigma_j)aij)    
        
        if self.feedback:
            K_rob_horizon = [opti.variable(2,2) for t in range(self.robust_horizon-1)]
            num_dec_var+= (self.robust_horizon-1)*2*2
            # print(f"Decision variables after declaring robust policy terms: {num_dec_var}")
            
        else:
            K_rob_horizon = [ca.DM(2,2) for t in range(self.robust_horizon-1)]
        if clusters is None or 'SM-MPC' not in self.scenario:
            pol_gains = []
            T_obs, c_obs, E_obs=[], [], []  
            
            for k, agent_prediction_mm_u_tv in enumerate(zip(gmm_predictions_vector, mm_input_vector)):
                agent_prediction, mm_u_tv = agent_prediction_mm_u_tv
                T_obs_k, c_obs_k, E_obs_k=[], [], []
                pol_gains_k=[]

                for mode, prediction in enumerate(agent_prediction):
                    u_tv = mm_u_tv[mode]
            
                    covariances = ca.diag(noise_chars[k])

                    if self.feedback:
                        K = K_rob_horizon+[opti.variable(2,2) for t in range(self.N-self.robust_horizon)]
                        num_dec_var+= (self.N-self.robust_horizon)*2*2
                    else:
                        K = K_rob_horizon+[ca.DM(2,2) for t in range(self.N-self.robust_horizon)]
                    
                    K_stack=ca.diagcat(ca.DM(2,2),*[K[t] for t in range(self.N-1)]) 
                    obs_xy_cov = ca.diagcat(*[ covariances[:2,:2] for i in range(self.N)])
        
                    total_cost+= 0.5*ca.trace((K_stack@obs_xy_cov@obs_xy_cov.T@K_stack.T))

                    pol_gains_k.append(K_stack)
            
                    T_o, c_o, E_o= self._get_obs_ATV_dynamics(u_tv, noise_chars[k])

                    T_obs_k.append(T_o)
                    c_obs_k.append(c_o)
                    E_obs_k.append(E_o)

                pol_gains.append(pol_gains_k)
                T_obs.append(T_obs_k)
                c_obs.append(c_obs_k)
                E_obs.append(E_obs_k)
        else:
            mm_tv_pred = {k: {mode :  pred for mode, pred in enumerate(agent_prediction)} for k, (agent_prediction, _) in enumerate(zip(gmm_predictions_vector, mm_input_vector))}
            mm_tv_u    = {k: {mode :  u_tv for mode, u_tv in enumerate(mm_u_tv)} for k, (_, mm_u_tv) in enumerate(zip(gmm_predictions_vector, mm_input_vector))}
            pol_gains = {k: {j :  None for j in range(scene_modes)} for k, agent_prediction in enumerate(zip(gmm_predictions_vector))}
            T_obs, c_obs, E_obs={k: {mode :  None for mode, _ in enumerate(agent_prediction)} for k, agent_prediction in enumerate(zip(gmm_predictions_vector))},\
                                {k: {mode :  None for mode, _ in enumerate(agent_prediction)} for k, agent_prediction in enumerate(zip(gmm_predictions_vector))},\
                                {k: {mode :  None for mode, _ in enumerate(agent_prediction)} for k, agent_prediction in enumerate(zip(gmm_predictions_vector))}
            
            # clusters = [[], []]
            # u = h + K(o1 + o2)         (agnostic to everyting)
            # u = h + K1o1 + K2o2   (agnostic to modes, but reactive to individual obstacle)
            # Scene modes
            # [(0, 0), (1,0), (2,0), (0,1), (0,2), (1,1), (1,2), (2,1), (2,2)]
            # num_K = n_obs*xy_dim*num_mode*(N-robust) (simplified MM without joint reasoning)
            # num_K = xy_dim*n_obs*(N-robust)*num_modes**(n_obs) (true MM with joint reasoning)
            
            
            
            # Clusters
            # C1: [(0,0), (1, 0), (2, 0)],   C2: [(0,1), (0,2), (1,1), (1,2), (2,1), (2,2)]
            #   K_1 [n_obs*xy_dim*(N-robust)]          K_2 [n_obs*xy_dim*(N-robust)]
            
            # given C clusters,
            # decision var comparison:   C*n_obs*xy_dim*(N-robust) vs n_obs*xy_dim*num_mode*(N-robust)
            for j, cluster in enumerate(clusters):   #cluster= [(0,0), (1, 0)]
                # print(f"Cluster {j} : {cluster}")
                for scen in cluster:
                    for k in range(n_obs):
                        if self.feedback:
                            if pol_gains[k][j] is None:
                                K = K_rob_horizon+[opti.variable(2,2) for t in range(self.N-self.robust_horizon)]
                                K_stack=ca.diagcat(ca.DM(2,2),*[K[t] for t in range(self.N-1)]) 
                                num_dec_var+= (self.N-self.robust_horizon)*2*2
                            else:
                                K_stack = pol_gains[k][j]
                                
                            
                        else:
                            K = K_rob_horizon+[ca.DM(2,2) for t in range(self.N-self.robust_horizon)]
                            K_stack=ca.diagcat(ca.DM(2,2),*[K[t] for t in range(self.N-1)]) 
                        
                        u_tv = mm_tv_u[k][scen[k]]
                        prediction = mm_tv_pred[k][scen[k]]
                    
                        covariances = ca.diag(noise_chars[k])
                        
                        obs_xy_cov = ca.diagcat(*[ covariances[:2,:2] for i in range(self.N)])
                        obs_xy_cov = ca.diagcat(*[ covariances[:2,:2] for i in range(self.N)])
            
                        total_cost+= 0.5*ca.trace((K_stack@obs_xy_cov@obs_xy_cov.T@K_stack.T))
                        
                        pol_gains[k][j] = K_stack
                        
                        T_o, c_o, E_o= self._get_obs_ATV_dynamics(u_tv, noise_chars[k])
                        
                        T_obs[k][scen[k]], c_obs[k][scen[k]], E_obs[k][scen[k]] = T_o, c_o, E_o
    
                
        ev_global_pos = update_dict['x_pos']
        agg_Q = update_dict['Qs']
        jac_ev_pos = update_dict['dpos']
        jac_tv_pos = update_dict['droutes']
        # print(f"Decision variables after declaring mm policy terms: {num_dec_var}")
        r_fun =update_dict['route_fun']
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("EV PRED: ", ev_global_pos)
        # print("TV PRED: ", gmm_predictions_vector[0][0])
        # print("Dist :", np.linalg.norm(ev_global_pos[:,1:]-gmm_predictions_vector[0][0][:,1:], axis = 0))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if clusters is None or 'SM-MPC' not in self.scenario:
            for k, agent_prediction in enumerate(gmm_predictions_vector):
                for j, prediction in enumerate(agent_prediction):
                    # print("Jacobians", "Ev", dpos, "Tv", jac_tv_pos[k][j])
                    # print("Positions","TV", prediction,  "EV", ev_global_pos)
                    for t in range(1,self.N,1):
                        
                        if k == 0:
                            ev_ref = ev_global_pos[:,t]
                        else:
                            ev_ref = ev_global_pos[:,t]
                        obs_avoid_lin_ref=prediction[:,t]+(ev_ref-prediction[:,t])/\
                            ((ev_ref-prediction[:,t]).T@agg_Q[k][j][t-1]@(ev_ref-prediction[:,t]))**(0.5)
                        # if j == 0 and k == 0 :
                        #     print(obs_avoid_lin_ref)
                        # Coefficient of random variables in affine chance constraint
                        for m in range( self.num_modes**n_obs):
                            # print(f"Scenario {m}, Obstacle {k}, Time {t} ")
                            if mode_map((m,k))!=j:
                                continue
                            # if t==1:
                            #     print(f"Scenario { m}, Obstacle {k}, Time {t}, Obstacle mode {j} ")
                            # else:
                            # ey  = A[3*(t+1)-1,:]@ca.DM(current_state)+B[3*(t+1)-1,:]@ca.vec(opt_controls[m].T)
                            # pos_dev = ca.vertcat(-ey*ca.sin(psi), ey*ca.cos(psi)) 
                            lin_dist = (obs_avoid_lin_ref-prediction[:,t]).T
                            # # print(f" time:{t} linearized relative displacement:", lin_dist, "ev position proj", obs_avoid_lin_ref, "tv position", prediction[:,t])
                            noise_coeff = ca.horzcat(jac_ev_pos[t-1]@(E[3*t:3*(t+1):2,:]),*[jac_ev_pos[t-1]@B[3*t:3*(t+1):2,:]@pol_gains[l][mode_map((m,l))]@E_obs[l][mode_map((m,l))][:2*self.N,:]-int(l==k)*(jac_tv_pos[k][j][t-1]@E_obs[k][j][2*t,:]) for l in range(self.n_obs)])
                            # noise_coeff_const = ca.horzcat(jac_ev_pos[t-1].T@(E[3*t:3*(t+1):2,:]),*[0*jac_ev_pos[t-1]@B[2*t,:]@pol_gains[l][mode_map((m,l))]@E_obs[l][mode_map((m,l))][:2*self.N,:]-int(l==k)*jac_tv_pos[k][j][t-1]@E_obs[k][j][2*t,:] for l in range(self.n_obs)])
                            rv_dist=sp.erfinv(1-self.delta)*(lin_dist@agg_Q[k][j][t-1]@noise_coeff)
                        
                        
                            try:
                                ds = A[3*t:3*(t+1):2,:]@ca.DM(current_state)+B[3*t:3*(t+1):2,:]@ca.vec(opt_controls[m].T)
                                dx = jac_ev_pos[t-1]@(ds-x_lin[0:3:2,t].T)
                                nom_dist=lin_dist@agg_Q[k][j][t-1]@(ev_global_pos[:,t].T+dx-obs_avoid_lin_ref)
                                # print(f"nominal distance delta:  {lin_dist@agg_Q[k][j][t-1]@(ev_global_pos[:,t]-obs_avoid_lin_ref + jac_ev_pos[t-1]*(A[2*t,:]@ca.DM(current_state)-z_lin[0,t]))}" )
                            except:
                                import pdb; pdb.set_trace()
                        
                            # opti.subject_to(rv_dist@rv_dist.T<=(nom_dist)**2)
                            # opti.subject_to(nom_dist>=0)
                            soc = ca.soc(rv_dist, nom_dist)
                            opti.subject_to(soc>0)

                            num_constr+= 2
        else:
            for j, cluster in enumerate(clusters):
                obs_mode_considered = set()
                for s_idx, scen in enumerate(cluster):

                    for k in range(n_obs):
                       
                        u_tv = mm_tv_u[k][scen[k]]
                        prediction = mm_tv_pred[k][scen[k]]
                        # print(f"Considering obstacle {k} in mode {scen[k]}")
                        if (k, scen[k]) not in obs_mode_considered:
                            obs_mode_considered.add((k, scen[k]))
                        else:
                        #     # print(f"obstacle {k} and mode {scen[k]} already in {obs_mode_considered}")
                            continue
                        for t in range(1,self.N):
                            # if t==1:
                            #     print(f"Cluster {j}, Scenario {s_idx}, Obstacle {k}, Time {t} ")
                            if k == 0:
                                ev_ref = ev_global_pos[:,t]
                            else:
                                ev_ref = ev_global_pos[:,t]
                            obs_avoid_lin_ref=prediction[:,t]+(ev_ref-prediction[:,t])/\
                                ((ev_ref-prediction[:,t]).T@agg_Q[k][scen[k]][t-1]@(ev_ref-prediction[:,t]))**(0.5)
                            # obs_avoid_lin_ref=prediction[:,t]
                            # obs_avoid_lin_ref+=(ev_global_pos[:,t]-obs_avoid_lin_ref)/((ev_global_pos[:,t]-obs_avoid_lin_ref).T@agg_Q[k][scen[k]][t-1]@(ev_global_pos[:,t]-obs_avoid_lin_ref))**(0.5)
                            
                            lin_dist = (obs_avoid_lin_ref-prediction[:,t]).T
                            noise_coeff = ca.horzcat(jac_ev_pos[t-1]@(E[3*t:3*(t+1):2,:]),*[jac_ev_pos[t-1]@B[3*t:3*(t+1):2,:]@pol_gains[l][j]@E_obs[l][scen[l]][:2*self.N,:]-int(l==k)*(jac_tv_pos[k][scen[k]][t-1]@E_obs[k][scen[k]][2*t,:]) for l in range(n_obs)])
                            rv_dist=sp.erfinv(1-self.delta)*(lin_dist@agg_Q[k][scen[k]][t-1]@noise_coeff)
                            ds = A[3*t:3*(t+1):2,:]@ca.DM(current_state)+B[3*t:3*(t+1):2,:]@ca.vec(opt_controls[j].T)
                            dx = jac_ev_pos[t-1]@(ds-x_lin[0:3:2,t].T)
                            nom_dist=lin_dist@agg_Q[k][scen[k]][t-1]@(ev_global_pos[:,t].T+dx-obs_avoid_lin_ref)
                            opti.subject_to(rv_dist@rv_dist.T<=(nom_dist)**2)
                            opti.subject_to(nom_dist>=0)
                            # soc = ca.soc(rv_dist, nom_dist)
                            # opti.subject_to(soc>0)
                            num_constr+=2
                        
        
        
        opti.set_value(opt_xs, ca.vertcat(self.final_state[agent_id],0))

        # set optimizing target withe init guess
        if type(self.prev_controls[agent_id])==type({}) and 'rob_u' in self.prev_controls[agent_id]:
            rob_u_init = self.prev_controls[agent_id]['rob_u']
            opti.set_initial(rob_u, rob_u_init)
            for k in range(n_obs):
                if len(self.prev_controls[agent_id]['bias'][k]) != num_modes:
                    continue
                for j in range(num_modes):
                    bias_init  = self.prev_controls[agent_id]['bias'][k][j]
                    opti.set_initial(opt_bias_mm[k][j], bias_init)

                # opti.set_initial(bias_terms[j], bias_init)  # (N, 2)
            # else:
            #     opti.set_initial(opt_controls[j], self.prev_controls[agent_id][j])

        u_res = None
        next_states_pred = None
        ev_glob_sol = None
        
        print(f"#constraints : {num_constr} and #decision_vars : {num_dec_var}")

        try:     
            # solve the optimization problem
            t_ = time.time()
            # import pdb; pdb.set_trace()
            sol = opti.solve()
            # print(sol)
            
            solve_time = time.time() - t_
            print("Agent " + str(agent_id) + " Solve Time: " + str(solve_time))
            
            # for mode in range(num_modes):
            #     self.feedback_gains[0][mode] = sol.value(pol_gains[0][mode]).toarray()
            #     self.feedback_gains_cache[0][mode].append(sol.value(pol_gains[0][mode]).toarray())

            # obtain the control input
            if clusters is None:
                u_res = [sol.value(opt_controls[j]) for j in range(scene_modes)]
                # next_states_pred = sol.value(opt_states)
                next_states_pred = [[ca.DM(current_state).T] for j in range(scene_modes)]

                ev_glob_sol      = [sol.value(ca.vertcat(opt_x[j].reshape((1,-1)), opt_y[j].reshape((1,-1)))) for j in range(scene_modes)]
                rob_u_sol   = sol.value(rob_u)
                bias_sols = [[None for j in range(num_modes)] for k in range(n_obs)]
                # obca_sols  = [[None for j in range(self.num_modes)] for k in range(n_obs)]
                for j in range(scene_modes):
                    # for t in range(u_res[j].shape[0]):
                    #     next_states_pred[j].append()
                    # next_states_pred[j] = ca.vertcat(*next_states_pred[j])
                    next_states_pred[j] = sol.value(opt_states[j])
                    for k in range(n_obs):
                        bias_sols[k][mode_map((j,k))] = sol.value(opt_bias_mm[k][mode_map((j,k))])
                    

            else:
                u_res = [sol.value(opt_controls[j]) for j in range(scene_modes)] 
                # next_states_pred = sol.value(opt_states)
                next_states_pred = [[ca.DM(current_state).T] for j in range(scene_modes)]
                rob_u_sol   = sol.value(rob_u)
                bias_sols = [[None for j in range(scene_modes)] for k in range(n_obs)]
                ev_glob_sol      = [sol.value(ca.vertcat(opt_x[j].reshape((1,-1)), opt_y[j].reshape((1,-1)))) for j in range(scene_modes)]
                
                
                for j in range(scene_modes):
                    # for t in range(u_res[j].shape[0]):
                    #     next_states_pred[j].append()
                    # next_states_pred[j] = ca.vertcat(*next_states_pred[j])
                    next_states_pred[j] = sol.value(opt_states[j])
                    if clusters is None:
                        for k in range(n_obs):
                            bias_sols[k][mode_map((j,k))] = sol.value(opt_bias_mm[k][mode_map((j,k))])
                    else:
                        for k in range(n_obs):
                            bias_sols[k][j] = sol.value(opt_bias_mm[k][j])
                        
                    
            
            self.prev_states[agent_id] = next_states_pred
            self.prev_controls[agent_id] = {'control': u_res, 'rob_u' : rob_u_sol, 'bias': bias_sols}#, 'obca_lmbd':obca_sols}
            self.prev_pol = pol_gains
            
            # self.prev_epsilon_o[agent_id] = eps_o 
        
        except RuntimeError as e:
            print("Infeasible solve")
  
        return u_res, next_states_pred, ev_glob_sol
    
    def simulate(self, Sim):
        # self.setup_visualization()
        # self.setup_visualization_heatmap()
        
        # parallelized implementation
        # while (not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
        collision_probability_traj = []
        while Sim.t<self.total_sim_timestep and not Sim.done() and not Sim._check_collision():
            time_1 = time.time()
            print(self.num_timestep)
    
            # Create a multiprocessing pool
            # pool = mp.Pool()
    
            # Apply MPC solve to each agent in parallel
            if type(self.prev_controls[0]) == type({}) and 'control' in self.prev_controls[0]:
                u_ws = self.prev_controls[0]['control'][0]
            else:
                
                # u_ws = self.prev_controls[0,:]
                u_ws = np.hstack([1*np.ones((self.N, 1)), np.zeros((self.N,1))])
            update_dict = Sim.get_update_dict(u_ws.T)
            results = [self.run_single_mpc(0, update_dict)]
            
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
                u, next_states_pred, ev_global_trajectories = result
                if u is None:

                    self.infeasible_count += 1
                    self.infeasible = True
                    u = [np.hstack([-5*np.ones((self.N, 1)), np.zeros((self.N,1))])]
                    current_state = Sim.ev.traj[:,Sim.t]
                    Sim.step(np.array([-5.0, 0]))
                    Sim.infeas_status.append(True)
                    next_state = Sim.ev.traj[:,Sim.t]

                    self.prediction_cache[agent_id] = next_states_pred
                    self.control_cache[agent_id].append(u[0][0,:])
                    self.current_state[agent_id] = next_state
                    self.state_cache[agent_id].append(next_state)
                    
                else:
                    # current_state = np.array(self.current_state[agent_id])
                    # next_state = self.shift_movement(current_state, u[0], self.f_np)
                    Sim.infeas_status.append(False)
                    Sim.step(u[0][0,:])
                    next_state = Sim.ev.traj[:,Sim.t]
                    self.prediction_cache[agent_id] = next_states_pred
                    self.control_cache[agent_id].append(u[0][0,:])
                    self.current_state[agent_id] = next_state
                    self.state_cache[agent_id].append(next_state)
                    p_collision  = Sim._get_collision_probability(
                        ev_global_trajectories,
                        update_dict['o_glob'], update_dict['global_covs'], update_dict['Qs'], update_dict['mode_probabilities'], update_dict['clusters']
                    )
                    collision_probability_traj.append(p_collision)
                
                print("Agent state: ", Sim.ev.traj[:,Sim.t], " Agent control: ", u[0].T)
                print("Agent pos: ", Sim.ev.traj_glob[:,Sim.t-1])
                # print("TV pos: ", Sim.tvs[0].traj_glob[:, Sim.t-1])
                # print("Ped pos: ", Sim.peds[0].traj_glob[:, Sim.t-1])
                # print(f"")
            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)

        if Sim.done():
            print("Executed solution is GOOD!")
            self.max_comp_time = max(self.avg_comp_time)
            self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            # self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
            self.success = True
            self.feedback_gain_avg = 0#compute_average_norm(self.feedback_gains_cache)
        else:
            self.success = False
            self.max_comp_time = max(self.avg_comp_time)
            self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            # self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
        
        run_description = self.scenario 
        print(f"Num infeas: {self.infeasible_count}")
        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.control_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached, self.infeasible_count, self.feedback_gain_avg, self.uncontrolled_fleet_data, self.num_timestep)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)
        