import casadi as ca
import numpy as np
import time
from mpc_base import MPC_Base
import multiprocessing as mp
from utils import *
import scipy.special as sp
from scipy.stats import multivariate_normal
import pdb
from mm_node import MM_Node
import math as m
from itertools import product

class MM_CBS(MPC_Base):
    
    def _collision_check(self, agent_id, rob_sol, obs_id, obs_pred, obs_constraints):
        
        modes_set = set(range(self.num_modes))-set(obs_constraints[obs_id])
        
        N_samples = 50
        
        N_t = multivariate_normal.rvs(np.zeros(self.N), np.eye(self.N), N_samples)
        
        collision_prob = 0.
        num_collisions = [0 for _ in range(self.num_modes)]
        min_distance = self.rob_dia*2+0.5
        
        eps = 0.01
        
        obs_set = set(c for c in obs_constraints.keys())
        n_obs = len(obs_set)
                
        current_state_obs = obs_pred[obs_id]['current_state']
        mode_prob = obs_pred[obs_id]['mode_probs']
        other_modes_prob = { k : obs_pred[k]['mode_probs'] for k in obs_set if k!=obs_id}
        
        
       

        n_modes = {obs_idx: [] for obs_idx in obs_set}
        
        for obs_id, modes in obs_constraints.items():
            n_modes[obs_id] = modes
        conflicts, resolved = [], []
        time1= time.time()
    
        for mode in modes_set:   # these are modes that are NOT in constraints_list for obs_id
            obs_mean = (self.obs_affine_model[obs_id][mode]['T']@current_state_obs[:2]).reshape((-1,1)) + self.obs_affine_model[obs_id][mode]['c']
            # rob_mean = ca.vec(self.prev_states[agent_id][mode].T)
            rob_mean = ca.vec(rob_sol["states"][mode].T)
            
            
            if obs_constraints[obs_id] == []:   # if this obstacle 
                feedback_gains = ca.DM(self.N*2, self.N*2)
                for sample in range(N_samples):
                    obs_dist = obs_mean + self.obs_affine_model[obs_id][mode]['E']@N_t[sample,:].reshape((-1,1))
        
                    rob_dist = rob_mean + self.rob_affine_model['B'].toarray()@feedback_gains@(obs_dist[:-2]-obs_mean[:-2])
                    
                    rob_pos_dist = ca.vec(rob_dist.reshape((-1,self.N+1))[:2,:])

                    num_collisions[mode]+=np.linalg.norm(rob_pos_dist-obs_dist)<=min_distance*self.N
                    
                
                if num_collisions[mode] > self.delta:
                    conflicts.append([agent_id, obs_id, mode])
                else:
                    resolved.append([agent_id, obs_id, mode])
            else:
                for o_mode, feedback_gains in rob_sol['pol_gains'][obs_id].items():
                    # Do collision check below. IF it didnt pass, try with anoteher gain
                    feedback_bias = rob_sol['pol_bias'][obs_id][o_mode]
                    other_obs_modes = {o_idx: m for o_idx, m in n_modes.items() if o_idx!=obs_id}
                    
                    obs_mode_combinations = list(product(o_mode,*[other_obs_modes[obs] for obs in other_obs_modes.keys()]))
                    # Sort combinations by probabiliity, and pick top qth percentile
                    obs_list = [obs_id]+list(other_obs_modes.keys())
                    
                    num_collisions[mode] = 0
                    for combination in obs_mode_combinations:
                        # do collision check for all possible high prob mode combinations of other obstacles to define pol = h + h_obs+K_obs (o_obs) +  h_i + K_i o_i
                
                        for sample in range(N_samples):
                            obs_dist = obs_mean + self.obs_affine_model[obs_id][mode]['E']@N_t[sample,:].reshape((-1,1))
                            
                            # this is h_obs + K_obs@ (o_obs - E[o_obs])
                            feedback_term = feedback_gains@(obs_dist[:-2]-obs_mean[:-2]) + feedback_bias
                            
                            # this loop computes the rest for the policy
                            for i in range(1, len(combination)):
                                obs_mean = (self.obs_affine_model[i][combination[i]]['T']@current_state_obs[:2]).reshape((-1,1)) + self.obs_affine_model[i][combination[i]]['c']
                                obs_dist = obs_mean + self.obs_affine_model[i][combination[i]]['E']@N_t[sample,:].reshape((-1,1))
                                feedback_term += rob_sol['pol_bias'][i][combination[i]]    +    rob_sol['pol_gains'][i][combination[i]]@(obs_dist[:-2]-obs_mean[:-2])
                
                            rob_dist = rob_mean + self.rob_affine_model['B'].toarray()@feedback_term
                            
                            rob_pos_dist = ca.vec(rob_dist.reshape((-1,self.N+1))[:2,:])
                            # rob_reach_dist = ca.vec(rob_dist_reach.reshape((-1,self.N+1))[:2,:])

                            num_collisions[mode]+=np.linalg.norm(rob_pos_dist-obs_dist)<=min_distance*self.N
                        
                    num_collisions[mode]=num_collisions[mode]/N_samples*mode_prob[mode]
                    if num_collisions[mode] <= self.delta:
                        rob_sol['pol_gains'][obs_id][mode] = feedback_gains
                        rob_sol['pol_bias'][obs_id][mode] = feedback_bias
                        resolved.append([agent_id, obs_id, mode])
                        break
                
                if mode not in rob_sol['pol_gains'].keys():
                    conflicts.append([agent_id, obs_id, mode])
                    
                

            #collision probability for mode
            
            
                
            collision_prob+=num_collisions[mode]
            
        time2=time.time()-time1
        
        return conflicts, resolved, collision_prob  
    
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

        A=[ca.DM(n_x, n_x) for _ in range(self.N+1)]
        B=[ca.DM(n_x, n_u) for _ in range(self.N+1)]
        C=[ca.DM(n_x, 1) for _ in range(self.N+1)]

        A_block=ca.DM(n_x*(self.N+1), n_x)
        B_block=ca.DM(n_x*(self.N+1), n_u*self.N)
        C_block=ca.DM(n_x*(self.N+1), 1)
        E_block=ca.DM(n_x*(self.N+1), n_x*self.N)

        A_block[0:n_x, 0:n_x]=ca.DM.eye(n_x)

        E=0.001*ca.DM.eye(n_x)  # Coefficients of Linearization error

        if x_lin is None:
            x_lin=ca.DM(self.N+1,3)
            u_lin=ca.horzcat(.2*ca.DM.ones(self.N,1), ca.DM(self.N,1))
            x_lin[0,:]=current_state.reshape((1,-1))
            for t in range(self.N):
                x_lin[t+1,:]=self.model.fCd(x_lin[t,:], u_lin[t,:])

        for t in range(self.N):
            # if u_lin[t,0] > 0 and u_lin[t,0] < 0.1 :
            #     u_lin[t,0]=0.1
            # elif u_lin[t,0] < 0 and u_lin[t,0]> -0.1:
            #     u_lin[t,0]=-0.1
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

    def run_single_mpc(self, agent_id, current_state, constraints, linearized_ca = False):
        '''
        constraints = [(obs_i, 1, 'new'), (obs_i, 2, 'old'), (obs_j, 0), (obs_j, 3), (obs_k, 0)] (as an eg.)
        
        '''
        # casadi parameters
        opti = ca.Opti()
        

        
        obs_set = set(c[0] for c in constraints)
        n_obs = len(obs_set)
        
        
        # for the example in docstring, n_modes = {obs_i = [1,2], obs_j=[0,3], obs_k= [0]}
        n_modes = {obs_idx: [] for obs_idx in obs_set}
        for constraint in constraints:
            n_modes[constraint[0]].append(constraint[1])        
        

        ####
        # EV feedforward + TV state feedback policies from https://arxiv.org/abs/2109.09792
        # U_stack[j] = opt_controls + sum_i=1^{N_obs} K_stack[i][j]@(O_stack[i][j] - E[O_stack[i][j]])
        #### 
        # nominal feedforward controls
        u_ff = opti.variable(self.N, 2)
        
        
        opt_controls_h = {obs_idx: [ca.vertcat(ca.DM(self.robust_horizon,2), opti.variable(self.N-self.robust_horizon,2)) for _ in range(n_modes[obs_idx])] for obs_idx in obs_set}
        
        # opt_epsilon_r = [opti.variable(self.N, 1) for _ in range(len(constraints))]
        # opt_epsilon_r = {obs_idx: [opti.variable(self.N,1) for _ in range(n_modes[obs_idx])] for obs_idx in obs_set}
        
        # other_obs_modes = {o_idx: m for o_idx, m in n_modes.items() if o_idx!=obs_idx}
                    
        obs_mode_combinations = list(product([n_modes[obs_idx] for obs_idx in n_modes.keys()]))
        # this collects h term in policy for each MODE COMBO. Eg., if mode combo is (0,1,1), then h term := u_ff + h[obs_0][0]+h[obs_1][1]+ h_[obs_2][1]
        
        if not constraints:
            opt_controls = [ca.vec(u_ff)]
            # mode_prob = {obs_idx: self.uncontrolled_fleet_data[obs_idx]['mode_probabilities'][self.num_timestep] for obs_idx in obs_set}
        
            mode_prob_combo = [ 1.]
        else:
            opt_controls   =     [   ca.vec(u_ff)+ca.vec(ca.sum2(ca.horzcat(*[ca.vec(opt_controls_h[obs_idx][combo[i]]) for i, obs_idx in enumerate(obs_set)]))) for n, combo in enumerate(obs_mode_combinations)]
            mode_prob = {obs_idx: self.uncontrolled_fleet_data[obs_idx]['mode_probabilities'][self.num_timestep] for obs_idx in obs_set}
        
            mode_prob_combo = [ ca.prod(*[mode_prob[obs_idx][combo[i]] for i, obs_idx in enumerate(obs_set)]) for n, combo in enumerate(obs_mode_combinations) ]

        # A_rob, B_rob, C_rob, E_rob = [], [], [], []
        opt_states, opt_x, opt_y, v, omega = [], [], [], [], []
        
        # TODO: just linearize using reference (from root node solve)
        A, B, C, E =  self.rob_affine_model['A'], self.rob_affine_model['B'], self.rob_affine_model['C'], self.rob_affine_model['E']
        for control in opt_controls:         
            
            # nominal state predictions
            opt_states.append(ca.vec(A@ca.DM(current_state)+B@control+C).reshape((-1,self.N+1)).T)

            opt_x.append(opt_states[-1][:,0])
            opt_y.append(opt_states[-1][:,1])
            v.append(control.reshape((-1,self.N)).T[:,0])
            omega.append(control.reshape((-1,self.N)).T[:,1])

                
        
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

       
        # opt_eps_combo   = [ for ]
        for i, (opt_states_branch, opt_controls_branch) in enumerate(zip(opt_states, opt_controls)):
            opt_controls_branch = opt_controls_branch.reshape((-1, self.N)).T
            for k in range(self.N):
                mode_weight = mode_prob_combo[i]
 
                robot_cost = robot_cost + mode_weight*(ca.mtimes([(opt_states_branch[k, :]-opt_xs.T), Q, (opt_states_branch[k, :]-opt_xs.T).T] 
                            )+ ca.mtimes([opt_controls_branch[k, :], R, opt_controls_branch[k, :].T]) ) #+ 100000 * opt_epsilon_o[k] 
            
                for obs in self.static_obs:
                    obs_x = obs[0]
                    obs_y = obs[1]
                    obs_dia = obs[2]
                    
                    rob_obs_constraints_ = ca.sqrt((opt_states_branch[k, 0]-obs_x)**2+(opt_states_branch[k, 1]-obs_y)**2)-obs_dia/2 - self.rob_dia/2 - self.safety_margin #+ opt_epsilon_o[l]
                    opti.subject_to(rob_obs_constraints_ >= 0)
            
            # boundrary and control conditions
            opti.subject_to(opti.bounded(-10.0, opt_x[i], 10.0))
            opti.subject_to(opti.bounded(-5, opt_y[i], 5))
            opti.subject_to(opti.bounded(-self.v_lim, v[i], self.v_lim))
            opti.subject_to(opti.bounded(-self.omega_lim, omega[i], self.omega_lim))
            # static obstacle constraint
            
        total_cost = robot_cost + collision_cost
        
        ##### Get chance constraints from the given GMM prediction
        ## aij = (pi - pj) / ||pi - pj|| and bij = ri + rj 
        ## aij^T(pi - pj) - bij >= erf^-1(1 - 2delta)sqrt(2*aij^T(sigma_i + sigma_j)aij)    
    
        if self.feedback:
            K_rob_horizon = {obs_idx: [opti.variable(2,2) for t in range(self.robust_horizon-1)] for obs_idx in obs_set} 
        else:
            K_rob_horizon = {obs_idx: [ca.DM(2,2) for t in range(self.robust_horizon-1)] for obs_idx in obs_set}
     
        # Creatig pol_gains
        pol_gains ={obs_idx: {j: None for j in n_modes[obs_idx]} for obs_idx in obs_set}
         
        for obs_idx, modes in n_modes.items():  
            for j in range(modes):
                if self.feedback:
                    K = K_rob_horizon+[opti.variable(2,2) for t in range(self.N-self.robust_horizon)]
                else: 
                    K = K_rob_horizon+[ca.DM(2,2) for t in range(self.N-self.robust_horizon)] 
                
                K_stack = ca.diagcat(ca.DM(2,2),*[K[t] for t in range(self.N-1)])
                
                obs_xy_cov = self.obs_affine_model[obs_idx][j]['covars']
     
                total_cost+= ca.trace((K_stack@obs_xy_cov@obs_xy_cov.T@K_stack.T))
                
                pol_gains[obs_idx][j]=K_stack
                
        
        for obs_idx, modes in n_modes.items():
            for j in modes:
                prediction = self.gmm_predictions[obs_idx][j]
                # opti.subject_to(opti.bounded(0, opt_epsilon_r[obs_idx][j][:], 0.001))
                for t in range(1,self.N):
                        
                    obs_pos   = ca.DM(prediction['means'][t-1][:2])
                    
                    ref_pos = ca.DM(current_state)[:2]
                    rob_proj = obs_pos+2*self.rob_dia*(ref_pos-obs_pos)/ca.norm_2(ref_pos-obs_pos)
                    
                
                    
                    # other_obs_constraints = {obs_i:[0,1], obs_k:[2,1]}
                    #  robot _policy = h[0][j_0] + K[0][j_0]*(o0[j_0] -E[o0|j_0]) + h[1][j_1]+K[1][j_1]*(o1[j_1]-E[o1|j_1])
                    
                    # Collision_avoidance for o0 in j_0 :   ||Ax+B*policy  - o0[j0]|| > d_min   for all  modes of o1: [j_1, j'_1,......] (MM-mpc) / for all modes of o1 in constraints (MM_CBS)
                    
                    other_obs_modes = {o_idx: m for o_idx, m in n_modes.items() if o_idx!=obs_idx}
                    
                    obs_mode_combinations = list(product(j,*[other_obs_modes[obs] for obs in other_obs_modes.keys()]))
                    obs_list = [obs_idx]+list(other_obs_modes.keys())
                    for combination in obs_mode_combinations:
                        #combination = [j, one mode each of other obstacles in constraints list]
                        _2_norm_coeff =2*sp.erfinv(1-2*self.delta)*(rob_proj-obs_pos).T
                        rv_dist  =_2_norm_coeff@ca.horzcat(E[t*3:(t+1)*3-1,:],                                                        
                                                            *[B[t*3:(t+1)*3-1,:]@pol_gains[l][combination[i]]@self.obs_affine_model[l][combination[i]]['E'][:-2,:]\
                                                             -int(l==obs_idx)*self.obs_affine_model[obs_idx][j]['E'][t*2:(t+1)*2,:] for i,l in enumerate(obs_list)])

                        robot_ff_control = ca.sum2(ca.horzcat(*[ca.vec(opt_controls[l][combination[i]]) for i,l in enumerate(obs_list)]))
                        nom_robot_state = ca.vec(A@ca.DM(current_state)+B@ca.vec(robot_ff_control.T)+C).reshape((-1,self.N+1)).T
                        nom_dist = (rob_proj-obs_pos).T@(nom_robot_state[t,:2].T-rob_proj)

                        # opti.subject_to(rv_dist@rv_dist.T<=(opt_epsilon_r[j][t-1]+nom_dist)**2)
                        # opti.subject_to(nom_dist>=-opt_epsilon_r[obs_idx][j][t-1])
                        
                        opti.subject_to(rv_dist@rv_dist.T<=(nom_dist)**2)
                        opti.subject_to(nom_dist>=0)
                        
                        
                        # For Gurobi   
                        # opti.subject_to(ca.soc(rv_dist, nom_dist+opt_epsilon_r[obs_idx][j][t-1])>=0)
                            

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
                            'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
                            'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}
        # opts_setting = {'ipopt.print_level': 0, 'print_time': 0,}
        opti.minimize(total_cost)
        opti.solver('ipopt', opts_setting)
        # opti.solver("OSQP")
        opti.set_value(opt_xs, self.final_state[agent_id])

        # # set optimizing target withe init guess
        # for j in range(n_modes):
        #     if type(self.prev_controls[agent_id])!=type([]):
        #         opti.set_initial(opt_controls[j], self.prev_controls[agent_id])  # (N, 2)
        #     else:
        #         opti.set_initial(opt_controls[j], self.prev_controls[agent_id][j])

        u_res = None
        next_states_pred = None

        try:     
            # solve the optimization problem
            t_ = time.time()
            sol = opti.solve()
            solve_time = time.time() - t_
            print("Agent " + str(agent_id) + " Solve Time: " + str(solve_time))

            u_res = [ sol.value(control.reshape((-1,self.N)).T) for control in opt_controls]
            self.feedforward = sol.value(u_ff)
            for obs_idx, modes in n_modes.items():
                for mode in modes:
                    self.feedback_gains[obs_idx][mode] = sol.value(pol_gains[obs_idx][mode]).toarray()
                    self.feedback_bias[obs_idx][mode]  = sol.value(opt_controls_h[obs_idx][mode]).toarray()
                   
             
            
                       

            # obtain the control input
            
            # next_states_pred = sol.value(opt_states)
            next_states_pred = [[ca.DM(current_state).T] for j in range(1)]

            for j in range(1):
                for t in range(u_res[0].shape[0]):
                    next_states_pred[j].append(self.model.fCd(next_states_pred[j][-1], u_res[j][t,:]).T)
                next_states_pred[j] = ca.vertcat(*next_states_pred[j])
            # eps_o = sol.value(opt_epsilon_o)
            
            
            self.prev_states[agent_id] = next_states_pred
            self.prev_controls[agent_id] = u_res
            self.prev_pol = pol_gains
            
            # self.prev_epsilon_o[agent_id] = eps_o 
        
        except RuntimeError as e:
            print("Infeasible solve")
  
        return u_res, next_states_pred
    
    def simulate(self, incremental=True):
        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1, self.num_modes)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N, self.num_modes)) for agent_id in range(self.num_agent)}

        self.setup_visualization()
        # self.setup_visualization_heatmap()
        
        
        # parallelized implementation
        while (not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
            time_1 = time.time()
            print(self.num_timestep)
    
            # Create a multiprocessing pool
            pool = mp.Pool()
            
            uncontrolled_traj = [self.uncontrolled_fleet_data[obs_idx]['executed_traj'] for obs_idx in range(self.n_obs)]
            current_uncontrolled_state = [uncontrolled_traj[obs_idx][self.num_timestep] for obs_idx in range(self.n_obs)]
            self.gmm_predictions = self.uncontrolled_fleet.get_gmm_predictions_from_current(current_uncontrolled_state)
            noise_chars = self.uncontrolled_fleet.get_gmm_predictions()
            mode_prob = [self.uncontrolled_fleet_data[obs_idx]['mode_probabilities'][self.num_timestep] for obs_idx in range(self.n_obs)]
            #  mode_prob_eg = [(0.9, 0.1), (0.5, 0.5)]
            
            obs_mode_combinations = list(product(*[list(range(self.num_modes)) for obs in range(self.n_obs)]))
            
            probs_combinations_unsorted = [ m.prod([mode_prob[k][mode_conf[k]] for k in range(self.n_obs)])  for mode_conf in obs_mode_combinations]
            mode_prob_combinations = sorted(enumerate(probs_combinations_unsorted), key=lambda x: -x[1])    # mode_conf = [1,1] ==> [0.1, 0.5]
            
            
            # [[0,0]:0.45, [0,1]:0.45, [1,0]:0.05, [1,1]:0.05] ====> [[0,0], [0,1]]  ===> 0.9
            n_mode_confs = 1
            prob_sum = mode_prob_combinations[0][1]
            
            mode_conf_idx =[mode_prob_combinations[0][0]]
            for i in range(1,len(obs_mode_combinations)):
                if prob_sum >= self.prob_thresh:
                    break
                else:
                    prob_sum +=mode_prob_combinations[i][1]
                    n_mode_confs += 1
                    mode_conf_idx.append(mode_prob_combinations[i][0])
      
            self.obs_affine_model ={obs_idx: {mode: {'T': None, 'c':None, 'E':None, 'covars':None} for mode in range(len(self.gmm_predictions[obs_idx]))} for obs_idx in range(self.n_obs)}
            
            for obs_idx, (agent_prediction, agent_noise) in enumerate(zip(self.gmm_predictions, noise_chars)):
                for mode, prediction in agent_prediction.items():
                    mean_traj = prediction['means']
                    covariances = prediction['covariances']
                    mean_inputs  = agent_noise[mode]['means']
                    covar_inputs = agent_noise[mode]['covariances']
                    
                    obs_xy_cov = ca.diagcat(*[ covariances[i][:2,:2] for i in range(self.N)])
                    T_o, c_o, E_o= self._get_obs_ATV_dynamics(mean_inputs, covar_inputs, mean_traj)
                    
                    self.obs_affine_model[obs_idx][mode].update({'T': T_o, 'c':c_o, 'E':E_o, 'covars':obs_xy_cov})
                
            A, B, C, E = self._get_robot_ATV_dynamics(np.array(self.current_state[0]))
            import pdb; pdb.set_trace()
            self.rob_affine_model ={'A': A, 'B':B, 'C':C, 'E':E}
    
            # Apply MPC solve to each agent in parallel
            self.reference_x, self.reference_u = None, None
            results = [self.run_single_mpc(0, np.array(self.current_state[0]), [])]
                  
            
            
            self.reference_x, self.reference_u =self.prev_states[0][0], self.prev_controls[0][0]
            A, B, C, E = self._get_robot_ATV_dynamics(np.array(self.current_state[0]), self.reference_x, self.reference_u)
            self.rob_affine_model ={'A': A, 'B':B, 'C':C, 'E':E}
            # results = pool.starmap(self.run_single_mpc, [(agent_id, np.array(self.current_state[agent_id]), []) for agent_id in range(self.num_agent)])
    
            pool.close()
            pool.join()

            # uncontrolled_traj = self.uncontrolled_fleet_data[0]['executed_traj']
            # current_uncontrolled_state = uncontrolled_traj[self.num_timestep]
            # gmm_predictions = self.uncontrolled_fleet.get_gmm_predictions_from_current(current_uncontrolled_state)

            # mode_prob = self.uncontrolled_fleet_data[0]['mode_probabilities'][self.num_timestep] 
            
            obs_pred = {obs_idx:
                {'predictions':self.uncontrolled_fleet.get_gmm_predictions_from_current(current_uncontrolled_state)[obs_idx],
                 'mode_probs':self.uncontrolled_fleet_data[obs_idx]['mode_probabilities'][self.num_timestep],
                 'current_state':current_uncontrolled_state[obs_idx]} for obs_idx in range(self.n_obs)}
            # self.plot_gmm_means_and_state(self.current_state[0], self.prediction_cache[0], self.gmm_predictions[0], mode_prob)
      
            # plt.plot(self._collision_check(0))
            # plt.show()
            # print("Collision Probs:",self._collision_check(0))
            # self.plot_feedback_gains()

            # Process the results and update the current state
            # for agent_id, result in enumerate(results):
            #     u, next_states_pred = result
            #     if u is None:
            #         self.infeasible_count += 1
            #         self.infeasible = True
            #         u = [np.zeros((self.N, 2))]
            #         current_state = np.array(self.current_state[agent_id])
            #         next_state = self.shift_movement(current_state, u[0], self.f_np)

            #         self.prediction_cache[agent_id] = next_states_pred
            #         self.control_cache[agent_id] = u
            #         self.current_state[agent_id] = next_state
            #         self.state_cache[agent_id].append(next_state)
            #     else:
            #         current_state = np.array(self.current_state[agent_id])
            #         next_state = self.shift_movement(current_state, u[0], self.f_np)

            #         self.prediction_cache[agent_id] = next_states_pred
            #         self.control_cache[agent_id] = u
            #         self.current_state[agent_id] = next_state
            #         self.state_cache[agent_id].append(next_state)
                    # print("Agent state: ", next_state, " Agent control: ", u[0,:])
            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)
            
            rob_sol ={'states': self.prev_states[0], 'controls': self.prev_controls[0], 'pol_gains': self.feedback_gains, "pol_bias": self.feedback_bias}
            
            obs_constraints = {obs_idx: [] for obs_idx in range(self.n_obs)}
            
            obs_id = 0
            conflicts, resolved, collision_prob = self._collision_check(0,  rob_sol, obs_id, obs_pred, obs_constraints)
            
            while len(conflicts)==0 and obs_id < self.n_obs:
                obs_id+=1
                conflicts, resolved, collision_prob = self._collision_check(0,  rob_sol, obs_id, obs_pred, obs_constraints)
            
            if len(conflicts)> 0:
                obs_constraints[obs_id].append([c[-1] for c in conflicts if c[1]==obs_id])
            
            MPC_constraints = {agent_id : []}
            node = MM_Node(0, {"states": self.prev_states[0], "controls":self.prev_controls[0], "pol_gains":self.feedback_gains, "pol_bias":self.feedback_bias},
                                conflicts,
                                resolved, MPC_constraints)   

            tree = [node]
            # tree[node.node_id]=node
            
            while True:
                node = tree.pop()
                
                conflicts, resolved =   node.conflicts, node.resolved
                
                if node.is_conflict_free():
                    print("Executed solution is GOOD!")
                    self.max_comp_time = max(self.avg_comp_time)
                    self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
                    # self.traj_length = get_traj_length(self.state_cache)
                    self.makespan = self.num_timestep * self.dt
                    self.success = True
                    
                    break

                
                self.success = False
                
                agent_id, mode, obstacle =  conflicts[-1]
                
                MPC_constraints = {agent_id: node.constraints[agent_id].append((obstacle, mode))}
                import pdb; pdb.set_trace()
                self.run_single_mpc(agent_id, MPC_constraints)
                
                rob_sol ={'states': self.prev_states[agent_id], 'controls': self.prev_controls[agent_id], 'pol_gains':self.feedback_gains}
                new_conflicts, new_resolved, collision_prob = self._collision_check(agent_id, rob_sol, 0, obs_pred)
                
                print(f"Collision probs: {collision_prob}")
                
                new_node = MM_Node(len(tree), rob_sol, new_conflicts, new_resolved, collision_prob)
                tree.append(new_node)
            
            
            # if node.is_conflict_free():
            #     print("Executed solution is GOOD!")
            #     self.max_comp_time = max(self.avg_comp_time)
            #     self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            #     # self.traj_length = get_traj_length(self.state_cache)
            #     self.makespan = self.num_timestep * self.dt
            #     self.success = True
            # else:
            #     self.success = False
        
        run_description = self.scenario 
        

        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached, self.infeasible_count, self.num_timestep)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # draw function
        # draw_result = Draw_MPC_point_stabilization_v1(
        #     rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs)
        