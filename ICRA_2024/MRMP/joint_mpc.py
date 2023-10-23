import casadi as ca
import numpy as np
import time
from models import DiffDrive
from draw import Draw_MPC_point_stabilization_v1
import multiprocessing as mp
from utils import *
from metrics_logger import MetricsLogger

class Joint_MPC:
    def __init__(self, initial_state, final_state, cost_func_params, obs, mpc_params, scenario, trial, ref=None, map=None):
        self.num_agent = mpc_params['num_agents']
        self.dt = mpc_params['dt']
        self.N = mpc_params['N']
        self.rob_dia = mpc_params['rob_dia']
        self.v_lim = mpc_params['v_lim']
        self.omega_lim = mpc_params['omega_lim']
        self.total_sim_timestep = mpc_params['total_sim_timestep']
        self.goal_tolerence = mpc_params['goal_tolerence']
        self.epsilon_o = mpc_params['epsilon_o']
        self.epsilon_r = mpc_params['epsilon_r']
        self.safety_margin = mpc_params['safety_margin']
        self.initial_state = initial_state[:self.num_agent]
        self.final_state = final_state[:self.num_agent]
        self.cost_func_params = cost_func_params
        self.scenario = scenario
        self.trial = trial

        self.model = DiffDrive(self.rob_dia)

        self.state_cache = []
        self.prediction_cache = []
        self.control_cache = []
        
        self.current_state = self.initial_state

        self.obs = obs
        self.dyn_obs = obs["dynamic"]
        self.static_obs = obs["static"]

        self.ref = ref
        self.map = map

        self.num_timestep = 0

        # check for failures after simulation is done
        max_time_reached = False
        execution_collision = False

        # metrics for logging
        self.algorithm_name = ""
        self.trial_num = 0
        self.avg_comp_time = []
        self.max_comp_time = 0.0
        self.traj_length = 0.0
        self.makespan = 0.0
        self.avg_rob_dist = 0.0
        self.c_avg = []
        self.success = False

        self.execution_collision = False
        self.max_time_reached = False

        self.logger = MetricsLogger()

          # variables holding previous solutions
        self.prev_states = np.zeros((self.N+1, 3*self.num_agent)) 
        self.prev_controls = np.zeros((self.N, 2*self.num_agent)) 
        self.prev_epsilon_o = np.zeros((self.N+1, self.num_agent))
        self.prev_epsilon_r = np.zeros((self.N+1, self.num_agent))

    def shift_movement(self, x0, u, x_n, f):
        next_state = np.zeros((self.num_agent, 3))
        for agent in range(self.num_agent):
            f_value = f(x0[agent,:], u[0, agent*2:agent*2+2])
            next_state[agent,:] = x0[agent,:] + self.dt*f_value
    
        return next_state

    def prediction_state(self, x0, u, dt, N):
        # define predition horizon function
        states = np.zeros((N+1, 3))
        states[0, :] = x0
        for i in range(N):
            states[i+1, 0] = states[i, 0] + u[i, 0] * np.cos(states[i, 2]) * dt
            states[i+1, 1] = states[i, 1] + u[i, 0] * np.sin(states[i, 2]) * dt
            states[i+1, 2] = states[i, 2] + u[i, 1] * dt
        return states

    # create model
    def f(self, x_, u_): return ca.vertcat(
        *[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

    def f_np(self, x_, u_): return np.array(
        [u_[0]*np.cos(x_[2]), u_[0]*np.sin(x_[2]), u_[1]])
    
    def are_all_agents_arrived(self):
        print(self.current_state)
        print(self.final_state)
        print("------")
        if(np.linalg.norm(self.current_state-self.final_state) > self.goal_tolerence):
            return False
        return True
        
    def check_for_collisions(self, state_cache):
        for i in range(self.num_agent):
            for j in range(i + 1, self.num_agent):
                traj_1 = state_cache[i]
                traj_2 = state_cache[j]
                min_length = min(len(traj_1), len(traj_2))
                
                for k in range(min_length):
                    wp_1 = traj_1[k]
                    wp_2 = traj_2[k]
                    distance = math.sqrt((wp_1[0] - wp_2[0])**2 + (wp_1[1] - wp_2[1])**2)
                    
                    if distance < self.rob_dia:
                        print("COLLISION")
                        return True
        return False
    
    def is_solution_valid(self, state_cache):
        if self.check_for_collisions(state_cache):
            print("Executed trajectory has collisions")
            self.execution_collision = True
            return False
        elif self.num_timestep == self.total_sim_timestep:
            print("Maximum time is reached")
            self.max_time_reached = True
            return False
        else: 
            return True

    def run_joint_mpc(self):
        # casadi parameters
        opti = ca.Opti()

        opt_states = opti.variable(self.N + 1, 3*self.num_agent)
        opt_x = opt_states[:, ::3]
        opt_y = opt_states[:, 1::3]

        opt_controls = opti.variable(self.N, 2*self.num_agent)
        v = opt_controls[:, ::2]
        omega = opt_controls[:, 1::2]
        
        # opt_epsilon_o = opti.variable(self.N+1, self.num_agent)

        opti.set_initial(opt_states, self.prev_states)
        opti.set_initial(opt_controls, self.prev_controls)

        # parameters
        opt_x0 = opti.parameter(3*self.num_agent)
        opt_xs = opti.parameter(3*self.num_agent)

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)

        for j in range(self.N):
            for agent in range(self.num_agent):
                start_idx = agent * 3
                end_idx = start_idx + 3

                if end_idx > opt_states.shape[1]:
                    break

                agent_states = opt_states[j, start_idx:end_idx]
                agent_controls = opt_controls[j, agent*2:agent*2+2]
        
                x_next = agent_states + self.f(agent_states, agent_controls).T * self.dt

                opti.subject_to(opt_states[j+1, start_idx:end_idx] == x_next)
            
                # opti.subject_to(opti.bounded(0, opt_epsilon_o[j, agent], ca.inf))

        # define the cost function
        robot_cost = 0  # cost
        collision_cost = 0
        total_cost = 0
            
        Q = self.cost_func_params['Q']
        R = self.cost_func_params['R']
        P = self.cost_func_params['P']

        for k in range(self.N):
            for agent in range(self.num_agent):
                start_idx = agent * 3
                end_idx = start_idx + 3

                agent_states = opt_states[k, start_idx:end_idx]
                agent_controls = opt_controls[k, agent*2:agent*2+2]
                agent_opt_xs = opt_xs[start_idx:end_idx]
                # agent_opt_epsilon_o = opt_epsilon_o[k, agent]
                
                state_diff = agent_states - agent_opt_xs.T
                
                control_cost = ca.mtimes([agent_controls, R, agent_controls.T])
                state_cost = ca.mtimes([state_diff, Q, state_diff.T])

                # epsilon_cost = 100000 * np.dot(agent_opt_epsilon_o, agent_opt_epsilon_o.T)
                
                robot_cost += state_cost + control_cost

                total_cost = robot_cost 
        
        opti.minimize(total_cost)

        # boundrary and control conditions
        opti.subject_to(opti.bounded(-12.0, opt_x, 12.0))
        opti.subject_to(opti.bounded(-12.0, opt_y, 12.0))
        opti.subject_to(opti.bounded(-self.v_lim, v, self.v_lim))
        opti.subject_to(opti.bounded(-self.omega_lim, omega, self.omega_lim))        

        # static obstacle constraint
        # for obs in self.static_obs:
        #     for agent in range(self.num_agent):
        #         obs_x = obs[0]
        #         obs_y = obs[1]
        #         obs_dia = obs[2]
        #         start = agent * 3
        #         end = start + 3

        #         for l in range(self.N+1):
        #             state = opt_states[l, start:end]
        #             print(state.shape)
        #             rob_obs_constraints_ = ca.sqrt((state[0]-obs_x)**2+(state[1]-obs_y)**2)-self.rob_dia/2.0-obs_dia/2.0 
        #             opti.subject_to(opti.bounded(0.0, rob_obs_constraints_, ca.inf))

        # for k in range(self.N):
        #     for agent1 in range(self.num_agent):
        #         for agent2 in range(agent1 + 1, self.num_agent):
        #             start_1 = agent1 * 3
        #             end_1 = start_1 + 3

        #             start_2 = agent2 * 3
        #             end_2 = start_2 + 3

        #             state_1 = opt_states[k, start_1:end_1]
        #             state_2 = opt_states[k, start_2:end_2]

        #             # Calculate the Euclidean distance between the positions (x_i, y_i) and (x_j, y_j)
        #             rob_rob_constraints_ = ca.sqrt((state_1[:,0] - state_2[:,0])**2 + (state_1[:,1] - state_2[:,1])**2) - self.rob_dia
                    
        #             # Add the constraint to the optimization problem
        #             opti.subject_to(opti.bounded(0.0, rob_rob_constraints_, ca.inf))

        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 1, 'print_time': 0,
                            'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6, 'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-9,
                            'ipopt.warm_start_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_frac': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_slack_bound_push': 1e-9, 'ipopt.warm_start_mult_bound_push': 1e-9}

        opti.solver('ipopt', opts_setting)

        for agent in range(self.num_agent):
            opti.set_value(opt_xs[agent*3:agent*3+3], self.final_state[agent])
            opti.set_value(opt_x0[agent*3:agent*3+3], self.current_state[agent])

        t_ = time.time()
        sol = opti.solve()
        solve_time = time.time() - t_
        print("Joint" + " Solve Time: " + str(solve_time))

        # obtain the control input
        u_res = sol.value(opt_controls)
        next_states_pred = sol.value(opt_states)
        # epsilon_r = sol.value(opt_epsilon_r)

        self.prev_states = next_states_pred
        self.prev_controls = u_res
  
        return u_res, next_states_pred


    def simulate(self):
        # parallelized implementation
        agent_list = list(range(self.num_agent))
        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}

        while (not self.are_all_agents_arrived() and self.num_timestep < self.total_sim_timestep):
            time_1 = time.time()
            print(self.num_timestep)
                
            u, next_states_pred = self.run_joint_mpc()
            # Next state dimensions are num_agent x num states
            # Next state predictions are horizon len x (num_agent x num states)
            next_state = self.shift_movement(self.current_state, u, next_states_pred, self.f_np)

            self.prediction_cache = next_states_pred
            self.control_cache = u
            self.current_state = next_state

            for agent_id in range(self.num_agent):
                self.state_cache[agent_id].append(self.current_state[agent_id,:])
            
            self.num_timestep += 1
            time_2 = time.time()
            self.avg_comp_time.append(time_2-time_1)

        if self.is_solution_valid(self.state_cache):
            print("Executed solution is GOOD!")
            self.max_comp_time = max(self.avg_comp_time)
            self.avg_comp_time = (sum(self.avg_comp_time) / len(self.avg_comp_time)) / self.num_agent
            self.traj_length = get_traj_length(self.state_cache)
            self.makespan = self.num_timestep * self.dt
            self.avg_rob_dist = get_avg_rob_dist(self.state_cache)
            self.success = True
        else:
            self.success = False
        
        run_description = "Joint_MPC_" + self.scenario 

        self.logger.log_metrics(run_description, self.trial, self.state_cache, self.map, self.initial_state, self.final_state, self.avg_comp_time, self.max_comp_time, self.traj_length, self.makespan, self.avg_rob_dist, self.c_avg, self.success, self.execution_collision, self.max_time_reached)
        self.logger.print_metrics_summary()
        self.logger.save_metrics_data()
        
        # draw function
        draw_result = Draw_MPC_point_stabilization_v1(
            rob_dia=self.rob_dia, init_state=self.initial_state, target_state=self.final_state, robot_states=self.state_cache, obs_state=self.obs, map=self.map)