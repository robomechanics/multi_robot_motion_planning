import casadi as ca
import numpy as np
from models import DiffDrive
import math 
import matplotlib.pyplot as plt
from metrics_logger import MetricsLogger

class MPC_Base:
    def __init__(self, initial_state, final_state, cost_func_params, obs, mpc_params, scenario, trial, map=None, ref=None):
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
        self.initial_state = initial_state
        self.final_state = final_state
        self.cost_func_params = cost_func_params
        self.scenario = scenario
        self.trial = trial

        self.model = DiffDrive(self.rob_dia)

        # rollout mpc params
        self.v_rollout_res = 0.2
        self.omega_rollout_res = 0.2

        self.state_cache = {agent_id: [] for agent_id in range(self.num_agent)}
        self.prediction_cache = {agent_id: np.empty((3, self.N+1)) for agent_id in range(self.num_agent)}
        self.control_cache = {agent_id: np.empty((2, self.N)) for agent_id in range(self.num_agent)}

        # variables holding previous solutions
        self.prev_states = {agent_id: np.zeros((self.N+1, 3)) for agent_id in range(self.num_agent)}
        self.prev_controls = {agent_id: np.zeros((self.N, 2)) for agent_id in range(self.num_agent)}
        self.prev_epsilon_o = {agent_id: np.zeros((self.N+1, 1)) for agent_id in range(self.num_agent)}
        self.prev_epsilon_r = {agent_id: np.zeros((self.N+1, 1)) for agent_id in range(self.num_agent)}
        
        self.current_state = {}
        for i in range(self.num_agent):
            self.current_state[i] = self.initial_state[i]

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

        self.logger = MetricsLogger()
    
    def shift_movement(self, x0, u, x_n, f):
        f_value = f(x0, u[0])
        st = x0 + self.dt*f_value
        u_end = np.concatenate((u[1:], u[-1:]))
        x_n = np.concatenate((x_n[1:], x_n[-1:]))

        return st, u_end, x_n

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
    
    def collision_cost(self, x0, x1):
        """
        Cost of collision between two robot_state
        """
        d = ca.norm_2(x0 - x1)
        Qc = self.cost_func_params['Qc']
        kappa = self.cost_func_params['kappa']
        cost = Qc / (1 + ca.exp(kappa * (d - 2 * (self.rob_dia))))

        return cost
    
    def find_collisions(self):
        collision_map = {i: [] for i in range(self.num_agent)}
        for i in range(self.num_agent):
            for j in range(i+1, self.num_agent):
                if i == j:
                    continue
                agent_1_traj = self.prediction_cache[i]
                agent_2_traj = self.prediction_cache[j]

                for index, (wp_1, wp_2) in enumerate(zip(agent_1_traj, agent_2_traj)):
                    distance = math.sqrt((wp_1[0] - wp_2[0])**2 + (wp_1[1] - wp_2[1])**2)
                    if distance < self.rob_dia:
                        print("Collision detected between " + str(i) + " and " + str(j) + " at index " + str(index))
                    
                        if (j, index) not in collision_map[i]:
                            collision_map[i].append((j, index))
        
                        if (i, index) not in collision_map[j]:
                            collision_map[j].append((i, index))
                        break
                        
        return collision_map
    
    def are_all_agents_arrived(self):
        for i in range(self.num_agent):
            current_state = np.array(self.current_state[i])
            final_state = np.array(self.final_state[i])
            print(np.linalg.norm(current_state-final_state))
            if(np.linalg.norm(current_state-final_state) > self.goal_tolerence):
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
        
    def find_closest_waypoint(self, waypoints, state):
        closest_distance = math.inf
        closest_index = None

        for i, waypoint in enumerate(waypoints):
            x_diff = waypoint['x'] - state[0]
            y_diff = waypoint['y'] - state[1]
            distance = math.sqrt(x_diff**2 + y_diff**2)

            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        return closest_index

    def extract_trajectory_segment(self, current_state):
        segment_dict = {}

        for robot_id, waypoints in self.ref.items():
            closest_index = self.find_closest_waypoint(waypoints, current_state)

            if closest_index is not None:
                segment_waypoints = []
                waypoints_len = len(waypoints)

                for i in range(closest_index, waypoints_len):
                    waypoint = waypoints[i]
                    segment_waypoints.append(waypoint)

                    if len(segment_waypoints) >= self.N:
                        break

                last_waypoint = segment_waypoints[-1]
                while len(segment_waypoints) < self.N:
                    segment_waypoints.append(last_waypoint)

                segment_dict[robot_id] = segment_waypoints

        return segment_dict

    def simulate(self):
        raise NotImplementedError("Subclasses must implement the functionality")