import casadi as ca

# Define parameters
N = 10  # Prediction horizon
robust_horizon = 2  # Number of timesteps with branching
actions = ['cruise', 'left', 'right']
action_probs = [0.7, 0.15, 0.15]

# Define dynamics
def ego_dynamics(x, u):
    return x + u

def agent_dynamics(x, action):
    if action == 'cruise':
        return x
    elif action == 'left':
        return x - 1
    elif action == 'right':
        return x + 1

# Define optimization variables
x_ego = [ca.MX.sym(f'x_ego_{i}') for i in range(N+1)]
u_ego = [ca.MX.sym(f'u_ego_{i}') for i in range(N)]
x_agent_scenarios = [[ca.MX.sym(f'x_agent_{i}_{j}') for i in range(N+1)] for j in range(len(actions))]

# Set up cost and constraints
cost = 0
constraints = []

for t in range(N):
    # Ego dynamics and cost
    cost += (x_ego[t] - 5)**2  # Ego tries to stay close to position 5
    constraints.append(x_ego[t+1] == ego_dynamics(x_ego[t], u_ego[t]))
    
    for j, action in enumerate(actions):
        # Agent dynamics
        if t < robust_horizon:
            constraints.append(x_agent_scenarios[j][t+1] == agent_dynamics(x_agent_scenarios[j][t], action))
            cost += action_probs[j] * (x_ego[t] - x_agent_scenarios[j][t])**2  # Ego tries to avoid the agent
        else:
            # After robust horizon, assume agent cruises
            constraints.append(x_agent_scenarios[j][t+1] == agent_dynamics(x_agent_scenarios[j][t], 'cruise'))

# Create NLP
nlp = {'x': ca.vertcat(*x_ego, *u_ego, *[item for sublist in x_agent_scenarios for item in sublist]),
       'f': cost,
       'g': ca.vertcat(*constraints)}

opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve
initial_conditions = [0] * (N+1)  # Starting at position 0 for all vehicles
u0 = [0] * N
agent_initial_conditions = [[0]*(N+1) for _ in actions]
initial_guess = initial_conditions + u0 + [item for sublist in agent_initial_conditions for item in sublist]

res = solver(x0=initial_guess, lbg=0, ubg=0)

# Extract solution
solution = res['x'].full().flatten()
ego_trajectory = solution[:N+1]

print(ego_trajectory)
