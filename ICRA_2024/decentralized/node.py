import math

class Node:
    def __init__(self, constraints=None, state_solution=None, control_solution=None):
        self.cost = 0

        self.state_solution = {}
        if state_solution:
            self.state_solution = state_solution
        
        self.control_solution = {}
        if control_solution:
            self.control_solution = control_solution
        
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = []

    def add_constraint(self, new_constraints):
        # self.constraints.clear()
        self.constraints.append(new_constraints)
    
    def update_solution(self, control_solution, state_solution, agent_id=None):
        if agent_id is not None:
            self.control_solution[agent_id] = control_solution
            self.state_solution[agent_id] = state_solution
        else:
            self.control_solution = control_solution
            self.state_solution = state_solution

    def update_cost(self, final_state):
        path_length = 0
        cost_to_go = 0
        for agent_id, plan in self.state_solution.items():
            for i in range(1, plan.shape[0]):   
                x1 = plan[i - 1][0]
                x2 = plan[i][0]
                y1 = plan[i - 1][1]
                y2 = plan[i][1]
                
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                path_length += distance

            # distance_to_goal = math.sqrt((plan[0][0] - final_state[agent_id][0])**2 + (plan[0][1] - final_state[agent_id][1])**2)
            # cost_to_go += distance_to_goal
            
        self.cost = cost_to_go
    

