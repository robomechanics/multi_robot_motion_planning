import math
import random 

class Task_Generator:
    def __init__(self, num_agents, map, rob_clearance):
        self.num_agents = num_agents
        self.map = map
        self.clearance = 0.5

    def generate_tasks(self):
        initial_states = []
        final_states = []
        grid_height = len(self.map)
        grid_width = len(self.map[0])

        def is_free_space(x, y):
            return 0 <= x < grid_width and 0 <= y < grid_height and self.map[y][x] == 0

        while len(initial_states) < self.num_agents:
            x_initial = random.randint(0, grid_width - 1)
            y_initial = random.randint(0, grid_height - 1)
            
            if all(is_free_space(x_initial + dx, y_initial + dy) for dx in range(-int(self.clearance), int(self.clearance) + 1)
                                                                for dy in range(-int(self.clearance), int(self.clearance) + 1)):
                x_final = random.randint(0, grid_width - 1)
                y_final = random.randint(0, grid_height - 1)
                
                if all(is_free_space(x_final + dx, y_final + dy) for dx in range(-int(self.clearance), int(self.clearance) + 1)
                                                                for dy in range(-int(self.clearance), int(self.clearance) + 1)):
                    valid_initial = True
                    for initial_state, final_state in zip(initial_states, final_states):
                        other_x_initial, other_y_initial, _ = initial_state
                        other_x_final, other_y_final, _ = final_state
                        
                        if (x_initial, y_initial) == (other_x_initial, other_y_initial) or (x_initial, y_initial) == (other_x_final, other_y_final):
                            valid_initial = False
                            break
                    
                    if valid_initial and math.sqrt((x_final - x_initial)**2 + (y_final - y_initial)**2) >= self.clearance:
                        initial_orientation = random.uniform(0, 2 * math.pi)  # Random initial orientation in radians
                        final_orientation = random.uniform(0, 2 * math.pi)    # Random final orientation in radians
                        
                        initial_state = [x_initial, y_initial, initial_orientation]
                        final_state = [x_final, y_final, final_orientation]
                        
                        initial_states.append(initial_state)
                        final_states.append(final_state)

        return initial_states, final_states

    