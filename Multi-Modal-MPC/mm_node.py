import numpy as np

class MM_Node:
    def __init__(self, node_id:int, mm_sol:dict, conflicts:list, resolved: dict, constraints : dict, collision_prob:float=1.0):
        '''
        mm_sol :  multi-modal trajectories that resolve the conflicts in resolved_conflicts
        resolved: branch id -> List of obstacle mode tuples indicating which obstacles and their modes are resolved
        
        EXAMPLE: 2 obstacles, 2 modes each. The binary number b_11 b_12 b_21 b_22 captures which obstacles 
        '''
        self.node_id =node_id
        self.mm_sol = mm_sol
        self.conflicts = conflicts
        self.resolved = resolved
        self.constraints = constraints
        self.collision_prob = collision_prob
        
    
        
    def get_conflicts(self):
            return NotImplementedError
        
    def update_node(self, new_conflicts, new_resolved):
        return NotImplementedError
    
    def _make_branch_id(self):
        return NotImplementedError
    
    def is_conflict_free(self):
        
        return len(self.conflicts)==0
        
        
        
        
        
        
        
