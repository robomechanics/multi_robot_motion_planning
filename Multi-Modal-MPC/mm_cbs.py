import numpy as np

class MM_Node:
    def __init__(self, mm_sol:dict, conflicts:list, resolved: dict):
        '''
        mm_sol :  multi-modal trajectories that resolve the conflicts in resolved_conflicts
        resolved: branch id -> List of obstacle mode tuples indicating which obstacles and their modes are resolved
        
        EXAMPLE: 2 obstacles, 2 modes each. The binary number b_11 b_12 b_21 b_22 captures which obstacles 
        '''
        self.mm_sol = mm_sol
        self.conflicts = conflicts
        
        
        
        
