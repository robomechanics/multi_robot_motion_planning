   env = Environment(map, map_size, initial_states, final_states)
        cbs = CBS(env)
        solution = cbs.search()