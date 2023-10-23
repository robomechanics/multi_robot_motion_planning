import numpy as np
import yaml
import copy
import math
class Map:
    def __init__(self, *args):
        self.diameter = 1.0
        if len(args)==3:
          self.num_agents = args[0]
          self.map_dim = args[1]
          self.obstacle_density = args[2]
          self.map = {'dimensions':[self.map_dim,self.map_dim],'obstacles':[]}
          self.agents = []
          self.obstacles = []
          self.inflated_obstacles = []
          self.place_obstacles()
          self.inflate_obstacles()
          self.create_start_goal()

        else:
          print("Map initialization")       
          # Read Map
          with open(args[0], 'r') as map_file:
            try:
              map = yaml.load(map_file, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
              print(exc) 
          self.map = map["map"]
          self.num_agents = len(map["agents"])
          self.map_dim = map["map"]["dimensions"][0]
          self.obstacles = map["map"]["obstacles"]
          self.inflated_obstacles = []
          self.agents = []
          self.inflate_obstacles()
          if map_file.name == 'warehouse.yaml':
            self.create_start_goal()
          else:
            self.agents = map["agents"]

    def place_obstacles(self):
        num_obstacles = int(self.map_dim ** 2 * self.obstacle_density/100)
        obstacle = 0
        while obstacle<num_obstacles:
            index = (np.random.randint(self.diameter, self.map_dim - self.diameter),np.random.randint(self.diameter, self.map_dim - self.diameter))
            if index not in self.map['obstacles']:
                self.map['obstacles'].append(index)
                self.obstacles.append(index)
                obstacle = obstacle + 1

    def create_start_goal(self):
        starts = []
        goals = []
        numAgent = 0
        while numAgent<self.num_agents:
            start = (np.random.randint(self.diameter, self.map_dim - self.diameter),np.random.randint(self.diameter, self.map_dim - self.diameter), 0.0)
            if (start not in self.inflated_obstacles) and (start not in starts):
              flag = True
              if len(starts) > 0:
                for i in range(len(starts)):
                  if math.dist(start, starts[i]) < self.diameter:
                    flag = False
                    break
              if flag == True:
                starts.append(start)
                numAgent = numAgent + 1

        numAgent = 0
        while numAgent<self.num_agents:
            goal = (np.random.randint(self.diameter ,self.map_dim - self.diameter),np.random.randint(self.diameter, self.map_dim - self.diameter), 0.0)
            if (goal not in self.inflated_obstacles) and (goal not in goals) and goal!=starts[numAgent]:
              flag = True
              if len(goals) > 0:
                for i in range(len(goals)):
                  if math.dist(goal, goals[i]) < self.diameter:
                    flag = False
                    break
              if flag == True:
                goals.append(goal)
                numAgent = numAgent + 1

        numAgent = 0
        while numAgent<self.num_agents:
            agent = {'start':starts[numAgent],'goal':goals[numAgent],'name':'agent'+str(numAgent)}
            self.agents.append(agent)
            numAgent = numAgent + 1

    def __str__(self):
        return f"Map with {self.num_agents} agents, {self.map_dim}x{self.map_dim} dimensions, and {self.obstacle_density:.2%} obstacle density."

    def inflate_obstacles(self):
      d = math.ceil(self.diameter)
      x_options = [-d,-d,-d,0,0,0,d,d,d]
      y_options = [-d,0,d,-d,0,d,-d,0,d]
      for obstacle in self.obstacles:
        x, y = obstacle[0], obstacle[1]
        for i in range(len(x_options)):
          x_new = x + x_options[i]
          y_new = y + y_options[i]
          if(x_new >= 0 and x_new < self.map_dim and y_new >= 0 and y_new < self.map_dim):
            if (x_new,y_new) not in self.inflated_obstacles:
              self.inflated_obstacles.append((x_new,y_new))
      

                            

