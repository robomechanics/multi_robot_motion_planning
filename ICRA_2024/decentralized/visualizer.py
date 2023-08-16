import pickle 
import os 
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class visualizer:
    def __init__(self):

        self.num_agents = [2,3,4,5]

        self.directory = '/Users/ardalan/Desktop/project'  # replace with the path to your directory
        self.file_extension = '.pickle'  # replace with the file extension you want to filter for

        self.files = [filename for filename in os.listdir(self.directory) if filename.endswith(self.file_extension)]
        self.nmpc_comp_time_avg_all = []
        self.nmpc_no_ref_comp_time_avg_all = []
        self.nmpc_control_avg_all = []
        self.nmpc_no_ref_control_avg_all = []
        self.nmpc_control_std_all = []
        self.nmpc_no_ref_control_std_all = []

    def extract_data(self):
        vo_comp_time = defaultdict(list)
        nmpc_comp_time = defaultdict(list)
        nmpc_no_ref_comp_time = defaultdict(list)

        vo_solution_cost = defaultdict(list)
        nmpc_solution_cost = defaultdict(list)
        nmpc_no_ref_solution_cost = defaultdict(list)

        vo_control_std = defaultdict(list)
        nmpc_control_std = defaultdict(list)
        nmpc_no_ref_control_std = defaultdict(list)

        vo_control_avg = defaultdict(list)
        nmpc_control_avg = defaultdict(list)
        nmpc_no_ref_control_avg = defaultdict(list)

        for file in self.files:
            index_scenario = file.find("scenario_")
            scenario_num = file[index_scenario + len("scenario_")]

            index_num_agents = file.find("num_agent_")
            agent_num = int(file[index_num_agents + len("num_agent_")])

            with open(file, 'rb') as f:
                data = pickle.load(f)
                
                is_vo = file.find("vo")
                is_nmpc = file.find("nmpc")
                is_nmpc_no_ref = file.find("no_ref")
                if(is_vo != -1):
                    vo_comp_time[(agent_num)].append(data["computation_time"])
                    # vo_solution_cost[scenario_num].append(data["solution_cost"])
                    vo_control_std[agent_num].append(data["control_std"])
                    vo_control_avg[agent_num].append(data["control_avg"])
                elif(is_nmpc != -1):
                    nmpc_comp_time[agent_num].append(data["computation_time"])
                    # nmpc_solution_cost[scenario_num].append(data["solution_cost"])
                    nmpc_control_std[agent_num].append(data["control_std"])
                    nmpc_control_avg[agent_num].append(data["control_avg"])
                elif(is_nmpc_no_ref != -1):
                    nmpc_no_ref_comp_time[agent_num].append(data["computation_time"])
                    # nmpc_no_ref_solution_cost[scenario_num].append(data["solution_cost"])
                    nmpc_no_ref_control_std[agent_num].append(data["control_std"])
                    nmpc_no_ref_control_avg[agent_num].append(data["control_avg"])
                else:
                    print("File not found")

        for num_agent in self.num_agents:
            # vo_comp_time_avg = np.mean(vo_comp_time[scenario])
            self.nmpc_comp_time_avg_all.append(np.mean(nmpc_comp_time[num_agent]))
            self.nmpc_no_ref_comp_time_avg_all.append(np.mean(nmpc_no_ref_comp_time[num_agent]))
            # vo_control_avg = np.mean(vo_control_std[scenario])
            
            self.nmpc_control_avg_all.append(np.mean(nmpc_control_avg[num_agent]))
            self.nmpc_no_ref_control_avg_all.append(np.mean(nmpc_no_ref_control_avg[num_agent]))
            
            self.nmpc_control_std_all.append(np.mean(nmpc_control_std[num_agent]))
            self.nmpc_no_ref_control_std_all.append(np.mean(nmpc_no_ref_control_std[num_agent]))
    
    def visualize_results(self):
        self.extract_data()

        bar_width = 0.35
        r1 = [x - bar_width/2 for x in self.num_agents]
        r2 = [x + bar_width/2 for x in self.num_agents]

        # Create the bar chart
        plt.bar(r1, self.nmpc_no_ref_comp_time_avg_all, color='blue', width=bar_width, edgecolor='grey', label='NMPC no ref')
        plt.bar(r2, self.nmpc_comp_time_avg_all, color='orange', width=bar_width, edgecolor='grey', label='NMPC with ref')
        error1 = np.std(self.nmpc_no_ref_comp_time_avg_all) / np.sqrt(len(self.nmpc_no_ref_comp_time_avg_all))
        error2 = np.std(self.nmpc_comp_time_avg_all) / np.sqrt(len(self.nmpc_comp_time_avg_all))

        plt.errorbar(r1,  self.nmpc_no_ref_comp_time_avg_all, yerr=error1, fmt='o', color='red', capsize=5)
        plt.errorbar(r2,  self.nmpc_comp_time_avg_all, yerr=error2, fmt='o', color='red', capsize=5)

        # Add xticks on the middle of the group bars
        plt.xlabel('Number of agents')
        plt.ylabel('Average Computation Time (s)')
        plt.xticks([r for r in self.num_agents])
        plt.title("Average computation time comparison")

        plt.legend()
        plt.show()


        bar_width = 0.35
        r1 = [x - bar_width/2 for x in self.num_agents]
        r2 = [x + bar_width/2 for x in self.num_agents]

        # Create the bar chart
        plt.bar(r1, self.nmpc_control_avg_all, color='blue', width=bar_width, edgecolor='grey', label='NMPC with ref')
        plt.bar(r2, self.nmpc_no_ref_control_avg_all, color='orange', width=bar_width, edgecolor='grey', label='NMPC no ref')

        error1 = np.std(self.nmpc_control_avg_all) / np.sqrt(len(self.nmpc_control_avg_all))
        error2 = np.std(self.nmpc_no_ref_control_avg_all) / np.sqrt(len(self.nmpc_no_ref_control_avg_all))

        plt.errorbar(r1,  self.nmpc_control_avg_all, yerr=error1, fmt='o', color='red', capsize=5)
        plt.errorbar(r2,  self.nmpc_no_ref_control_avg_all, yerr=error2, fmt='o', color='red', capsize=5)

        # Add xticks on the middle of the group bars
        plt.xlabel('Number of agents')
        plt.ylabel('Average control effort')
        plt.xticks([r for r in self.num_agents])
        plt.title("Average control effort comparison")

        plt.legend()
        plt.show()