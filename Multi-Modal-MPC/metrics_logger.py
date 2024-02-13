import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

class MetricsLogger:
    def __init__(self):
        self.metrics_data = {}

    def log_metrics(self, run_description, trial_num, state_cache, control_cache, map, initial_state, final_state, avg_comp_time, max_comp_time, traj_length, makespan, avg_rob_dist, c_avg, success, execution_collision, max_time_reached, infeasible_count, feedback_gain_avg, uncontrolled_fleet_data, num_timesteps):
        # Log the metrics for a specific algorithm trial
        if run_description not in self.metrics_data:
            self.metrics_data[run_description] = {}
        if trial_num not in self.metrics_data[run_description]:
            self.metrics_data[run_description][trial_num] = {
                "state_cache": {},
                "control_cache": {},
                "initial_state": [],
                "final_state": [],
                "map": [],
                "avg_comp_time": [],
                "max_comp_time": [],
                "c_avg": [],
                "traj_length": [],
                "makespan": [],
                "avg_rob_dist": [],
                "success": [],
                "execution_collision": [],
                "max_time_reached": [],
                "infeasible_count": [],
                "feedback_gain_avg": [],
                "uncontrolled_fleet_data": [],
                "num_timesteps": []
            }
        self.metrics_data[run_description][trial_num]["state_cache"] = state_cache
        self.metrics_data[run_description][trial_num]["control_cache"] = control_cache
        self.metrics_data[run_description][trial_num]["initial_state"] = initial_state
        self.metrics_data[run_description][trial_num]["final_state"] = final_state
        self.metrics_data[run_description][trial_num]["map"] = map
        self.metrics_data[run_description][trial_num]["avg_comp_time"] = avg_comp_time
        self.metrics_data[run_description][trial_num]["max_comp_time"] = max_comp_time
        self.metrics_data[run_description][trial_num]["traj_length"] = traj_length
        self.metrics_data[run_description][trial_num]["makespan"] = makespan
        self.metrics_data[run_description][trial_num]["avg_rob_dist"] = avg_rob_dist
        self.metrics_data[run_description][trial_num]["success"] = success
        self.metrics_data[run_description][trial_num]["c_avg"] = c_avg
        self.metrics_data[run_description][trial_num]["execution_collision"] = execution_collision
        self.metrics_data[run_description][trial_num]["max_time_reached"] = max_time_reached
        self.metrics_data[run_description][trial_num]["infeasible_count"] = infeasible_count
        self.metrics_data[run_description][trial_num]["feedback_gain_avg"] = feedback_gain_avg
        self.metrics_data[run_description][trial_num]["uncontrolled_fleet_data"] = uncontrolled_fleet_data
        self.metrics_data[run_description][trial_num]["num_timesteps"] = num_timesteps

    def print_metrics_summary(self):
        # Returns the collected metrics data
        for algorithm_name, trials in self.metrics_data.items():
            for trial_num, metrics in trials.items():
                avg_computation_time = metrics["avg_comp_time"]
                max_computation_time = metrics["max_comp_time"]
                traj_length = metrics["traj_length"]
                makespan = metrics["makespan"]
                avg_rob_dist = metrics["avg_rob_dist"]
                success = metrics["success"]
                c_avg = metrics["c_avg"]
                execution_collision = metrics["execution_collision"]
                max_time_reached = metrics["max_time_reached"]
                infeasible_count = metrics["infeasible_count"]

                if(success):
                    print("Avg Comp Time:")
                    print(avg_computation_time)
                    print("Max Comp time:")
                    print(max_computation_time)
                    print("Traj Length:")
                    print(traj_length)
                    print("Makespan:")
                    print(makespan)
                    print("Infeasible_count:")
                    print(infeasible_count)
                    print("Success:")
                    print(bool(success))
                    print("===================")
                else:
                    print("Success:")
                    print(bool(success))
                    print("Collision:")
                    print(execution_collision)
                    print("Timeout:")
                    print(max_time_reached)
                    print("===================")

    def save_metrics_data(self, base_folder="mm_results"):
        # Create the base folder if it doesn't exist
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        for run_description, trials in self.metrics_data.items():
            run_folder = os.path.join(base_folder, run_description)
            # Create a folder for each run description if it doesn't exist
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)

            for trial_num, metrics in trials.items():
                file_name = f"trial_{trial_num}.pkl"
                file_path = os.path.join(run_folder, file_name)
                with open(file_path, 'wb') as file:
                    pickle.dump(metrics, file)


