## Generate and Extract Predictions and Visualize Pred Trajectories from UCY and ETH Datasets
## dologan 02/29/24

import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
from tqdm import tqdm
import visualization
import matplotlib.pyplot as plt
from argument_parser import args
from model.ScePT import ScePT
from model.components import *
from model.model_registrar import ModelRegistrar
from model.dataset import *
from torch.utils.tensorboard import SummaryWriter
from model.mgcvae_clique import MultimodalGenerativeCVAE_clique
from Planning import FTOCP
import pdb
from model.components import *
from model.model_utils import *
from model.dynamics import *
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.image as mpimg
import matplotlib.patches as patches
import re
from matplotlib import animation

# torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist


from functools import partial
from pathos.multiprocessing import ProcessPool as Pool
import model.dynamics as dynamic_module
from scipy import linalg, interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import animation
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import pdb

thismodule = sys.modules[__name__]


def animate_scene(rank):
    if torch.cuda.is_available():
        args.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        args.device = f"cpu"

    data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(data_path, "rb") as f:
        env = dill.load(f)
    # print(env.scenes)

    model_dir = os.path.join(args.log_dir, args.log_tag + args.trained_model_dir)
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    hyperparams["use_map_encoding"] = args.map_encoding
    hyperparams["offline_scene_graph"] = args.offline_scene_graph
    hyperparams["incl_robot_node"] = args.incl_robot_node

    hyperparams["edge_encoding"] = not args.no_edge_encoding

    model_registrar = ModelRegistrar(model_dir, args.device)
    ScePT_model = ScePT(model_registrar, hyperparams, None, args.device)
    model_registrar.load_models(iter_num=args.iter_num)
    # model_registrar.model_dict["policy_net"].max_Nnode = hyperparams["max_clique_size"]
    ScePT_model.set_environment(env)
    if (
        args.eval_data_dict == "nuScenes_train.pkl"
        or args.eval_data_dict == "nuScenes_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-trainval_meta"
    elif (
        args.eval_data_dict == "nuScenes_mini_train.pkl"
        or args.eval_data_dict == "nuScenes_mini_val.pkl"
    ):
        nusc_path = "../experiments/nuScenes/v1.0-mini"
    else:
        nusc_path = None
    scene_idx = range(0, 30)

    # Define the Number of Unqiue Scenes to Be Displayed
    scene_idx = range(0,1)

    # Load Only Pedestrian Params

    if "default_con" in hyperparams["dynamic"]["PEDESTRIAN"]:
        default_con = dict()
        input_scale = dict()
        dynamics = dict()
        for nt in hyperparams["dynamic"]:
            model = getattr(dynamic_module, hyperparams["dynamic"][nt]["name"])
            input_scale[nt] = torch.tensor(hyperparams["dynamic"][nt]["limits"])
            dynamics[nt] = model(env.dt, input_scale[nt], "cpu", None, None, nt)
            model = getattr(thismodule, hyperparams["dynamic"][nt]["default_con"])
            default_con[nt] = model
    else:
        dynamics = None
        default_con = None

    for k in scene_idx:
        scene = env.scenes[k]

        ft = hyperparams["prediction_horizon"]
        max_clique_size = hyperparams["max_clique_size"]
        results, _, _ = ScePT_model.replay_prediction(
            scene,
            None,
            ft,
            max_clique_size,
            dynamics,
            default_con,
            num_samples=2,
            nusc_path=nusc_path,
        )
        num_traj_show = 5
        # print("Results", results)
        # print(len(results))
        animate = False
        extract_trajectories(results, None, env.dt, num_traj_show)
        if animate:
            if args.video_name is None:
                sim_clique_prediction(
                    results,
                    None,
                    env.dt,
                    num_traj_show,
                    limits=[100, 100],
                )
            else:
                sim_clique_prediction(
                    results,
                    None,
                    env.dt,
                    num_traj_show,
                    args.video_name + str(k) + ".mp4",
                    limits=[100, 100],
                )


def sim_clique_prediction(
    results,
    map,
    dt,
    num_traj_show,
    output=None,
    limits=None,
    robot_plan=None,
    extra_node_info=None,
    focus_node=None,
    circle_edge_width=0.5,
    line_alpha=0.7,
    line_width=3,
):
    if output:
        matplotlib.use("Agg")

    nframe = len(results)

    if not map is None:
        map_shape = map.as_image().shape
        fig, ax = plt.subplots(figsize=(25, 25 / map_shape[1] * map_shape[0]))
    else:
        fig, ax = plt.subplots(figsize=(25, 25))

    if robot_plan is None:
        robot, traj_plan = None, None
    else:
        robot, traj_plan = robot_plan

    def animate(t, results, map, robot, traj_plan, extra_node_info):

        cmap = ["k", "tab:brown", "g", "tab:orange"]
        limits = [-15,15,-15,15]

        (
            clique_nodes,
            clique_state_history,
            clique_state_pred,
            clique_node_size,
            _,
            _,
        ) = results[t]

        ax.clear()
        ax.grid(False)
        if map is not None and t ==0:
            print("Theres a Map")
        else:
            # print("Here")
            if not limits is None:

                plt.xlim(limits[0:2])
                plt.ylim(limits[2:4])
            ts = ax.transData

            # plot nodes

            for n in range(len(clique_nodes)):
                coords = list()
                for i in range(len(clique_nodes[n])):
                    coords.append(clique_state_history[n][i][-1, 0:2])
                for i in range(len(clique_nodes[n])):
                    if clique_nodes[n][i].type == "PEDESTRIAN":

                        circle = plt.Circle(
                            (coords[i][0], coords[i][1]),
                            clique_node_size[n][i][0],
                            facecolor=cmap[clique_nodes[n][i].type.value],
                            edgecolor="k",
                            lw=circle_edge_width,
                            zorder=3,
                        )

                        ax.add_artist(circle)

                    # elif clique_nodes[n][i].type == "VEHICLE":
                    #     coords = ts.transform([coords[i][0], coords[i][1]])
                    #     patch = plt.Rectangle(
                    #         (
                    #             coords[i][0] - clique_node_size[n][i][0] / 2,
                    #             coords[i][1] - clique_node_size[n][i][1] / 2,
                    #         ),
                    #         clique_node_size[n][i][0],
                    #         clique_node_size[n][i][1],
                    #         fc=cmap[clique_nodes[n][i].type.value],
                    #         zorder=1,
                    #     )
                    #     tr = matplotlib.transforms.Affine2D().rotate_around(
                    #         coords[0], coords[1], clique_state_history[n][i][-1, 3]
                    #     )
                    #     patch.set_transform(ts + tr)
                    #     ax.add_artist(patch)
                    for k in range(min(len(clique_state_pred[n][i]), num_traj_show)):
                        traj = clique_state_pred[n][i][k][:, 0:2]
                        traj = np.vstack((coords[i], traj))
                        ax.plot(
                            traj[:, 0],
                            traj[:, 1],
                            color=cmap[clique_nodes[n][i].type.value],
                            linewidth=line_width,
                            alpha=line_alpha,
                        )
                    for j in range(i + 1, len(clique_nodes[n])):
                        ax.plot(
                            [coords[i][0], coords[j][0]],
                            [coords[i][1], coords[j][1]],
                            color="r",
                            linewidth=line_width,
                            alpha=line_alpha,
                        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            results,
            map,
            robot,
            traj_plan,
            extra_node_info,
        ),
        frames=nframe,
        interval=(1000 * dt),
        blit=False,
        repeat=False,
    )

    if output:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=int(1.0 / dt), metadata=dict(artist="Me"), bitrate=1800)
        anim_name = output
        anim.save(anim_name, writer=writer)
    else:
        plt.show()



# Add mode probabilities
class Scene:
    def __init__(self, time, dt):
        self.time = time
        self.dt = dt
        self.agents = []
        self.current_positions = {}
        self.state_history = {} ## Listed Backwards, i.e current position is last element in history
        self.pred = {}
    
    def update_agents(self, agent):
        # Updates List of Agent ID's in a given scene
        if agent not in self.agents:
            self.agents.append(agent)

    def update_current_pos(self, agent_id, pos):
        # Dictionary that maps agent_id to current pose in a scene
        self.current_positions[agent_id] = pos

    def update_state_history(self, agent_id, history):
        self.state_history[agent_id] = history 
    
    def update_predictions(self, agent_id, predictions):
        # Updates pred an np.ndarray organized as follows (prediction length, mode, dim(x,y))
        self.pred[agent_id] = predictions
    
    def visualize_scene(self):
        """
        Plot a Snapshot of a Given Scene
        """
        plt.figure()

        for id in self.agents:
            plt.plot(self.current_positions[id][0], self.current_positions[id][1],
                     color="blue", marker = 'o', markersize = 8, label= 'Current Pose')
            # plt.plot(self.state_history[id][:,:1], self.state_history[id][:,1:2],
            #          color="red", marker = 'o', markersize = 4, markerfacecolor='none', label= 'State History')
            dims = self.pred[id].shape

            # Draw each Mode Prediction
            for mode in range(dims[0]):
                plt.plot(self.pred[id][mode, :,0:1], self.pred[id][mode,:,1:2],
                         color="green", marker = 'o', markersize = 4, markerfacecolor='none', label= f'Prediction Mode {mode}, Agent {id}',linestyle='dotted')

        plt.title(f'Scene at Time {self.time}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        # plt.legend()
        plt.grid(True)
        plt.show()


#Add Ellipses during Plotting
class Predictions:
    def __init__(self, dt):
        self.timestamps = []
        self.dt = dt
        self.scenes = {}
        self.person_img = mpimg.imread('./ped.jpg')

    def updateScene(self, time, scene):
        self.scenes[time] = scene

    def animate_scene(self, record):
        nframes = len(self.scenes.keys())
        fig, ax = plt.subplots()

        def update(frame):
            cmap = ["k", "tab:brown", "g", "tab:orange"]
            limits = [-5,15,-5,5]
            na = 0.005
            colors  = ['green', 'red', 'blue']
            count = True

            ax.clear()
            plt.xlim(limits[0:2])
            plt.ylim(limits[2:4])
            t = frame*self.dt
            scene = self.scenes[t]
            plt.title('ScePT Pedestrian Trajectory Predictions (Zara Dataset)')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')

            for id in scene.agents:
                dims = scene.pred[id].shape
                # Plot each Mode Prediction, Covariances
                for mode in range(dims[0]):
                    if count:
                        plt.plot(np.insert(scene.pred[id][mode, :,0:1], 0, scene.current_positions[id][0], axis=0), 
                                np.insert(scene.pred[id][mode,:,1:2], 0, scene.current_positions[id][1], axis=0), #  scene.pred[id][mode, :,0:1], scene.pred[id][mode,:,1:2],
                                color=colors[mode], marker = 'o', markersize = 4, markerfacecolor='none',linestyle='dotted', zorder = 1, label= f'Mode {mode}')
                    else:
                        plt.plot(np.insert(scene.pred[id][mode, :,0:1], 0, scene.current_positions[id][0], axis=0), 
                                np.insert(scene.pred[id][mode,:,1:2], 0, scene.current_positions[id][1], axis=0), #  scene.pred[id][mode, :,0:1], scene.pred[id][mode,:,1:2],
                                color=colors[mode], marker = 'o', markersize = 4, markerfacecolor='none',linestyle='dotted', zorder = 1) #,label= f'Mode {mode}, Agent {id}')
                    # plt.plot(scene.pred[id][mode, :,0:1], scene.current_positions[id][1], scene.pred[id][mode,:,1:2],
                    #         color="green", marker = 'o', markersize = 4, markerfacecolor='none', label= f'Prediction Mode {mode}, Agent {id}',linestyle='dotted') # Doesnt Connect to Curr State

                    counter = t
                    Q_matrix = np.identity(2)*na
                    cov = np.identity(2)* na
                    for pair in scene.pred[id][mode]:
                        mean = pair
                        cov = cov + Q_matrix * self.dt
                        confidence = 0.95 # Might be Representative of Mode Prob
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        order = eigenvalues.argsort()[::-1]
                        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

                        width, height = 2 * np.sqrt(eigenvalues * -2 * np.log(1 - confidence))

                        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

                        ellipse = patches.Ellipse(mean, width, height, angle=angle, fill=True, edgecolor="green", linestyle='solid', alpha = 0.2)
                        ax.add_patch(ellipse)

                count = False
                # Plot Current Pedestrian Positions, State Histories (Dots)
                # plt.plot(scene.current_positions[id][0], scene.current_positions[id][1],
                #      color="blue", marker = 'o', markersize = 8, label= 'Current Pose')
                
                # plt.plot(scene.state_history[id][:,:1], scene.state_history[id][:,1:2],
                #          color="red", marker = 'o', markersize = 4, markerfacecolor='none', label= 'State History')

                # Plot Current Pedestrian Positions as Image
                ax.imshow(self.person_img, extent=(scene.current_positions[id][0] - 0.325, scene.current_positions[id][0] + 0.325,
                                                    scene.current_positions[id][1] - 0.25, scene.current_positions[id][1] + 0.25), zorder = 2)
                
                # Plot Pedestrian ID
                # ax.text(scene.current_positions[id][0] - 0.325, scene.current_positions[id][1]-0.25, [int(match) for match in re.findall(r'\d+', id.id)], fontsize=9, ha='center', va='center', color='black')
            plt.legend()
                



        anim = animation.FuncAnimation(fig,
        update,
        frames=nframes,
        interval=(500 * self.dt),
        blit=False,
        repeat=False,)

        if record:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=int(1.0 / self.dt), metadata=dict(artist="Me"), bitrate=1800)
            anim_name = 'anim.mp4'
            anim.save(anim_name, writer=writer)
        else:
            plt.show()
        
            
    



def extract_trajectories(results, map, dt, num_traj_show):
    timesteps = len(results)
    predictions = Predictions(dt)
    # timesteps = 1
    # for t in range(72,74):
    for t in range(timesteps):  # Iterate over each time step in the scene
        # print("END")
        (
            clique_nodes,
            clique_state_history,
            clique_state_pred,
            clique_node_size,
            _,
            _,
        ) = results[t]

        curr_time = dt*t
        current_scene =  Scene(curr_time, dt)
        for n in range(len(clique_nodes)): # Iterate over each clique in the scene
            for i in range(len(clique_nodes[n])): # Iterate over each agent in each clique in the scene
                agent_id = clique_nodes[n][i]
                current_scene.update_agents(agent_id)
                current_scene.update_current_pos(agent_id,clique_state_history[n][i][-1, 0:2]) 

                # Change End Index to 4 to get Velocities as well
                current_scene.update_state_history(agent_id,clique_state_history[n][i][:, :2])
                current_scene.update_predictions(agent_id, np.array(clique_state_pred[n][i])[:,:,0:2])
                counter = 1
                # print(current_scene.state_history[agent_id])
                # print(current_scene.state_history[agent_id][:,1:2])
                # if counter:
                #     print(agent_id)
                #     print(clique_state_pred[n][i])
                #     print(len(clique_state_pred[n][i]))
                #     print(np.array(clique_state_pred[n][i])[:,:,0:2])
                #     # print(clique_state_pred[n][i][0])
                #     print("________a________________")
                #     counter = 0
        predictions.timestamps.append(curr_time)
        predictions.scenes[curr_time] = current_scene

        ## Uncomment to Visualize a Single Scene
        # current_scene.visualize_scene()
    predictions.animate_scene(record=False)
    return predictions



if __name__ == "__main__":
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    # Generate and Display Animation of Predicted Pedestrians Trajectories
    eval(args.eval_task)(args.local_rank)