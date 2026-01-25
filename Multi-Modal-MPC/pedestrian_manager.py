import numpy as np
import logging
import matplotlib.pyplot as plt

def barrier_function(d, d_thresh, k=10):
    """
    Computes a scaling factor based on the distance using a sigmoid barrier function.

    Parameters:
    - d: Minimum distance between pedestrian and robot.
    - d_thresh: Threshold distance for activation of the barrier.
    - k: Steepness of the sigmoid function.

    Returns:
    - scaling_factor: Value between 0 and 1 to adjust mode probabilities.
    """
    scaling_factor = 1 / (1 + np.exp(k * (d_thresh - d)))
    return scaling_factor

class Pedestrian:
    def __init__(self, position, v_mean, v_sigma, omega_mean, omega_sigma, heading, color):
        """
        Initializes a Pedestrian.

        Parameters:
        - position: Initial position as a NumPy array [x, y].
        - v_mean: Mean speed.
        - v_sigma: Standard deviation for speed.
        - omega_mean: List/array with two elements, one for each mode (mean angular velocity).
        - omega_sigma: Standard deviation for angular velocity.
        - heading: Initial heading (theta) in radians.
        - color: Color for plotting.
        """
        self.position = np.array(position, dtype=np.float64)
        self.v_mean = v_mean
        self.v_sigma = v_sigma
        self.omega_mean = omega_mean  # Two elements: one for 'mode1' and one for 'mode2'
        self.omega_sigma = omega_sigma
        self.heading = heading  # In radians
        self.color = color
        self.current_mode = None  # 'mode1' or 'mode2'
        self.time_since_last_mode_switch = 0.0
        self.mode_probabilities = {'mode1': 0.5, 'mode2': 0.5}
        
        # Initialize desired speed and angular velocity.
        self.desired_speed = self.v_mean
        self.current_omega = self.omega_mean[0]

        # Store initial states for reset purposes
        self.initial_position = self.position.copy()
        self.initial_heading = heading
        self.initial_mode_probabilities = self.mode_probabilities.copy() 

        # For adding noise in predictions.
        self.noise_std = 0.01

    def assign_behavior_mode(self):
        """
        Assigns a behavior mode based on the current mode probabilities.
        """
        modes = ['mode1', 'mode2']
        probabilities = [self.mode_probabilities['mode1'], self.mode_probabilities['mode2']]
        self.current_mode = np.random.choice(modes, p=probabilities)

    def update_mode(self, dt, mode_sample_freq):
        """
        Update the behavior mode at a fixed sampling interval.
        """
        self.time_since_last_mode_switch += dt
        if self.time_since_last_mode_switch >= mode_sample_freq:
            self.assign_behavior_mode()
            self.time_since_last_mode_switch = 0.0

    def update_velocity(self, dt):
        """
        Update the pedestrian's speed and angular velocity based on the current mode.
        Speed is sampled directly from N(v_mean, v_sigma).
        Angular velocity is sampled directly from N(omega_mean, omega_sigma) where omega_mean depends on the mode.
        """
        if self.current_mode is None:
            self.assign_behavior_mode()

        # Sample new speed directly from the Gaussian.
        new_speed = max(0.2, np.random.normal(self.v_mean, self.v_sigma))
        self.desired_speed = new_speed

        # Select the base angular velocity according to the current mode.
        if self.current_mode == 'mode1':
            new_omega = np.random.normal(self.omega_mean[0], self.omega_sigma)
        else:
            new_omega = np.random.normal(self.omega_mean[1], self.omega_sigma)
        self.current_omega = new_omega

        # Compute the velocity vector using the current heading.
        self.velocity = self.desired_speed * np.array([np.cos(self.heading), np.sin(self.heading)])

    def update_position(self, dt):
        """
        Update the pedestrian's state (position and heading) using unicycle dynamics.
        """
        self.position += self.velocity * dt
        self.heading += self.current_omega * dt

    def predict_future_positions(self, dt, prediction_steps, mode=None):
        """
        Predict future states [x, y, theta] using the unicycle model.
        Control inputs (speed and angular velocity) are sampled directly from their respective Gaussian distributions.

        Parameters:
        - dt: Timestep duration.
        - prediction_steps: Number of steps to predict.
        - mode: Mode to simulate ('mode1' or 'mode2'). If None, uses current_mode.

        Returns:
        - future_states: List of predicted states (each a NumPy array [x, y, theta]).
        """
        future_states = []
        # Start with the current state.
        state = np.array([self.position[0], self.position[1], self.heading])
        if mode is None:
            mode = self.current_mode

        # Select the base angular velocity for the given mode.
        omega_base = self.omega_mean[0] if mode == 'mode1' else self.omega_mean[1]

        for _ in range(prediction_steps):
            # Sample speed directly from N(v_mean, v_sigma).
            new_speed = max(0.3, np.random.normal(self.v_mean, self.v_sigma))
            # Sample angular velocity directly from N(omega_base, omega_sigma).
            new_omega = np.random.normal(omega_base, self.omega_sigma)

            # Unicycle dynamics: update heading first, then position.
            theta = state[2] + new_omega * dt
            new_x = state[0] + new_speed * np.cos(theta) * dt
            new_y = state[1] + new_speed * np.sin(theta) * dt

            state = np.array([new_x, new_y, theta])
            # Optionally add Gaussian noise to the positional components.
            noisy_state = state.copy()
            noisy_state[0:2] += np.random.normal(0, self.noise_std, 2)
            future_states.append(noisy_state)
        return future_states

    def update_mode_probabilities(self, robot_future_states, d_thresh, k, temperature):
        """
        Update the pedestrian's mode probabilities based on the robot's predicted trajectory.
        Predictions are computed using the unicycle model.

        Parameters:
        - robot_future_positions: NumPy array representing the robot's future positions.
        - d_thresh: Threshold distance for the barrier function.
        - k: Steepness parameter for the barrier function.
        - temperature: Temperature parameter for the softmax conversion.

        The updated probabilities are stored in self.mode_probabilities.
        """
        modes = ['mode1', 'mode2']
        cbf_values = []

        robot_future_states = np.array(robot_future_states)

        if robot_future_states.ndim == 2:
            robot_future_states = robot_future_states
        elif robot_future_states.ndim == 3:
            robot_future_states = robot_future_states[-1]  # Get the last prediction
        try:
            for mode in modes:
                ped_future_states = self.predict_future_positions(
                    dt=self.pedestrian_simulation_dt,
                    prediction_steps=max(robot_future_states.shape),
                    mode=mode
                )
                ped_future_states = np.array(ped_future_states)  # (prediction_steps, 3)
                # import pdb; pdb.set_trace()
                robot_future_states = robot_future_states.reshape(-1,3)
                
                robot_x = robot_future_states[:, 0]
                robot_y = robot_future_states[:, 1]
                
                ped_x = ped_future_states[:, 0]
                ped_y = ped_future_states[:, 1]

                # plt.scatter(ped_x, ped_y, label=f"Pedestrian {mode}")
                distances = np.sqrt((robot_x - ped_x)**2 + (robot_y - ped_y)**2)
                d_min = np.min(distances)
                cbf = barrier_function(d_min, d_thresh, k)
                cbf_values.append(cbf)

            # Softmax conversion of CBF values.
            cbf_values = np.array(cbf_values)
            scaled_cbf = cbf_values / temperature
            max_cbf = np.max(scaled_cbf)  # For numerical stability.
            exp_cbf = np.exp(scaled_cbf - max_cbf)
            probabilities = exp_cbf / np.sum(exp_cbf)
            self.mode_probabilities = {'mode1': probabilities[0], 'mode2': probabilities[1]}
        except:
            # import pdb; pdb.set_trace()
            self.mode_probabilities = {'mode1': 0.5, 'mode2': 0.5}
            # pass

    def reset(self):
        self.position = self.initial_position.copy()
        self.heading = self.initial_heading
        self.current_mode = None
        self.time_since_last_mode_switch = 0.0
        self.mode_probabilities = self.initial_mode_probabilities.copy()
        self.desired_speed = self.v_mean
        self.current_omega = self.omega_mean[0]

class PedestrianManager:
    def __init__(self, ped_params):
        """
        Initializes the PedestrianManager.

        Parameters:
        - ped_params: Dictionary containing keys such as:
          "num_pedestrians", "num_samples", "N", "dt", "mode_sample_freq",
          "v_mean", "v_sigma", "omega_mean", and "omega_sigma".
        """
        self.ped_params = ped_params
        self.num_pedestrians = ped_params["num_pedestrians"]
        self.num_samples = ped_params["num_samples"]  # Per mode
        self.prediction_horizon = ped_params["N"] * ped_params["dt"]
        self.dt = ped_params["dt"]
        self.prediction_steps = ped_params["N"]
        self.mode_sample_freq = ped_params["mode_sample_freq"]
        self.v_mean = ped_params["v_mean"]
        self.omega_mean = ped_params["omega_mean"]  # Two elements (one per mode)
        self.v_sigma = ped_params["v_sigma"]
        self.omega_sigma = ped_params["omega_sigma"]

        self.pedestrians = self.initialize_pedestrians()

    def initialize_pedestrians(self):
        pedestrians = []
        manual_peds = self.ped_params.get("manual_pedestrians", None)

        if manual_peds is not None:
            # Use manually specified pedestrians.
            for ped_def in manual_peds:
                position = np.array(
                    ped_def.get("position", np.random.uniform(-10, 10, size=2))
                )
                heading = ped_def.get("heading", np.random.uniform(-np.pi, np.pi))
                color = ped_def.get("color", "black")  # Default color if not provided
                pedestrian = Pedestrian(
                    position=position,
                    v_mean=self.v_mean,
                    v_sigma=self.v_sigma,
                    omega_mean=self.omega_mean,
                    omega_sigma=self.omega_sigma,
                    heading=heading,
                    color=color
                )
                pedestrian.pedestrian_simulation_dt = self.dt  # For use in predictions.
                pedestrians.append(pedestrian)
        else:
            # Automatically spawn pedestrians.
            cmap = plt.get_cmap('hsv')
            colors = cmap(np.linspace(0, 1, self.num_pedestrians))
            for i in range(self.num_pedestrians):
                position = np.random.uniform(low=-10, high=10, size=2)
                heading = np.random.uniform(-np.pi, np.pi)
                color = colors[i]
                pedestrian = Pedestrian(
                    position=position,
                    v_mean=self.v_mean,
                    v_sigma=self.v_sigma,
                    omega_mean=self.omega_mean,
                    omega_sigma=self.omega_sigma,
                    heading=heading,
                    color=color
                )
                pedestrian.pedestrian_simulation_dt = self.dt  # For use in predictions.
                pedestrians.append(pedestrian)

        return pedestrians

    def update_pedestrians(self):
        """
        Update each pedestrian's state (mode, velocity, and position) using unicycle dynamics.
        """
        for ped in self.pedestrians:
            ped.update_mode(self.dt, self.mode_sample_freq)
            ped.update_velocity(self.dt)
            ped.update_position(self.dt)

    def get_gmm_predictions_from_current(self):
        """
        Generate GMM predictions for each pedestrian over the prediction horizon.
        Predictions for each mode include a list of state means ([x, y, theta]) and associated covariance matrices.
        """
        prediction_samples = []

        for ped in self.pedestrians:
            agent_prediction = {}

            for mode_idx, mode in enumerate(['mode1', 'mode2']):
                means = []
                covariances = []

                for _ in range(self.prediction_steps):
                    future_states = ped.predict_future_positions(
                        dt=self.dt,
                        prediction_steps=self.prediction_steps,
                        mode=mode
                    )

                    for state in future_states:
                        x, y, theta = state
                        means.append([x, y, theta])
                        covariances.append(np.diag([self.v_sigma, self.omega_sigma]))

                agent_prediction[mode_idx] = {'means': means, 'covariances': covariances}

            prediction_samples.append(agent_prediction)

        return prediction_samples

    def get_gmm_predictions(self):
        gmm_predictions = []

        for agent in range(self.num_pedestrians):
            agent_prediction = {}
            # Calculate the mean and covariance for each action at each timestep within the prediction horizon
            for mode, omega in enumerate(self.omega_mean):
                # Mean and covariance vectors for the entire prediction horizon
                means = []
                covariances = []

                # Populate the means and covariances for each timestep within the prediction horizon
                for _ in np.arange(0, self.prediction_steps):
                    means.append([self.v_mean, omega])  # The mean of v and omega is the action's value
                    covariance = np.diag([self.v_sigma, self.omega_sigma])  # Diagonal covariance matrix
                    covariances.append(covariance)
        
                # Assign the mean and covariance vectors to the corresponding mode
                agent_prediction[mode] = {
                    'means': means,  # List of means over the prediction horizon
                    'covariances': covariances  # List of covariance matrices over the prediction horizon
                }

            # The predictions for all modes of the single agent are encapsulated in a list
            gmm_predictions.append(agent_prediction)

        return gmm_predictions
    
    def get_mode_probabilities(self):
        """
        Returns a dictionary mapping each pedestrian's id (index) to a list of mode probabilities.
        The probabilities are ordered based on the sorted order of mode keys.
        """
        mode_prob_map = {}
        for idx, ped in enumerate(self.pedestrians):
            # If your pedestrian.mode_probabilities is a dictionary like {'mode1': 0.5, 'mode2': 0.5},
            # you can order them (here alphabetically) to get a list.
            probs = [ped.mode_probabilities[key] for key in sorted(ped.mode_probabilities.keys())]
            mode_prob_map[idx] = probs
        return mode_prob_map
    
    def reset(self):
        for ped in self.pedestrians:
            ped.reset()
