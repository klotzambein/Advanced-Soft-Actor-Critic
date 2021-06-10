from gym import Env, spaces
from osim import Pose, RUGTFPEnv
import numpy as np
import pandas as pd
import os
import random
import time
import math
from scipy.special import expit

imi_learning_reward_columns = ['pelvis_ty', 'pelvis_tilt', 'pelvis_list', 'ankle_angle_l', 'ankle_angle_r', 'knee_angle_l', 'knee_angle_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 'hip_adduction_r']

def imi_learning_observ_columns(observations):
    data = np.zeros(11)
    data[0] = observations["body_pos"]["pelvis"][1]
    data[1] = observations["joint_pos"]["ground_pelvis"][0]
    data[2] = observations["joint_pos"]["ground_pelvis"][1]
    data[3] = observations['l_leg']['joint']['ankle']
    data[4] = observations['r_leg']['joint']['ankle']
    data[5] = observations['l_leg']['joint']['knee']
    data[6] = observations['r_leg']['joint']['knee']
    data[7] = observations['l_leg']['joint']['hip']
    data[8] = observations['r_leg']['joint']['hip']
    data[9] = observations['l_leg']['joint']['hip_abd']
    data[10] = observations['r_leg']['joint']['hip_abd']
    return data

class OpenSimEnv(Env):
    def __init__(self, data_dir, visualize: bool, integrator_accuracy=1e-2, model_name="OS4_gait14dof15musc_2act_LTFP_VR_Joint_Fix.osim"):
        # print("Init: ", os.getpid())
        self.imitation_data = {}
        self.imitation_data_start_times = {}
        self.imitation_data_tensor = {}
        for root,dirs,files in os.walk(data_dir):
            # print("Init-walk: ", os.getpid())
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)
                    name = file.split('.')[0]
                    df = pd.read_csv(path, sep=',')
                    # skip row 0, it contains the default pose
                    df.drop(index=df.index[0], 
                            axis=0, 
                            inplace=True)
                    df.reset_index(inplace=True, drop=True)
                    
                    self.imitation_data[name] = df
                    # To mitigate reset errors we choose a few start times that are likely to be good.
                    # start_times = [df['time'][t] for t in np.random.randint(0, len(df) - 100, (100, )) if df["knee_angle_l"][t] > -0.5]
                    start_times = [0]
                    self.imitation_data_start_times[name] = start_times

                    self.imitation_data_tensor[name] = df.loc[:,imi_learning_reward_columns].values
        
        self.env = RUGTFPEnv(model_name=model_name, visualize=visualize, integrator_accuracy=integrator_accuracy)
        
        self.observation_space = spaces.Box(0, 1, shape=(97, ))
        self.action_space = spaces.Box(0, 1, shape=(17, ))

        # print("Init-done: ", os.getpid())
        pass

    def observation_arrays(self, state_desc):
        obs = self.env.get_observation_dict(state_desc)
        obs_r_leg = self.env.get_leg_observations("r", state_desc)
        obs_l_leg = self.env.get_leg_observations("l", state_desc)

        human = np.zeros((72,))
        i = 0
        
        human[i+0] = obs["pelvis"]["pos"][1]
        human[i+1] = obs["pelvis"]["pitch"]
        human[i+2] = obs["pelvis"]["roll"]
        human[i+3:i+9] = obs["pelvis"]["vel"]
        i += 9

        # Right Leg
        human[i:i+3] = obs_r_leg["ground_reaction_forces"]
        i += 3

        human[i+0] = obs_r_leg["joint"]["hip_abd"]
        human[i+1] = obs_r_leg["joint"]["hip"]
        human[i+2] = obs_r_leg["joint"]["knee"]
        human[i+3] = obs_r_leg["joint"]["ankle"]
        i += 4

        human[i+0] = obs_r_leg["d_joint"]["hip_abd"]
        human[i+1] = obs_r_leg["d_joint"]["hip"]
        human[i+2] = obs_r_leg["d_joint"]["knee"]
        human[i+3] = obs_r_leg["d_joint"]["ankle"]
        i += 4

        for m in ["HAB_R", "HAD_R", "HFL_R", "GLU_R", "HAM_R", "RF_R", "VAS_R", "BFSH_R", "GAS_R", "SOL_R", "TA_R"]:
            human[i+0] = obs_r_leg[m]["f"]
            human[i+1] = obs_r_leg[m]["l"]
            human[i+2] = obs_r_leg[m]["v"]
            i += 3
        
        # Left leg
        human[i:i+3] = obs_l_leg["ground_reaction_forces"]
        i += 3

        human[i+0] = obs_l_leg["joint"]["hip_abd"]
        human[i+1] = obs_l_leg["joint"]["hip"]
        i += 2
        human[i+0] = obs_l_leg["d_joint"]["hip_abd"]
        human[i+1] = obs_l_leg["d_joint"]["hip"]
        i += 2

        for m in ["HAB_L", "ADD_L", "GLU_L", "HFL_L"]:
            human[i+0] = obs_l_leg[m]["f"]
            human[i+1] = obs_l_leg[m]["l"]
            human[i+2] = obs_l_leg[m]["v"]
            i += 3

        assert i == len(human)

        prosthesis = np.zeros(25)
        i = 0

        prosthesis[i+0] = obs["pelvis"]["pos"][1]
        prosthesis[i+1] = obs["pelvis"]["pitch"]
        prosthesis[i+2] = obs["pelvis"]["roll"]
        prosthesis[i+3:i+9] = obs["pelvis"]["vel"]
        i += 9

        prosthesis[i+0] = obs_l_leg["joint"]["knee"]
        prosthesis[i+1] = obs_l_leg["joint"]["ankle"]
        prosthesis[i+2] = obs_l_leg["d_joint"]["knee"]
        prosthesis[i+3] = obs_l_leg["d_joint"]["ankle"]
        i += 4

        prosthesis[i+0] = obs_l_leg['force']['knee']
        prosthesis[i+1] = obs_l_leg['force']['ankle']
        prosthesis[i+2] = obs_l_leg['actuator']['knee']['speed']
        prosthesis[i+3] = obs_l_leg['actuator']['knee']['control']
        prosthesis[i+4] = obs_l_leg['actuator']['knee']['power']
        prosthesis[i+5] = obs_l_leg['actuator']['knee']['stress']
        prosthesis[i+6] = obs_l_leg['actuator']['knee']['actuation']
        prosthesis[i+7] = obs_l_leg['actuator']['ankle']['speed']
        prosthesis[i+8] = obs_l_leg['actuator']['ankle']['control']
        prosthesis[i+9] = obs_l_leg['actuator']['ankle']['power']
        prosthesis[i+10] = obs_l_leg['actuator']['ankle']['stress']
        prosthesis[i+11] = obs_l_leg['actuator']['ankle']['actuation']
        i += 12

        assert i == len(prosthesis)

        return { "human": human, "prosthesis": prosthesis }

    def observation_array_normalized(self, observation):
        obs_arrays = self.observation_arrays(observation)
        obs_array = np.append(obs_arrays["human"], obs_arrays["prosthesis"])
        if np.isnan(np.sum(obs_array)):
            print("NaN in observation array")
        # obs_array = expit(obs_array)
        # print(obs_array)
        return obs_array

    def is_in_bounds(self, t, obs):
        imi_data = self.imitation_data[self.current_imitation_data]
        bounds = imi_data.iloc[(imi_data['time']-t).abs().argsort()[0]]
        pose = Pose()
        pose.set_from_dict_degrees(bounds)
        
        return obs["pose"].is_in_bounds(pose, 15 * np.pi / 180, 0.2)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Fix action space
        action[0:15] = action[0:15] / 2 + 0.5

        # Reward is one, because the model is constraint to feasible regions
        reward = 1
        done = False

        try:
            start_time = time.time()

            self.env.step(action)
            self.t += self.env.stepsize

            time_diff = time.time() - start_time
            if time_diff > 0.99:
                print("Step took too long: {} sec".format(time_diff))
                
        except RuntimeError as err:
            print("Caught runtime error during step: {}".format(err))
            done = True

        state_desc = self.env.get_state_desc()

        if not self.is_in_bounds(self.t, state_desc):
            done = True
            reward = 0

        obs_array = self.observation_array_normalized(state_desc)
        
        return obs_array, reward, done, {}

    def reset(self):
        # print("Reset")
        self.current_imitation_data = "new"#random.choice(list(self.imitation_data.keys()))
        imi = self.imitation_data[self.current_imitation_data]

        t = random.choice(self.imitation_data_start_times[self.current_imitation_data])
        self.start_time = t
        self.t = t
        
        imi_index = (imi['time'] - t).abs().argmin()
        bounds = imi.iloc[imi_index]
        bounds_next = imi.iloc[imi_index + 1]
       
        pose = Pose()
        pose.set_from_dict_degrees(bounds)
        
        pose_next = Pose()
        pose_next.set_from_dict_degrees(bounds_next)

        pose.compute_velocities(pose_next, bounds_next['time'] - bounds['time'])

        # print("Pre init at {}".format(t))
        self.env.reset(pose)
        # print("Post init")

        observation = self.env.get_state_desc()
        array = self.observation_array_normalized(observation)

        return array