from gym import Env, spaces
from osim.env import RUGTFPEnv
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
    data[0] = observations["pelvis"]["height"]
    data[1] = observations["pelvis"]["pitch"]
    data[2] = observations["pelvis"]["roll"]
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
                    # To mitigate reste errors we choose a few start times that are likely to be good.
                    # start_times = [t for t in np.random.randint(0, 500, (100, )) if df["knee_angle_l"][t] > -0.5]
                    start_times = [0]
                    self.imitation_data_start_times[name] = start_times

                    self.imitation_data_tensor[name] = df.loc[:,imi_learning_reward_columns].values
        
        self.env = RUGTFPEnv(model_name=model_name, visualize=visualize, integrator_accuracy=integrator_accuracy)
        
        self.observation_space = spaces.Box(0, 1, shape=(97, ))
        self.action_space = spaces.Box(0, 1, shape=(17, ))

        # print("Init-done: ", os.getpid())
        pass

    def observation_arrays(self, observation):
        human = np.zeros((72,))
        i = 0
        
        human[i+0] = observation["pelvis"]["height"]
        human[i+1] = observation["pelvis"]["pitch"]
        human[i+2] = observation["pelvis"]["roll"]
        human[i+3:i+9] = observation["pelvis"]["vel"]
        i += 9

        # Right Leg
        human[i:i+3] = observation["r_leg"]["ground_reaction_forces"]
        i += 3

        human[i+0] = observation["r_leg"]["joint"]["hip_abd"]
        human[i+1] = observation["r_leg"]["joint"]["hip"]
        human[i+2] = observation["r_leg"]["joint"]["knee"]
        human[i+3] = observation["r_leg"]["joint"]["ankle"]
        i += 4

        human[i+0] = observation["r_leg"]["d_joint"]["hip_abd"]
        human[i+1] = observation["r_leg"]["d_joint"]["hip"]
        human[i+2] = observation["r_leg"]["d_joint"]["knee"]
        human[i+3] = observation["r_leg"]["d_joint"]["ankle"]
        i += 4

        for m in ["HAB_R", "HAD_R", "HFL_R", "GLU_R", "HAM_R", "RF_R", "VAS_R", "BFSH_R", "GAS_R", "SOL_R", "TA_R"]:
            human[i+0] = observation["r_leg"][m]["f"]
            human[i+1] = observation["r_leg"][m]["l"]
            human[i+2] = observation["r_leg"][m]["v"]
            i += 3
        
        # Left leg
        human[i:i+3] = observation["l_leg"]["ground_reaction_forces"]
        i += 3

        human[i+0] = observation["l_leg"]["joint"]["hip_abd"]
        human[i+1] = observation["l_leg"]["joint"]["hip"]
        i += 2
        human[i+0] = observation["l_leg"]["d_joint"]["hip_abd"]
        human[i+1] = observation["l_leg"]["d_joint"]["hip"]
        i += 2

        for m in ["HAB_L", "ADD_L", "GLU_L", "HFL_L"]:
            human[i+0] = observation["l_leg"][m]["f"]
            human[i+1] = observation["l_leg"][m]["l"]
            human[i+2] = observation["l_leg"][m]["v"]
            i += 3

        assert i == len(human)

        prosthesis = np.zeros(25)
        i = 0

        prosthesis[i+0] = observation["pelvis"]["height"]
        prosthesis[i+1] = observation["pelvis"]["pitch"]
        prosthesis[i+2] = observation["pelvis"]["roll"]
        prosthesis[i+3:i+9] = observation["pelvis"]["vel"]
        i += 9

        prosthesis[i+0] = observation["l_leg"]["joint"]["knee"]
        prosthesis[i+1] = observation["l_leg"]["joint"]["ankle"]
        prosthesis[i+2] = observation["l_leg"]["d_joint"]["knee"]
        prosthesis[i+3] = observation["l_leg"]["d_joint"]["ankle"]
        i += 4

        prosthesis[i+0] = observation["l_leg"]['force']['knee']
        prosthesis[i+1] = observation["l_leg"]['force']['ankle']
        prosthesis[i+2] = observation["l_leg"]['actuator']['knee']['speed']
        prosthesis[i+3] = observation["l_leg"]['actuator']['knee']['control']
        prosthesis[i+4] = observation["l_leg"]['actuator']['knee']['power']
        prosthesis[i+5] = observation["l_leg"]['actuator']['knee']['stress']
        prosthesis[i+6] = observation["l_leg"]['actuator']['knee']['actuation']
        prosthesis[i+7] = observation["l_leg"]['actuator']['ankle']['speed']
        prosthesis[i+8] = observation["l_leg"]['actuator']['ankle']['control']
        prosthesis[i+9] = observation["l_leg"]['actuator']['ankle']['power']
        prosthesis[i+10] = observation["l_leg"]['actuator']['ankle']['stress']
        prosthesis[i+11] = observation["l_leg"]['actuator']['ankle']['actuation']
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

    def imitation_reward(self, t, imitation_data, observation):
        obs = imi_learning_observ_columns(observation)
        diffs = np.sum(np.square(imitation_data - obs), axis=1)
        loss = np.nanmin(diffs)
        reward = math.exp(-1 * loss)
        print(reward)
        return reward

    # def imitation_reward(self, t, imitation_data, observation):
        # if len(imitation_data.index) >= t:
        #     return 1
        # imi = imitation_data
        # obs = observation
        # pelvis_loss =   (obs["pelvis"]["height"] - imi['pelvis_ty'][t])**2
        # pelvis_rot_loss =   (obs["pelvis"]["pitch"] - imi['pelvis_tilt'][t])**2 +\
        #                     (obs["pelvis"]["roll"] - imi['pelvis_list'][t])**2
        # ankle_loss =    (obs['l_leg']['joint']['ankle'] - imi['ankle_angle_l'][t])**2 +\
        #                 (obs['r_leg']['joint']['ankle'] - imi['ankle_angle_r'][t])**2
        # knee_loss = (obs['l_leg']['joint']['knee'] - imi['knee_angle_l'][t])**2 +\
        #             (obs['r_leg']['joint']['knee'] - imi['knee_angle_r'][t])**2
        # hip_loss =  (obs['l_leg']['joint']['hip']     - imi['hip_flexion_l'][t])**2 +\
        #             (obs['r_leg']['joint']['hip']     - imi['hip_flexion_r'][t])**2 +\
        #             (obs['l_leg']['joint']['hip_abd'] - imi['hip_adduction_l'][t])**2 +\
        #             (obs['r_leg']['joint']['hip_abd'] - imi['hip_adduction_r'][t])**2
        # total_position_loss = ankle_loss + knee_loss + hip_loss + pelvis_loss + pelvis_rot_loss
        # pos_reward = np.exp(-2 * total_position_loss)
        # # velocity losses
        # pelvis_rot_loss_v = (obs["pelvis"]["vel"][3] - imi['pelvis_tilt_speed'][t])**2 +\
        #                     (obs["pelvis"]["vel"][5] - imi['pelvis_rotation_speed'][t])**2 +\
        #                     (obs["pelvis"]["vel"][4] - imi['pelvis_list_speed'][t])**2
        # pelvis_loss_v = (obs["pelvis"]["vel"][0] - imi['pelvis_tx_speed'][t])**2 +\
        #                 (obs["pelvis"]["vel"][1] - imi['pelvis_ty_speed'][t])**2 +\
        #                 (obs["pelvis"]["vel"][2] - imi['pelvis_tz_speed'][t])**2 
        # ankle_loss_v =  (obs['l_leg']['d_joint']['ankle'] - imi['ankle_angle_l_speed'][t])**2 +\
        #                 (obs['r_leg']['d_joint']['ankle'] - imi['ankle_angle_r_speed'][t])**2 
        # knee_loss_v =   (obs['l_leg']['d_joint']['knee'] - imi['knee_angle_l_speed'][t])**2 +\
        #                 (obs['r_leg']['d_joint']['knee'] - imi['knee_angle_r_speed'][t])**2
        # hip_loss_v =    (obs['l_leg']['d_joint']['hip']     - imi['hip_flexion_l_speed'][t])**2 +\
        #                 (obs['r_leg']['d_joint']['hip']     - imi['hip_flexion_r_speed'][t])**2 +\
        #                 (obs['l_leg']['d_joint']['hip_abd'] - imi['hip_adduction_l_speed'][t])**2 +\
        #                 (obs['r_leg']['d_joint']['hip_abd'] - imi['hip_adduction_r_speed'][t])**2
        # total_velocity_loss = ankle_loss_v + knee_loss_v + hip_loss_v 
        # velocity_reward = np.exp(-0.1*total_velocity_loss) 
        # im_rew = pos_reward
        # # im_rew =  0.75*pos_reward + 0.25*velocity_reward
        # return im_rew


    def reward(self, t, observation):
        # target_velocity = np.sqrt(target_x_speed**2 + target_z_speed**2)
        # np_speeds = np.array(list(self.k_paths_dict.keys()))
        # closest_clip_ix = (np.abs(np_speeds - target_velocity)).argmin() 
        # closest_clip_speed = np_speeds[closest_clip_ix]

        # Constant reward of 0.5 with additional 0.5 for each footstep.
        step_rew = 1.0 if self.env.footstep['new'] else 0.5
        im_rew = self.imitation_reward(t, self.imitation_data_tensor[self.current_imitation_data], observation)
        state = self.env.get_state_desc()
        dist = math.sqrt(state["body_pos"]["pelvis"][0]**2 + state["body_pos"]["pelvis"][2]**2)
        dist_rev = math.tanh(dist * 0.2) + dist * 0.01

        # Comment as it leads to the first action being to fall down
        # However, commented it leads to not moving at all
        # if t<=50:
        #    return 0.1 * im_rew + 0.9 * step_rew

        return 0.5 * im_rew + 0.2 * step_rew + 0.3 * dist_rev

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

        start_time = time.time()
        action[15] = action[15] * 2 - 1
        action[16] = action[16] * 2 - 1
        try:
            observation, reward, done = self.env.step(action)
        except RuntimeError as err:
            print("Caught runtime error during step: {}".format(err))
            done = True

        time_diff = time.time() - start_time
        if time_diff > 0.99:
            print("Step took too long: {} sec".format(time_diff))

        self.t += 1
        
        observation = self.env.get_observation_dict()
        obs_array = self.observation_array_normalized(observation)
        
        reward = self.reward(self.t, observation)

        if math.isnan(reward):
            print("Rewards is NAN")
        if reward > 10.0 or reward < 0.0:
            print("Rewards is out of reasonable range: ", reward)

        return obs_array, reward, done, {}

    def reset(self):
        self.current_imitation_data = "075-FIX"#random.choice(list(self.imitation_data.keys()))
        imi = self.imitation_data[self.current_imitation_data]

        self.start_time = self.t = t = random.choice(self.imitation_data_start_times[self.current_imitation_data])

        init_pose = np.array([
            imi['pelvis_tx_speed'][t], # 0,  # forward speed 
            imi['pelvis_tz_speed'][t], # 0,  # rightward speed 
            imi['pelvis_ty'][t], # 0.94,  # pelvis height 
            imi['pelvis_list'][t], # 0 * np.pi / 180,  # trunk lean 
            imi['hip_adduction_r'][t], # 0 * np.pi / 180,  # [right] hip adduct 
            imi['hip_flexion_r'][t], # 0 * np.pi / 180,  # hip flex 
            imi['knee_angle_r'][t], # 0 * np.pi / 180,  # knee extend 
            imi['ankle_angle_r'][t], # 0 * np.pi / 180,  # ankle flex 
            imi['hip_adduction_l'][t], # 0 * np.pi / 180,  # [left] hip adduct
            imi['hip_flexion_l'][t], # 0 * np.pi / 180,  # hip flex
            imi['knee_angle_l'][t], # 0 * np.pi / 180,  # knee extend
            imi['ankle_angle_l'][t]]) # 0 * np.pi / 180])  # ankle flex

        # print("Pre init at {}".format(t))
        self.env.reset(init_pose=init_pose)
        # print("Post init")

        observation = self.env.get_observation_dict()
        array = self.observation_array_normalized(observation)

        return array