from osim import RUGTFPEnv, Pose
from time import sleep

import pandas as pd
import numpy as np
import math

env = RUGTFPEnv("OS4_gait14dof15musc_2act_LTFP_VR_Joint_Fix.osim", visualize=True)

pose = Pose()

data = pd.read_csv("/home/robin/Desktop/rug-bachelor-project/data/new.csv", sep=',')

for i in range(1350, len(data), 1):
    print(data.iloc[i]['time'])
    pose.set_from_dict_degrees(data.iloc[i])
    sleep(0.01)
    env.reset(pose)
    env.step(np.zeros(17))
