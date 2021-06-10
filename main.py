import argparse
from pathlib import Path
import sys
import traceback

from algorithm.config_helper import set_logger

import gym
from env import OpenSimEnv

gym.envs.register(
    id="MyOpensimEnv-v1", 
    entry_point=OpenSimEnv,
    max_episode_steps=1000,
    kwargs = {"visualize": True, "data_dir": "/home/robin/Desktop/rug-bachelor-project/data"}
)

if __name__ == '__main__':
    set_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--config', '-c', help='config file')
    parser.add_argument('--run', action='store_true', help='inference mode')
    parser.add_argument('--render', action='store_true', help='render')
    parser.add_argument('--editor', action='store_true', help='running in Unity Editor')
    parser.add_argument('--logger_in_file', action='store_true', help='logging into a file')
    parser.add_argument('--name', '-n', help='training name')
    parser.add_argument('--port', '-p', type=int, default=5005, help='communication port')
    parser.add_argument('--nn', help='neural network model')
    parser.add_argument('--device', help='cpu or gpu')
    parser.add_argument('--ckpt', help='ckeckpoint to restore')
    parser.add_argument('--agents', type=int, help='number of agents')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeated experiments')
    args = parser.parse_args()

    if args.env in ['roller', 'square', 'pyramid', 'antisubmarine', 'usv']:
        from algorithm.sac_main_hitted import MainHitted as Main
    else:
        from algorithm.sac_main import Main

    root_dir = Path(__file__).resolve().parent
    if sys.platform == 'win32':
        for _ in range(args.repeat):
            Main(root_dir, f'envs/{args.env}', args)
    elif sys.platform == 'linux':
        for i in range(args.repeat):
            try:
                Main(root_dir, f'envs/{args.env}', args)
            except Exception as e:
                print("Error occurred: ", traceback.format_exc())
                # sys.exit()
                os.kill(os.getpid(), 9)

            args.port += 1
    
