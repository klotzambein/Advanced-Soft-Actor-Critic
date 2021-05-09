import importlib
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import algorithm.config_helper as config_helper

from .agent import Agent
from .sac_base import SAC_Base


class Main(object):
    train_mode = True
    _agent_class = Agent  # For different environments

    def __init__(self, root_dir, config_dir, args):
        """
        config_path: the directory of config file
        args: command arguments generated by argparse
        """
        self.logger = logging.getLogger('sac')

        config_abs_dir = self._init_config(root_dir, config_dir, args)

        self._init_env(config_abs_dir)
        self._run()

    def _init_config(self, root_dir, config_dir, args):
        config_abs_dir = Path(root_dir).joinpath(config_dir)
        config_abs_path = config_abs_dir.joinpath('config.yaml')
        default_config_abs_path = Path(__file__).resolve().parent.joinpath('default_config.yaml')
        # Merge default_config.yaml and custom config.yaml
        config = config_helper.initialize_config_from_yaml(default_config_abs_path,
                                                           config_abs_path,
                                                           args.config)

        # Initialize config from command line arguments
        self.device = args.device
        self.train_mode = not args.run
        self.last_ckpt = args.ckpt
        self.render = args.render
        self.run_in_editor = args.editor

        if args.name is not None:
            config['base_config']['name'] = args.name
        if args.port is not None:
            config['base_config']['port'] = args.port
        if args.nn is not None:
            config['base_config']['nn'] = args.nn
        if args.agents is not None:
            config['base_config']['n_agents'] = args.agents

        config['base_config']['name'] = config_helper.generate_base_name(config['base_config']['name'])

        # The absolute directory of a specific training
        model_abs_dir = Path(root_dir).joinpath('models',
                                                config['base_config']['scene'],
                                                config['base_config']['name'])
        model_abs_dir.mkdir(parents=True, exist_ok=True)
        self.model_abs_dir = model_abs_dir

        if args.logger_in_file:
            config_helper.set_logger(Path(model_abs_dir).joinpath(f'log.log'))

        if self.train_mode:
            config_helper.save_config(config, model_abs_dir, 'config.yaml')

        config_helper.display_config(config, self.logger)

        self.base_config = config['base_config']
        self.reset_config = config['reset_config']
        self.replay_config = config['replay_config']
        self.sac_config = config['sac_config']

        return config_abs_dir

    def _init_env(self, config_abs_dir: Path):
        if self.base_config['env_type'] == 'UNITY':
            from algorithm.env_wrapper.unity_wrapper import UnityWrapper

            if self.run_in_editor:
                self.env = UnityWrapper(train_mode=self.train_mode,
                                        base_port=5004,
                                        n_agents=self.base_config['n_agents'])
            else:
                self.env = UnityWrapper(train_mode=self.train_mode,
                                        file_name=self.base_config['build_path'][sys.platform],
                                        base_port=self.base_config['port'],
                                        no_graphics=self.base_config['no_graphics'] and not self.render,
                                        scene=self.base_config['scene'],
                                        n_agents=self.base_config['n_agents'])

        elif self.base_config['env_type'] == 'GYM':
            from algorithm.env_wrapper.gym_wrapper import GymWrapper

            self.env = GymWrapper(train_mode=self.train_mode,
                                  env_name=self.base_config['build_path'],
                                  render=self.render,
                                  n_agents=self.base_config['n_agents'])
        else:
            raise RuntimeError(f'Undefined Environment Type: {self.base_config["env_type"]}')

        self.obs_shapes, self.d_action_size, self.c_action_size = self.env.init()
        self.action_size = self.d_action_size + self.c_action_size

        # If model exists, load saved model, or copy a new one
        nn_model_abs_path = Path(self.model_abs_dir).joinpath('nn_models.py')
        if nn_model_abs_path.exists():
            spec = importlib.util.spec_from_file_location('nn', str(nn_model_abs_path))
        else:
            nn_abs_path = Path(config_abs_dir).joinpath(f'{self.base_config["nn"]}.py')
            spec = importlib.util.spec_from_file_location('nn', str(nn_abs_path))
            shutil.copyfile(nn_abs_path, nn_model_abs_path)

        custom_nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_nn_model)

        self.sac = SAC_Base(obs_shapes=self.obs_shapes,
                            d_action_size=self.d_action_size,
                            c_action_size=self.c_action_size,
                            model_abs_dir=self.model_abs_dir,
                            model=custom_nn_model,
                            device=self.device,
                            train_mode=self.train_mode,
                            last_ckpt=self.last_ckpt,

                            replay_config=self.replay_config,

                            **self.sac_config)

    def _run(self):
        use_rnn = self.sac.use_rnn

        obs_list = self.env.reset(reset_config=self.reset_config)

        agents = [self._agent_class(i, use_rnn=self.sac.use_rnn)
                  for i in range(self.base_config['n_agents'])]

        if use_rnn:
            initial_rnn_state = self.sac.get_initial_rnn_state(len(agents))
            rnn_state = initial_rnn_state

        is_max_reached = False
        iteration = 0
        trained_steps = 0

        while iteration != self.base_config['max_iter']:
            if self.base_config['max_step'] != -1 and trained_steps >= self.base_config['max_step']:
                break

            if self.base_config['reset_on_iteration'] or is_max_reached:
                obs_list = self.env.reset(reset_config=self.reset_config)
                for agent in agents:
                    agent.clear()

                if use_rnn:
                    rnn_state = initial_rnn_state
            else:
                for agent in agents:
                    agent.reset()

            is_max_reached = False
            action = np.zeros([len(agents), self.action_size], dtype=np.float32)
            step = 0

            while False in [a.done for a in agents]:
                if use_rnn:
                    # burn-in padding
                    for agent in [a for a in agents if a.is_empty()]:
                        for _ in range(self.sac.burn_in_step):
                            agent.add_transition([np.zeros(t) for t in self.obs_shapes],
                                                 np.zeros(self.action_size),
                                                 0, False, False,
                                                 [np.zeros(t) for t in self.obs_shapes],
                                                 initial_rnn_state[0])

                    action, next_rnn_state = self.sac.choose_rnn_action([o.astype(np.float32) for o in obs_list],
                                                                        action,
                                                                        rnn_state)
                else:
                    action = self.sac.choose_action([o.astype(np.float32) for o in obs_list])

                action = action.numpy()

                if np.isnan(np.sum(action)):
                    print("Oh no we found a NaN")

                next_obs_list, reward, local_done, max_reached = self.env.step(action[..., :self.d_action_size],
                                                                               action[..., self.d_action_size:])

                if step == self.base_config['max_step_each_iter']:
                    local_done = [True] * len(agents)
                    max_reached = [True] * len(agents)
                    is_max_reached = True

                episode_trans_list = [agents[i].add_transition([o[i] for o in obs_list],
                                                               action[i],
                                                               reward[i],
                                                               local_done[i],
                                                               max_reached[i],
                                                               [o[i] for o in next_obs_list],
                                                               rnn_state[i] if use_rnn else None)
                                      for i in range(len(agents))]

                if self.train_mode:
                    episode_trans_list = [t for t in episode_trans_list if t is not None]
                    if len(episode_trans_list) != 0:
                        # n_obses_list, n_actions, n_rewards, next_obs_list, n_dones,
                        # n_rnn_states
                        for episode_trans in episode_trans_list:
                            self.sac.fill_replay_buffer(*episode_trans)
                    trained_steps = self.sac.train()

                obs_list = next_obs_list
                action[local_done] = np.zeros(self.action_size)
                if use_rnn:
                    rnn_state = next_rnn_state
                    rnn_state[local_done] = initial_rnn_state[local_done]

                step += 1

            if self.train_mode:
                self._log_episode_summaries(agents)

            self._log_episode_info(iteration, agents)

            if (p := self.model_abs_dir.joinpath('save_model')).exists():
                self.sac.save_model()
                p.unlink()

            iteration += 1

        self.sac.save_model()
        self.env.close()

    def _log_episode_summaries(self, agents):
        rewards = np.array([a.reward for a in agents])
        self.sac.write_constant_summaries([
            {'tag': 'reward/mean', 'simple_value': rewards.mean()},
            {'tag': 'reward/max', 'simple_value': rewards.max()},
            {'tag': 'reward/min', 'simple_value': rewards.min()}
        ])

    def _log_episode_info(self, iteration, agents):
        rewards = [a.reward for a in agents]
        rewards = ", ".join([f"{i:6.1f}" for i in rewards])
        steps = [a.steps for a in agents]
        self.logger.info(f'{iteration}, S {max(steps)}, R {rewards}')
