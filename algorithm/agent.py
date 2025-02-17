from collections import deque

import numpy as np


class Agent(object):
    reward = 0  # The reward of the first complete episode
    _last_reward = 0  # The reward of the last episode
    steps = 0  # The step count of the first complete episode
    _last_steps = 0  # The step count of the last episode
    done = False  # If has one complete episode
    max_reached = False

    def __init__(self, agent_id, use_rnn=False, max_return_episode_trans=-1):
        self.agent_id = agent_id
        self.use_rnn = use_rnn
        self.max_return_episode_trans = max_return_episode_trans

        self._tmp_episode_trans = list()

    def add_transition(self,
                       obs_list,
                       action,
                       reward,
                       local_done,
                       max_reached,
                       next_obs_list,
                       rnn_state=None):

        transition = {
            'obs_list': obs_list,
            'action': action,
            'reward': reward,
            'local_done': local_done,
            'max_reached': max_reached,
            'next_obs_list': next_obs_list,
            'rnn_state': rnn_state
        }
        self._tmp_episode_trans.append(transition)

        if not self.done:
            self.reward += reward
            self.steps += 1
        self._last_reward += reward
        self._last_steps += 1

        self._extra_log(obs_list,
                        action,
                        reward,
                        local_done,
                        max_reached,
                        next_obs_list)

        if local_done:
            self.done = True
            self.max_reached = max_reached
            self._last_reward = 0
            self._last_steps = 0

        if local_done or len(self._tmp_episode_trans) == self.max_return_episode_trans:
            episode_trans = self._get_episode_trans()
            self._tmp_episode_trans.clear()

            return episode_trans

    def _extra_log(self,
                   obs_list,
                   action,
                   reward,
                   local_done,
                   max_reached,
                   next_obs_list):
        pass

    def _get_episode_trans(self):
        obs_list = [t['obs_list'] for t in self._tmp_episode_trans]
        obs_list = [np.stack(t, axis=0) for t in zip(*obs_list)]
        obs_list = [np.expand_dims(t, 0).astype(np.float32) for t in obs_list]
        # list([1, ep_len, obs_shape_i], ...)

        action = np.stack([t['action'] for t in self._tmp_episode_trans], axis=0)
        action = np.expand_dims(action, 0).astype(np.float32)  # [1, ep_len, action_size]

        reward = np.stack([t['reward'] for t in self._tmp_episode_trans], axis=0)
        reward = np.expand_dims(reward, 0).astype(np.float32)  # [1, ep_len]

        next_obs_list = [np.expand_dims(t, 0).astype(np.float32)
                         for t in self._tmp_episode_trans[-1]['next_obs_list']]
        # list([1, obs_shape_i], ...)

        done = np.stack([t['local_done'] and not t['max_reached'] for t in self._tmp_episode_trans],
                        axis=0)
        done = np.expand_dims(done, 0).astype(np.float32)  # [1, ep_len]

        episode_trans = [obs_list, action, reward, next_obs_list, done]

        if self.use_rnn:
            rnn_state = np.stack([t['rnn_state'] for t in self._tmp_episode_trans], axis=0)
            rnn_state = np.expand_dims(rnn_state, 0).astype(np.float32)
            # [1, ep_len, rnn_state_dim]
            episode_trans.append(rnn_state)

        return episode_trans

    def is_empty(self):
        return len(self._tmp_episode_trans) == 0

    def clear(self):
        self.reward = 0
        self.steps = 0
        self.done = False
        self.max_reached = False
        self._tmp_episode_trans.clear()

    def reset(self):
        """
        The agent may continue in a new iteration but save its last status
        """
        self.reward = self._last_reward
        self.steps = self._last_steps
        self.done = False
        self.max_reached = False


if __name__ == "__main__":
    agent = Agent(0, False)
    for i in range(10):
        print(i, '===')
        a = agent.add_transition([np.random.randn(4), np.random.randn(3)],
                                 np.random.randn(2), 1, False, False,
                                 [np.random.randn(4), np.random.randn(3)])
        print(a)

    print(10, '===')
    a = agent.add_transition([np.random.randn(4), np.random.randn(3)],
                             np.random.randn(2), 1, True, False,
                             [np.random.randn(4), np.random.randn(3)])
    print(a)
