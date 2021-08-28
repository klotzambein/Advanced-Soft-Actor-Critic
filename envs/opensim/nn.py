import torch
from torch import nn

import algorithm.nn_models as m

# Actuators
# 0 abd_r
# 1 add_r
# 2 hamstrings_r
# 3 bifemsh_r
# 4 glut_max_r
# 5 iliopsoas_r
# 6 rect_fem_r
# 7 vasti_r
# 8 gastroc_r
# 9 soleus_r
# 10 tib_ant_r
# 11 abd_l
# 12 add_l
# 13 glut_max_l
# 14 iliopsoas_l
# 15 knee_actuator
# 16 ankle_actuator

import algorithm.nn_models as m


ModelRep = m.ModelSimpleRep

class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(dense_n=64, dense_depth=2)

class ModelPolicy(m.ModelBasePolicy):
    def _build_model(self):
        assert self.d_action_size == 0
        assert self.c_action_size == 17
        assert self.state_size == 97

        p_mean_n = 32
        p_logstd_n = 32
        p_state_size = 25
        p_action_size = 2

        h_mean_n = 32
        h_logstd_n = 32
        h_state_size = 72 + p_state_size
        h_action_size = 15 + p_action_size

        self.h_common = nn.Sequential(
            nn.Linear(h_state_size, 312),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(312, 312),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.h_mean = nn.Sequential(
            nn.Linear(312, h_mean_n),
            nn.ReLU(),
            nn.Linear(h_mean_n, h_action_size),
            nn.Tanh()
        )

        self.h_logstd = nn.Sequential(
            nn.Linear(312, h_logstd_n),
            nn.ReLU(),
            nn.Linear(h_logstd_n, h_action_size),
            nn.Tanh()
        )

        self.p_common = nn.Sequential(
            nn.Linear(p_state_size, 100),
            nn.ReLU(),
            nn.Linear(100, 312),
            nn.ReLU()
        )

        self.p_mean = nn.Sequential(
            nn.Linear(312, p_mean_n),
            nn.ReLU(),
            nn.Linear(p_mean_n, p_action_size),
            nn.Tanh()
        )

        self.p_logstd = nn.Sequential(
            nn.Linear(312, p_logstd_n),
            nn.ReLU(),
            nn.Linear(p_logstd_n, p_action_size),
            nn.Tanh()
        )


    def forward(self, state):
        state_dims = state.dim()
        
        # human, prosthesis = torch.split(state, [72, 25], -1)

        h_common = self.h_common(state)
        h_mean = self.h_mean(h_common)
        h_logstd = self.h_logstd(h_common)

        # p_common = self.p_common(prosthesis)
        # p_mean = self.p_mean(p_common)
        # p_logstd = self.p_logstd(p_common)

        # mean = torch.cat((h_mean, p_mean), state_dims - 1)
        # logstd = torch.cat((h_logstd, p_logstd), state_dims - 1)

        mean = h_mean
        logstd = h_logstd

        c_policy = torch.distributions.Normal(mean, torch.clamp(torch.exp(logstd), 0.1, 1.0))

        d_policy = torch.zeros((0,))
        return d_policy, c_policy