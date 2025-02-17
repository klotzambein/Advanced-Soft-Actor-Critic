import torch
from torchvision import transforms

import algorithm.nn_models as m
from algorithm.nn_models.layers import Transform

from envs.realcar.nn import *


class ModelRep(ModelRep):
    def _bulid_model(self):
        super()._bulid_model()

        self.blurrer = Transform(transforms.GaussianBlur(3, sigma=3))

    def forward(self, obs_list):
        *vis, vec = obs_list
        vec = vec[..., :-EXTRA_SIZE]

        vis[1] = self.blurrer(vis[1])

        vis = self.conv(torch.cat(vis, dim=-1))

        state = self.dense(torch.cat([vis, vec], dim=-1))

        return state
