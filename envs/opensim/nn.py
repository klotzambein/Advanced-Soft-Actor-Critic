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
# class ModelRep(m.representation.ModelBaseSimpleRep):
#     def call(self, obs_list):
#         return obs_list[0]


class ModelQ(m.ModelQ):
    def _build_model(self):
        return super()._build_model(dense_n=64, dense_depth=2)


# class ModelQ(m.ModelQ):
#     def __init__(self, state_size, d_action_size, c_action_size, name=None):
#         super().__init__(state_size, d_action_size, c_action_size,
#                          dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelBasePolicy):
    def _build_model(self):
        assert self.d_action_size == 0
        assert self.c_action_size == 17
        assert self.state_size == 97

        p_mean_depth = 1
        p_logstd_depth = 1
        p_mean_n = 32
        p_logstd_n = 32
        p_state_size = 25
        p_action_size = 2

        h_mean_depth = 1
        h_logstd_depth = 1
        h_mean_n = 32
        h_logstd_n = 32
        h_state_size = 72
        h_action_size = 15

        self.h_common = nn.Sequential(
            nn.Linear(h_state_size, 100),
            nn.ReLU(),
            nn.Linear(100, 312),
            nn.ReLU()
        )
        # in_human = layers.Input(
        #     shape=(None, None, h_state_size), name="in_human")
        # layer_h_d1 = layers.Dense(100, tf.nn.tanh, kernel_regularizer=regularizers.l2(
        #     0.001), name="layer_h_d1")(in_human)
        # layer_h_d2 = layers.Dense(312, tf.nn.tanh, kernel_regularizer=regularizers.l2(
        #     0.001), name="layer_h_d2")(layer_h_d1)

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

        # h_mean_model = tf.keras.Sequential([
        #     layers.Dense(h_mean_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(h_mean_depth)] + [
        #     layers.Dense(h_action_size, name='h_normal_output_dense')
        # ], name='h_mean_seq')(layer_h_d2)
        # h_logstd_model = tf.keras.Sequential([
        #     layers.Dense(h_logstd_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(h_logstd_depth)] + [
        #     layers.Dense(h_action_size, name='h_normal_output_dense')
        # ], name='h_logstd_seq')(layer_h_d2)


        self.p_common = nn.Sequential(
            nn.Linear(p_state_size, 100),
            nn.ReLU(),
            nn.Linear(100, 312),
            nn.ReLU()
        )

        # in_prosthesis = layers.Input(
        #     shape=(None, None, p_state_size), name="in_prosthesis")
        # layer_p_l1 = layers.Dense(32, tf.nn.tanh, kernel_regularizer=regularizers.l2(
        #     0.001), name="layer_p_l1")(in_prosthesis)
        # layer_p_l2 = layers.Dense(64, tf.nn.tanh, kernel_regularizer=regularizers.l2(
        #     0.001), name="layer_p_l2")(layer_p_l1)
        # # layer_p_l3 = layers.Dense(32, name="layer_p_l3")(layer_p_l2)

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

        # p_mean_model = tf.keras.Sequential([
        #     layers.Dense(p_mean_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(p_mean_depth)] + [
        #     layers.Dense(p_action_size, name='p_normal_output_dense')
        # ], name='p_mean_seq')(layer_p_l2)
        # p_logstd_model = tf.keras.Sequential([
        #     layers.Dense(p_logstd_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(p_logstd_depth)] + [
        #     layers.Dense(p_action_size, name='p_normal_output_dense')
        # ], name='p_logstd_seq')(layer_p_l2)

        # out_mean = layers.concatenate(
        #     [h_mean_model, p_mean_model], name="out_mean")
        # out_logstd = layers.concatenate(
        #     [h_logstd_model, p_logstd_model], name="out_logstd")

        # self.c_tfpd = tfp.layers.DistributionLambda(
        #     make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))

        # self.model = tf.keras.Model(
        #     inputs=[in_human, in_prosthesis], outputs=[out_mean, out_logstd])

    def forward(self, state):
        state_dims = state.dim()
        
        human, prosthesis = torch.split(state, [72, 25], -1)

        h_common = self.h_common(human)
        h_mean = self.h_mean(h_common)
        h_logstd = self.h_logstd(h_common)

        p_common = self.p_common(prosthesis)
        p_mean = self.p_mean(p_common)
        p_logstd = self.p_logstd(p_common)

        mean = torch.cat((h_mean, p_mean), state_dims - 1)
        logstd = torch.cat((h_logstd, p_logstd), state_dims - 1)

        c_policy = torch.distributions.Normal(torch.tanh(mean), torch.clamp(torch.exp(logstd), 0.1, 1.0))

        d_policy = torch.zeros((0,))
        return d_policy, c_policy

# class ModelPolicy(m.ModelBasePolicy):
#     def __init__(self, state_size, d_action_size, c_action_size, name=None):
#         super().__init__(state_size, d_action_size, c_action_size, name)
#         assert d_action_size == 0
#         assert c_action_size == 17
#         assert state_size == 97

#         p_mean_depth = 1
#         p_logstd_depth = 1
#         p_mean_n = 32
#         p_logstd_n = 32
#         p_state_size = 25
#         p_action_size = 2

#         h_mean_depth = 1
#         h_logstd_depth = 1
#         h_mean_n = 32
#         h_logstd_n = 32
#         h_state_size = 72
#         h_action_size = 15

#         in_human = layers.Input(shape=(None, None, h_state_size), name="in_human")
#         layer_h_d1 = layers.Dense(100, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_h_d1")(in_human)
#         layer_h_d2 = layers.Dense(312, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_h_d2")(layer_h_d1)

#         h_mean_model = tf.keras.Sequential([
#             layers.Dense(h_mean_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(h_mean_depth)] + [
#             layers.Dense(h_action_size, name='h_normal_output_dense')
#         ], name='h_mean_seq')(layer_h_d2)
#         h_logstd_model = tf.keras.Sequential([
#             layers.Dense(h_logstd_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(h_logstd_depth)] + [
#             layers.Dense(h_action_size, name='h_normal_output_dense')
#         ], name='h_logstd_seq')(layer_h_d2)

#         in_prosthesis = layers.Input(shape=(None, None, p_state_size), name="in_prosthesis")
#         layer_p_l1 = layers.Dense(32, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_p_l1")(in_prosthesis)
#         layer_p_l2 = layers.Dense(64, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_p_l2")(layer_p_l1)
#         # layer_p_l3 = layers.Dense(32, name="layer_p_l3")(layer_p_l2)

#         p_mean_model = tf.keras.Sequential([
#             layers.Dense(p_mean_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(p_mean_depth)] + [
#             layers.Dense(p_action_size, name='p_normal_output_dense')
#         ], name='p_mean_seq')(layer_p_l2)
#         p_logstd_model = tf.keras.Sequential([
#             layers.Dense(p_logstd_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(p_logstd_depth)] + [
#             layers.Dense(p_action_size, name='p_normal_output_dense')
#         ], name='p_logstd_seq')(layer_p_l2)

#         out_mean = layers.concatenate([h_mean_model, p_mean_model], name="out_mean")
#         out_logstd = layers.concatenate([h_logstd_model, p_logstd_model], name="out_logstd")

#         self.c_tfpd = tfp.layers.DistributionLambda(
#             make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))


#         self.model = tf.keras.Model(inputs=[in_human, in_prosthesis], outputs=[out_mean, out_logstd])

#     def call(self, state):
#         human, prosthesis = tf.split(state, [72, 25], -1)


#         d_policy = tf.zeros((0,))

#         mean, logstd = self.model([human, prosthesis])

#         c_policy = self.c_tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])


#         return d_policy, c_policy
