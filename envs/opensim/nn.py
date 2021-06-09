import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import tensorflow_probability as tfp

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

class ModelRep(m.representation.ModelBaseSimpleRep):
    def call(self, obs_list):
        return obs_list[0]

# class ModelForward(m.ModelForward):
#     def __init__(self, state_dim, action_dim):
#         super().__init__(state_dim, action_dim,
#                          dense_n=state_dim + action_dim, dense_depth=1)


# class ModelRND(m.ModelBaseRND):
#     def __init__(self, state_dim, action_dim):
#         super().__init__(state_dim, action_dim)

#         self.dense = tf.keras.Sequential([
#             tf.keras.layers.Dense(32, activation=tf.nn.relu),
#             tf.keras.layers.Dense(32),
#         ])

#     def call(self, state, action):
#         return self.dense(tf.concat([state, action], axis=-1))


class ModelQ(m.ModelQ):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim,
                         dense_n=64, dense_depth=2)


class ModelPolicy(m.ModelBasePolicy):
    def __init__(self, state_dim, d_action_dim, c_action_dim, name=None):
        super().__init__(state_dim, d_action_dim, c_action_dim, name)
        assert d_action_dim == 0
        assert c_action_dim == 17
        assert state_dim == 97
        
        tstep = 10

        p_mean_depth = 1
        p_logstd_depth = 1
        p_mean_n = 32
        p_logstd_n = 32
        p_state_dim = 25
        p_action_dim = 2

        h_mean_depth = 1
        h_logstd_depth = 1
        h_mean_n = 32
        h_logstd_n = 32
        h_state_dim = 72
        h_action_dim = 15

        in_human = layers.Input(shape=(None, None, h_state_dim), name="in_human")
        layer_h_d1 = layers.Dense(100, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_h_d1")(in_human)
        layer_h_d2 = layers.Dense(312, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_h_d2")(layer_h_d1)
        
        h_mean_model = tf.keras.Sequential([
            layers.Dense(h_mean_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(h_mean_depth)] + [
            layers.Dense(h_action_dim, name='h_normal_output_dense')
        ], name='h_mean_seq')(layer_h_d2)
        h_logstd_model = tf.keras.Sequential([
            layers.Dense(h_logstd_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(h_logstd_depth)] + [
            layers.Dense(h_action_dim, name='h_normal_output_dense')
        ], name='h_logstd_seq')(layer_h_d2)

        in_prosthesis = layers.Input(shape=(None, None, p_state_dim), name="in_prosthesis")
        layer_p_l1 = layers.Dense(32, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_p_l1")(in_prosthesis)
        layer_p_l2 = layers.Dense(64, tf.nn.tanh, kernel_regularizer=regularizers.l2(0.001), name="layer_p_l2")(layer_p_l1)
        # layer_p_l3 = layers.Dense(32, name="layer_p_l3")(layer_p_l2)

        p_mean_model = tf.keras.Sequential([
            layers.Dense(p_mean_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(p_mean_depth)] + [
            layers.Dense(p_action_dim, name='p_normal_output_dense')
        ], name='p_mean_seq')(layer_p_l2)
        p_logstd_model = tf.keras.Sequential([
            layers.Dense(p_logstd_n, tf.nn.relu, kernel_regularizer=regularizers.l2(0.001)) for _ in range(p_logstd_depth)] + [
            layers.Dense(p_action_dim, name='p_normal_output_dense')
        ], name='p_logstd_seq')(layer_p_l2)

        out_mean = layers.concatenate([h_mean_model, p_mean_model], name="out_mean")
        out_logstd = layers.concatenate([h_logstd_model, p_logstd_model], name="out_logstd")

        self.c_tfpd = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.Normal(t[0], t[1]))



        self.model = tf.keras.Model(inputs=[in_human, in_prosthesis], outputs=[out_mean, out_logstd])
    
    def call(self, state):
        human, prosthesis = tf.split(state, [72, 25], -1)
        

        d_policy = tf.zeros((0,))

        mean, logstd = self.model([human, prosthesis])
        
        c_policy = self.c_tfpd([tf.tanh(mean), tf.clip_by_value(tf.exp(logstd), 0.1, 1.0)])


        return d_policy, c_policy
