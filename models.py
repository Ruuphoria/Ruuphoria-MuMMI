
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools

class RSSM(tools.Module):

    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super().__init__()
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._cell = tfkl.GRUCell(self._deter_size)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        return dict(
            mean=tf.zeros([batch_size, self._stoch_size], dtype),
            std=tf.zeros([batch_size, self._stoch_size], dtype),
            stoch=tf.zeros([batch_size, self._stoch_size], dtype),
            deter=self._cell.get_initial_state(None, batch_size, dtype))

    @tf.function
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        embed = tf.transpose(embed, [1, 0, 2])
        action = tf.transpose(action, [1, 0, 2])
        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action, embed), (state, state))
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = tf.transpose(action, [1, 0, 2])
        prior = tools.static_scan(self.img_step, action, state)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        return tf.concat([state['stoch'], state['deter']], -1)

    def get_dist(self, state):
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])

    @tf.function
    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action):
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x, deter = self._cell(x, [prev_state['deter']])
        deter = deter[0]  # Keras wraps the state in a list.
        x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

class ConvEncoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, modality="image"):
        self._act = act
        self._depth = depth
        self._modality = modality

    def __call__(self, obs):
        kwargs = dict(strides=2, activation=self._act)
        x = tf.reshape(obs[self._modality], (-1,) + tuple(obs[self._modality].shape[-3:]))
        x = self.get(self._modality + 'h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
        x = self.get(self._modality + 'h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
        x = self.get(self._modality + 'h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
        x = self.get(self._modality + 'h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
        shape = tf.concat([tf.shape(obs[self._modality])[:-3], [32 * self._depth]], 0)
        return tf.reshape(x, shape)


class Embed2z(tools.Module):

    def __init__(self, modality, stoch=30, hidden=200, act=tf.nn.relu):
        super().__init__()
        self._stoch_size = stoch
        self._activation = act
        self._hidden_size = hidden
        self._modality = modality

    def __call__(self, embed):
        x = self.get(self._modality + 'e2z1', tfkl.Dense, self._hidden_size, self._activation)(embed)
        x = self.get(self._modality + 'e2z2', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        return {"mean": mean, "std": std}


class Dense(tools.Module):
    """
    MLP with n layer