
import wrappers
import tools
import models
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
import tensorflow as tf
import numpy as np
import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import soft_actor_critic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

tf.executing_eagerly()

tf.get_logger().setLevel('ERROR')

sys.path.append(str(pathlib.Path(__file__).parent))

from config import define_config as define_config
from tools import cal_result

class CVRL(tools.Module):

    def __init__(self, config, datadir, actspace, writer):
        self._c = config
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(
            actspace, 'n') else actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(
                datadir, config), dtype=tf.int64)
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']  # Create variable for checkpoint.
        self._float = prec.global_policy().compute_dtype
        self._strategy = tf.distribute.MirroredStrategy()
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(
                load_dataset(datadir, self._c)))
            self._build_model()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        if self._should_train(step) and not self._c.test:
            log = self._should_log(step)
            n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
            print(f'Training for {n} steps.')
            for train_step in range(n):
                log_images = self._c.log_images and log and train_step == 0
                self.train(next(self._dataset), log_images)
            if log:
                self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
            self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._actdim), self._float)
        else:
            latent, action = state

        obs = preprocess(obs, self._c)
        embed = self._encode_img(obs)
        if self._c.multi_modal:
            embed_depth = self._encode_dep(obs)
            embed_touch = self._encode_touch(obs["touch"])
            embed = tf.concat([embed, embed_depth, embed_touch], -1)
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)

        if self._c.trajectory_opt:
            action = self._trajectory_optimization(latent)
        elif self._c.forward_search:
            action = self._forward_search_policy(latent)
        else:
            if training:
                action = self._actor(feat).sample()
            else:
                action = self._actor(feat).mode()

        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=False):
        self._strategy.experimental_run_v2(self._train, args=(data, log_images))

    def _train(self, data, log_images):
        with tf.GradientTape() as model_tape:
            embed = self._encode_img(data)  # * data["img_flag"]
            if self._c.multi_modal:
                embed_depth = self._encode_dep(data)
                embed_touch = self._encode_touch(data["touch"])
                embed = tf.concat([embed, embed_depth, embed_touch], -1)

            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)
            reward_pred = self._reward(feat)
            likes = tools.AttrDict()
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
            image_pred = self._decode_img(feat)
            depth_pred = self._decode_dep(feat)
            cont_loss = self._contrastive(feat, embed)

            if not self._c.reward_only:
                if self._c.obs_model == 'generative':
                    likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
                    likes.depth = tf.reduce_mean(depth_pred.log_prob(data['depth']))
                elif self._c.obs_model == 'contrastive':
                    likes.image = tf.reduce_mean(cont_loss)
            else:
                likes.image = 0

            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale

            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)
            model_loss = self._c.kl_scale * div - sum(likes.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

        assert self._c.use_dreamer or self._c.use_sac

        if self._c.use_dreamer:
            with tf.GradientTape() as actor_tape:
                imag_feat = self._imagine_ahead(post)
                reward = self._reward(imag_feat).mode()
                if self._c.pcont:
                    pcont = self._pcont(imag_feat).mean()
                else:
                    pcont = self._c.discount * tf.ones_like(reward)
                value = self._value(imag_feat).mode()
                returns = tools.lambda_return(
                    reward[:-1], value[:-1], pcont[:-1],
                    bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
                discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                    [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
                actor_loss = -tf.reduce_mean(discount * returns)
                actor_loss /= float(self._strategy.num_replicas_in_sync)

            with tf.GradientTape() as value_tape:
                value_pred = self._value(imag_feat)[:-1]
                target = tf.stop_gradient(returns)
                value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
                value_loss /= float(self._strategy.num_replicas_in_sync)

            actor_norm = self._actor_opt(actor_tape, actor_loss)
            value_norm = self._value_opt(value_tape, value_loss)
        else:
            actor_norm = actor_loss = 0
            value_norm = value_loss = 0

        model_norm = self._model_opt(model_tape, model_loss)
        states = tf.concat([post['stoch'], post['deter']], axis=-1)
        rewards = data['reward']
        dones = tf.zeros_like(rewards)
        actions = data['action']