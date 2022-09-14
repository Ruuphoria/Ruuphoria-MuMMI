
import atexit
import functools
import sys
import pickle
import threading
import traceback

import gym
import numpy as np
from PIL import Image


# wrapper
class DeepMindControl

    def __init__(self, name, size=(64, 64), camera=None):
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'], obs['depth'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'], obs['depth'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        rgb = self._env.physics.render(*self._size, camera_id=self._camera)
        depth = self._env.physics.render(*self._size, camera_id=self._camera, depth=True)
        depth = depth[:, :, np.newaxis]
        return rgb, depth


class Atari:
    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky_actions=True):
        import gym
        version = 0 if sticky_actions else 4
        name = ''.join(word.title() for word in name.split('_'))
        with self.LOCK:
            self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
        self._action_repeat = action_repeat
        self._size = size
        self._grayscale = grayscale
        self._noops = noops
        self._life_done = life_done
        self._lives = None
        shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
        self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
        self._random = np.random.RandomState(seed=None)

    @property
    def observation_space(self):
        shape = self._size + (1 if self._grayscale else 3,)
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({'image': space})

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            self._env.reset()
        noops = self._random.randint(1, self._noops + 1)
        for _ in range(noops):
            done = self._env.step(0)[2]
            if done:
                with self.LOCK:
                    self._env.reset()
        self._lives = self._env.ale.lives()
        if self._grayscale:
            self._env.ale.getScreenGrayscale(self._buffers[0])
        else:
            self._env.ale.getScreenRGB2(self._buffers[0])
        self._buffers[1].fill(0)
        return self._get_obs()

    def step(self, action):
        total_reward = 0.0
        for step in range(self._action_repeat):
            _, reward, done, info = self._env.step(action)
            total_reward += reward
            if self._life_done:
                lives = self._env.ale.lives()
                done = done or lives < self._lives
                self._lives = lives
            if done:
                break
            elif step >= self._action_repeat - 2:
                index = step - (self._action_repeat - 2)
                if self._grayscale:
                    self._env.ale.getScreenGrayscale(self._buffers[index])
                else:
                    self._env.ale.getScreenRGB2(self._buffers[index])
        obs = self._get_obs()
        return obs, total_reward, done, info

    def render(self, mode):
        return self._env.render(mode)

    def _get_obs(self):
        if self._action_repeat > 1:
            np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
        image = np.array(Image.fromarray(self._buffers[0]).resize(
            self._size, Image.BILINEAR))
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = image[:, :, None] if self._grayscale else image
        return {'image': image}


class Collect:

    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or () 
        self._precision = precision
        self._episode = None

    def __getattr__(self, name): 
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action) 
        obs = {k: self._convert(v) for k, v in obs.items()}  # obs {k,v}->{k,convert(v)}
        transition = obs.copy() 
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(done)))
        self._episode.append(transition)  
        if done:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info['episode'] = episode
            for callback in self._callbacks:
                callback(episode) 
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition['action'] = np.zeros(self._env.action_space.shape)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):