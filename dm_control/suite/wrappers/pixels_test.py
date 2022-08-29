
# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for the pixel wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.suite import cartpole
from dm_control.suite.wrappers import pixels
import dm_env
from dm_env import specs

import numpy as np


class FakePhysics(object):

  def render(self, *args, **kwargs):
    del args
    del kwargs
    return np.zeros((4, 5, 3), dtype=np.uint8)


class FakeArrayObservationEnvironment(dm_env.Environment):

  def __init__(self):
    self.physics = FakePhysics()

  def reset(self):
    return dm_env.restart(np.zeros((2,)))

  def step(self, action):
    del action
    return dm_env.transition(0.0, np.zeros((2,)))

  def action_spec(self):
    pass

  def observation_spec(self):
    return specs.Array(shape=(2,), dtype=np.float)


class PixelsTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_dict_observation(self, pixels_only):
    pixel_key = 'rgb'

    env = cartpole.swingup()

    # Make sure we are testing the right environment for the test.
    observation_spec = env.observation_spec()
    self.assertIsInstance(observation_spec, collections.OrderedDict)
