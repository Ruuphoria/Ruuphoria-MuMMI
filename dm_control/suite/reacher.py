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

"""Reacher domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('reacher.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Reacher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Reacher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinates."""
    return (self.named.data.geom_xpos['target', :2] -
            self.named.data.geom_xpos['finger', :2])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
  """A reacher `Task` to reach the target."""

  def __init__(self, target_size, random=None):
    """Initialize an instance of `Reacher`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
       