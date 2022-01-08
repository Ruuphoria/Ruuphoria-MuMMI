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

"""Finger Domain."""

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
import numpy as np
from six.moves import range

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)
# For TURN tasks, the 'tip' geom needs to enter a spherical target of sizes:
_EASY_TARGET_SIZE = 0.07
_HARD_TARGET_SIZE = 0.03
# Initial spin velocity for the Stop task.
_INITIAL_SPIN_VELOCITY = 100
# Spinning slower than this value (radian/second) is considered stopped.
_STOP_VELOCITY = 1e-6
# Spinning faster than this value (radian/second) is considered spinning.
_SPIN_VELOCITY = 15.0


SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('finger.xml'), common.ASSETS


@SUITE.add('benchmarking')
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Spin task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Spin(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def turn_easy(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the easy Turn task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Turn(target_radius=_EASY_TARGET_SIZE, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def turn_hard(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the hard Turn task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Turn(target_radius=_HARD_TARGET_SIZE, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Finger domain."""

  def touch(self):
    """Returns logarithmically scaled signals from the two touch sensors."""
    return np.log1p(self.named.data.sensordata[['touchtop', 'touchbottom']])

  def hinge_velocity(self):
    """Returns the velocity of the hinge joint."""
    return self.named.data.sensordata['hinge_velocity']

  def tip_position(self):
    """Returns the (x,z) pos