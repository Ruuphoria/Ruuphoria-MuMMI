
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

"""Humanoid_CMU Domain."""

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

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('humanoid_CMU.xml'), common.ASSETS


@SUITE.add()
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HumanoidCMU(move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,