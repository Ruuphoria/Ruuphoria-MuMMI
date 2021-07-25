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

"""A collection of MuJoCo-based Reinforcement Learning environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import itertools

from dm_control.rl import control

from dm_control.suite import acrobot
from dm_control.suite import ball_in_cup
from dm_control.suite import cartpole
from dm_control.suite import cheetah
from dm_control.suite import dog
from dm_control.suite import finger
from dm_control.suite import fish
from dm_control.suite import hopper
from dm_control.suite import humanoid
from dm_control.suite import humanoid_CMU
from dm_control.suite import lqr
from dm_control.suite import manipulator
from dm_control.suite import pendulum
from dm_control.suite import