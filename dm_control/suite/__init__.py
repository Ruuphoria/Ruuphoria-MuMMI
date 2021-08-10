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
from dm_control.suite import point_mass
from dm_control.suite import quadruped
from dm_control.suite import reacher
from dm_control.suite import stacker
from dm_control.suite import swimmer
from dm_control.suite import walker

# Find all domains imported.
_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def _get_tasks(tag):
  """Returns a sequence of (domain name, task name) pairs for the given tag."""
  result = []

  for domain_name in sorted(_DOMAINS.keys()):

    domain = _DOMAINS[domain_name]

    if tag is None:
      tasks_in_domain = domain.SUITE
    else:
      tasks_in_domain = domain.SUITE.tagged(tag)

    for task_name in tasks_in_domain.keys():
      result.append((domain_name, task_name))

  return tuple(result)


def _get_tasks_by_domain(tasks):
  """Returns a dict mapping from task name to a tuple of domain names."""
  result = collections.defaultdict(list)

  for domain_name, task_name in tasks:
    result[domain_name].append(task_name)

  return {k: tuple(v) for k, v in result.items()}


# A sequence containing all (domain name, task name) pairs.
ALL_TASKS = _get_tasks(tag=None)

# Subsets of ALL_TASKS, generated via the tag mechanism.
BENCHMARKING = _get_tasks('benchmarking')
EASY = _get_tasks('easy')
HARD = _get_tasks('hard')
EXTRA = tuple(sorted(set(ALL_TASKS) - set(BENCHMARKING)))
NO_REWARD_VIZ = _get_tasks('no_reward_visualization')
REWARD_VIZ = tuple(sorted(set(ALL_TASKS) - set(NO_REWARD_VIZ)))

# A mapping from each domain name to a sequence of its task names.
TASKS_BY_DOMAIN = _get_tasks_by_domain(ALL_TASKS)


def load(domain_name, task_name, 