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

"""Procedurally generated LQR domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import xml_tools
from lxml import etree
import numpy as np
from six.moves import range

from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT = float('inf')
_CONTROL_COST_COEF = 0.1
SUITE = containers.TaggedTasks()


def get_model_and_assets(n_bodies, n_actuators, random):
  """Returns the model description as an XML string and a dict of assets.

  Args:
    n_bodies: An int, number of bodies of the LQR.
    n_actuators: An int, number of actuated bodies of the LQR. `n_actuators`
      should be less or equal than `n_bodies`.
    random: A `numpy.random.RandomState` instance.

  Returns:
    A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
    `{filename: contents_string}` pairs.
  """
  return _make_model(n_bodies, n_actuators, random), common.ASSETS


@SUITE.add()
def lqr_2_1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns an LQR environment with 2 bodies of which the first is actuated."""
  return _make_lqr(n_bodies=2,
                   n_actuators=1,
                   control_cost_coef=_CONTROL_COST_COEF,
                   time_limit=time_limit,
                   random=random,
                   environment_kwargs=environment_kwargs)


@SUITE.add()
def lqr_6_2(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns an LQR environment with 6 bodies of which first 2 are actuated."""
  return _make_lqr(n_bodies=6,
                   n_actuators=2,
                   control_cost_coef=_CONTROL_COST_COEF,
                   time_limit=time_limit,
                   random=random,
                   environment_kwargs=environment_kwargs)


def _make_lqr(n_bodies, n_actuators, control_cost_coef, time_limit, random,
              environment_kwargs):
  """Returns a LQR environment.

  Args:
    n_bodies: An int, number of bodies of the LQR.
    n_actuators: An int, number of actuated bodies of the LQR. `n_actuators`
      should be less or equal than `n_bodies`.
    control_cost_coef: A number, the coefficient of the control cost.
    time_limit: An int, maximum time for each episode in seconds.
    random: Either an existing `numpy.random.RandomState` instance, an
      integer seed for creating a new `RandomState`, or None to select a seed
      automatically.
    environment_kwargs: A `dict` specifying keyword arguments for the
      environment, or None.

  Returns:
    A LQR environment with `n_bodies` bodies of which first `n_actuators` are
    actuated.
  """

  if not isinstance(random, np.random.RandomState):
    random = np.random.RandomState(random)

  model_string, assets = get_model_and_assets(n_bodies, n_actuators,
                                              random=random)
  physics = Physics.from_xml_string(model_string, assets=assets)
  task = LQRLevel(control_cost_coef, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)


def _make_body(body_id, stiffness_range, damping_range, random):
  """Returns an `etree.Element` defining a body.

  Args:
    body_id: Id of the created body.
    stiffness_range: A tuple of (stiffness_lower_bound, stiffness_uppder_bound).
      The stiffness of the joint is drawn uniformly from this range.
    damping_range: A tuple of (damping_lower_bound, damping_upper_bound). The
      damping of the joint is drawn uniformly from this range.
    random: A `numpy.random.RandomState` instance.

  Returns:
   A new instance of `etree.Element`. A body element with two children: joint
   and geom.
  """
  body_name = 'body_{}'.format(body_id)
  joint_name = 'joint_{}'.format(body_id)
  geom_name = 'geom_{}'.format(body_id)

  body = etree.Element('body', name=body_name)
  body.set('pos', '.25 0 0')
  joint = etree.SubElement(body, 'joint', name=joint_name)
  body.append(etree.Element('geom', name=geom_name))
  joint.set('stiffness',
            str(random.uniform(stiffness_range[0], stiffness_range[1])))
  joint.set('damping',
            str(random.uniform(damping_range[0], damping_range[1])))
  return body


def _make_model(n_bodies,
                n_actuators,
                random,
                stiffness_range=(15, 25),
                damping_range=(0, 0)):
  """Returns an MJCF XML string defining a model of springs and dampers.

  Args:
    n_bodies: An integer, the number of bodies (DoFs) in the system.
    n_actuators: An integer, the number of actuated bodies.
    random: A `numpy.random.RandomState` instance.
    stiffness_range: A tuple containing minimum and maximum stiffness. Each
      joint's stiffness is sampled uniformly from this interval.
    damping_range: A tuple containing minimum and maximum damping. Each joint's
      damping is sampled uniformly from this interval.

  Returns:
    An MJCF 