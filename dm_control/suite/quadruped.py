
# Copyright 2019 The dm_control Authors.
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

"""Quadruped Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np
from scipy import ndimage

enums = mjbindings.enums
mjlib = mjbindings.mjlib

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

SUITE = containers.TaggedTasks()


def make_model(floor_size=None, terrain=False, rangefinders=False,
               walls_and_ball=False):
    """Returns the model XML string."""
    xml_string = common.read_model('quadruped.xml')
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Set floor size.
    if floor_size is not None:
        floor_geom = mjcf.find('.//geom[@name={!r}]'.format('floor'))
        floor_geom.attrib['size'] = '{} {} .5'.format(floor_size, floor_size)

    # Remove walls, ball and target.
    if not walls_and_ball:
        for wall in _WALLS:
            wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
            wall_geom.getparent().remove(wall_geom)

        # Remove ball.
        ball_body = xml_tools.find_element(mjcf, 'body', 'ball')
        ball_body.getparent().remove(ball_body)

        # Remove target.
        target_site = xml_tools.find_element(mjcf, 'site', 'target')
        target_site.getparent().remove(target_site)

    # Remove terrain.
    if not terrain:
        terrain_geom = xml_tools.find_element(mjcf, 'geom', 'terrain')
        terrain_geom.getparent().remove(terrain_geom)

    # Remove rangefinders if they're not used, as range computations can be
    # expensive, especially in a scene with heightfields.
    if not rangefinders:
        rangefinder_sensors = mjcf.findall('.//rangefinder')
        for rf in rangefinder_sensors:
            rf.getparent().remove(rf)

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Move(desired_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _RUN_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Move(desired_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def escape(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
    """Returns the Escape task."""
    xml_string = make_model(floor_size=40, terrain=True, rangefinders=True)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Escape(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def fetch(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Fetch task."""
    xml_string = make_model(walls_and_ball=True)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Fetch(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Quadruped domain."""

    def _reload_from_data(self, data):
        super(Physics, self)._reload_from_data(data)
        # Clear cached sensor names when the physics is reloaded.
        self._sensor_types_to_names = {}
        self._hinge_names = []

    def _get_sensor_names(self, *sensor_types):
        try:
            sensor_names = self._sensor_types_to_names[sensor_types]
        except KeyError:
            [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
            sensor_names = [self.model.id2name(s_id, 'sensor') for s_id in sensor_ids]
            self._sensor_types_to_names[sensor_types] = sensor_names
        return sensor_names

    def torso_upright(self):
        """Returns the dot-product of the torso z-axis and the global z-axis."""
        return np.asarray(self.named.data.xmat['torso', 'zz'])

    def torso_velocity(self):
        """Returns the velocity of the torso, in the local frame."""
        return self.named.data.sensordata['velocimeter'].copy()

    def egocentric_state(self):
        """Returns the state without global orientation or position."""
        if not self._hinge_names:
            [hinge_ids] = np.nonzero(self.model.jnt_type ==
                                     enums.mjtJoint.mjJNT_HINGE)
            self._hinge_names = [self.model.id2name(j_id, 'joint')
                                 for j_id in hinge_ids]
        return np.hstack((self.named.data.qpos[self._hinge_names],
                          self.named.data.qvel[self._hinge_names],
                          self.data.act))

    def toe_positions(self):
        """Returns toe positions in egocentric frame."""
        torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
        torso_pos = self.named.data.xpos['torso']
        torso_to_toe = self.named.data.xpos[_TOES] - torso_pos