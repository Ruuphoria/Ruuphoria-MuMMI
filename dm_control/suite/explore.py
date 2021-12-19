# Copyright 2018 The dm_control Authors.
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
"""Control suite environments explorer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from dm_control import suite
from dm_control.suite.wrappers import action_noise
from six.moves import input

from dm_control import viewer


_ALL_NAMES = ['.'.join(domain_task) for domain_task in suite.ALL_TASKS]

flags.DEFINE_enum('environment_name', None, _ALL_NAMES,
                