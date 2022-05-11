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

"""Tests specific to the LQR domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  math

# Internal dependencies.
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.suite import lqr
from dm_control.suite import lqr_solver
import numpy as np
from six.moves import range


class LqrTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('lqr_2_1', lqr.lqr_2_1),
      ('lqr_6_2', lqr.lqr_6_2))
  def test_lqr_optimal_policy(self, make_env):
    env = make_env()
    p, k, beta = lqr_solver.solve(env)
    self.assertPolicyisOptimal(env, p, k, beta)

  def assertPolicyisOptimal(self, env, p, k, beta):
    tolerance = 1e-