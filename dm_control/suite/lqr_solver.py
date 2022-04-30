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

r"""Optimal policy for LQR levels.

LQR control problem is described in
https://en.wikipedia.org/wiki/Linear-quadratic_regulator#Infinite-horizon.2C_discrete-time_LQR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco import wrapper
import numpy as np
import scipy.linalg as scipy_linalg


def solve(env):
  """Returns the optimal value and policy for LQR problem.

  Args:
    env: An instance of `control.EnvironmentV2` with LQR level.

  Returns:
    p: A numpy array, the Hessian of the optimal total cost-to-go (value
      function at state x) is V(x) = .5 * x' * p * x.
    k: A numpy array which gives the optimal linear policy u = k * x.
    beta: The maximum eigenvalue of (a + b * k). Under optimal policy, at
      timestep n the state tends to 0 like beta^n.

  Raises:
    RuntimeError: If the controlled system is unstable.
  """
  n = env.physics.model.nq  # number of DoFs
  m = env.physics.model.nu  # number of controls

  # Compute the mass matrix.
  mass = np.zeros((n, n))
  wrapper.mjbindings.mjlib.mj_fullM(env.physics.model.ptr, mass,
                                    env.physics.data.qM)

  # Compute input matrices a, b, q and r to the DARE solvers.
  # State transition matrix a.
  stiffness = np.diag(env.physics.model.jnt_stiffness.ravel())
  damping = np.diag(env.physics.model.dof_damping.ravel())
  dt = env.physics.model.