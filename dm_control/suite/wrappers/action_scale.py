
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

"""Wrapper that scales actions to a specific range."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dm_env
from dm_env import specs
import numpy as np

_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class Wrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""
  __slots__ = ("_action_spec", "_env", "_transform")

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.

    Args:
      env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
        consist of a single `BoundedArray` with all-finite bounds.
      minimum: Scalar or array-like specifying element-wise lower bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
      maximum: Scalar or array-like specifying element-wise upper bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.

    Raises:
      ValueError: If `env.action_spec()` is not a single `BoundedArray`.
      ValueError: If `env.action_spec()` has non-finite bounds.
      ValueError: If `minimum` or `maximum` contain non-finite values.
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `env.action_spec().shape`.
    """
    action_spec = env.action_spec()
    if not isinstance(action_spec, specs.BoundedArray):
      raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

    minimum = np.array(minimum)
    maximum = np.array(maximum)
    shape = action_spec.shape
    orig_minimum = action_spec.minimum
    orig_maximum = action_spec.maximum
    orig_dtype = action_spec.dtype

    def validate(bounds, name):
      if not np.all(np.isfinite(bounds)):
        raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))