# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities related to cscg_he."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def get_default_emission_matrix(
    n_clones: np.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> np.ndarray:
  """Get the emission matrix.

  Args:
    n_clones: An array containing number of clones for each emission
    dtype: dtype for emission matrix

  Returns:
    emission_matrix
  """
  emission_matrix = np.zeros((n_clones.sum(), len(n_clones)), dtype=np.float32)
  n_clones_cumsum = np.hstack((0, n_clones.cumsum()))
  for i in range(len(n_clones)):
    m, n = n_clones_cumsum[i : i + 2]
    emission_matrix[m:n, i] = 1
  return np.array(emission_matrix, dtype=dtype)


def bcast_local_devices(
    value: np.ndarray | jax.Array,
    devices: list[jax.Device],
):
  """Broadcasts an object to all local devices."""
  return jax.tree_util.tree_map(
      lambda v: jax.device_put_sharded(len(devices) * [v], devices), value
  )


def shard_local_devices(value, devices):
  return jax.device_put_sharded(list(value), devices)


def _shard(seq_len: int, host_index: int, num_hosts: int) -> tuple[int, int]:
  """Shard the data based on host index and number of hosts.

  Args:
    seq_len: Input sequence length
    host_index: index of the host (jax.process_index())
    num_hosts: number of hosts (jax.process_count())

  Returns:
    start and end index to shard the data
  """
  assert seq_len % num_hosts == 0
  split = seq_len // num_hosts
  start = split * host_index
  end = start + split
  return start, end


def prepare_obs_act(
    observations: np.ndarray,
    actions: np.ndarray,
    num_devices: int,
    devices: list[jax.Device],
    training: bool = True,
):
  """Validate and prepare observations and actions."""
  local_device_count = len(devices)
  assert (
      observations.shape[0] == actions.shape[0]
  ), "observations and actions must be the same length"

  # ensure no. of time steps is a multiple of num_devices
  if observations.shape[0] % num_devices != 0:
    # Ignoring the last few timesteps to ensure no. of time-steps is a multiple
    # of num_devices
    observations = observations[
        : observations.shape[0] - observations.shape[0] % num_devices
    ]

  # ensure no. of time steps is a multiple of num_devices
  if actions.shape[0] % num_devices != 0:
    actions = actions[: actions.shape[0] - actions.shape[0] % num_devices]

  if not training:
    return observations, actions

  per_device_seq_len = observations.shape[0] // num_devices
  start, end = _shard(
      seq_len=observations.shape[0],
      host_index=jax.process_index(),
      num_hosts=jax.process_count(),
  )

  observations = observations[start:end]
  actions = actions[start:end]

  observations = observations.reshape(local_device_count, per_device_seq_len)

  actions = actions.reshape(local_device_count, per_device_seq_len)

  observations = shard_local_devices(observations, devices)
  actions = shard_local_devices(actions, devices)

  return observations, actions


def get_masked_multiplier(
    n_clones: list[int] | np.ndarray,
) -> np.ndarray:
  """Masked array to remove extra padding which is needed to pmap."""

  masked_multiplier = np.zeros((len(n_clones), max(n_clones)), dtype=int)
  for j in range(len(n_clones)):
    masked_multiplier[j, : n_clones[j]] = 1

  return masked_multiplier
