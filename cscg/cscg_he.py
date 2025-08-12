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

"""An implementation of CSCG with hard evidence in JAX."""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from cscg import cscg
from cscg import cscg_he_utils as cscg_utils


class CSCG(cscg.CSCG):
  """A class implementing CSCG with hard evidence using JAX.

  Attributes:
    num_states: The number of latentstates.
    num_emissions: The number of unique emissions.
    emission_matrix: The emission matrix.
    counts_matrix: The counts matrix.
    transition_matrix: The transition matrix.
  """

  def __init__(
      self,
      n_clones: Sequence[int],
      pseudocount: float,
      n_actions: int = 4,
      seed: int = 0,
      batched: bool = True,
      use_bfloat16: bool = False,
  ):
    """Init.

    Args:
      n_clones: A list containing number of clones for each emission.
      pseudocount: Smoothing factor. Should be non-negative.
      n_actions: Number of actions.
      seed: A seed for the random state.
      batched: Whether to use batched message computation.
      use_bfloat16: Whether to use bfloat16 for computation.
    """
    self._key = jax.random.PRNGKey(seed)
    assert pseudocount >= 0.0, "The pseudocount should be non-negative"
    self._pseudocount = pseudocount
    self._n_clones = np.array(n_clones, dtype=int)
    self._num_actions = n_actions
    self._num_states = self._n_clones.sum()
    self._num_emissions = len(self._n_clones)
    self._n_clones_matrix = (
        None  # used only if keep clone structure is true for E learning
    )

    if batched:
      self._batched = True
      self._num_devices = jax.device_count()
      self._local_device_count = jax.local_device_count()
      self._local_devices = jax.local_devices()
      self._devices = jax.devices()
    else:
      self._batched = False
      self._num_devices = 1
      self._local_device_count = 1
      self._local_devices = [jax.local_devices()[0]]
      self._devices = [jax.local_devices()[0]]

    self._use_bfloat16 = use_bfloat16
    if self._use_bfloat16:
      self._dtype = jnp.bfloat16
    else:
      self._dtype = jnp.float32

    self._max_clones = max(self._n_clones)
    # Brodcast necessary variables to local devices

    self._obs_to_state_start_idx = cscg_utils.bcast_local_devices(
        np.hstack(([0], self._n_clones.cumsum())),
        devices=self._local_devices,
    )
    self._masked_multiplier = cscg_utils.bcast_local_devices(
        cscg_utils.get_masked_multiplier(self._n_clones),
        devices=self._local_devices,
    )

    self._counts_matrix = cscg_utils.bcast_local_devices(
        jax.random.uniform(
            self._key,
            (self._num_actions, self._num_states, self._num_states),
            dtype=self._dtype,
        ),
        devices=self._local_devices,
    )
    self._emission_matrix = cscg_utils.bcast_local_devices(
        cscg_utils.get_default_emission_matrix(
            n_clones=self._n_clones, dtype=self._dtype
        ),
        devices=self._local_devices,
    )
    _, self._key = jax.random.split(self._key)
    self._pi_states = cscg_utils.bcast_local_devices(
        jnp.ones(self._num_states, dtype=self._dtype) / self._num_states,
        devices=self._local_devices,
    )

    self._pi_actions = cscg_utils.bcast_local_devices(
        jnp.ones(self._num_actions, dtype=self._dtype) / self._num_actions,
        devices=self._local_devices,
    )

    self._update_transition_matrix = jax.pmap(
        self.__update_transition_matrix,
        in_axes=(0,),
        out_axes=(0),
        static_broadcasted_argnums=(1,),
    )

    self._update_emission_matrix = jax.pmap(
        self.__update_emission_matrix,
        in_axes=(0,),
        out_axes=(0),
        static_broadcasted_argnums=(1, 2),
    )

    self._forward_mp = jax.pmap(
        self.__forward_mp,
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0, 0),
        static_broadcasted_argnums=(6,),
    )

    self._backtrace = jax.pmap(
        self.__backtrace,
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(6,),
    )

    self._update_transition_counts_mp = jax.pmap(
        self.__update_transition_counts_mp,
        axis_name="devices",
        in_axes=(0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )

    self._forward = jax.pmap(
        self.__forward,
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0, 0),
        static_broadcasted_argnums=(6,),
    )

    self._backward = jax.pmap(
        self.__backward,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(5,),
    )

    self._update_transition_counts = jax.pmap(
        self.__update_transition_counts,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(6,),
    )

    self._update_transition_counts_given_emission = jax.pmap(
        self.__update_transition_counts_given_emission,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )

    self._forward_emission = jax.pmap(
        self.__forward_emission,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )

    self._backward_emission = jax.pmap(
        self.__backward_emission,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )

    self._update_emission_counts = jax.pmap(
        self.__update_emission_counts,
        axis_name="devices",
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
        static_broadcasted_argnums=(4,),
        devices=self._devices,
    )

    self._update_emission_counts_mp = jax.pmap(
        self.__update_emission_counts_mp,
        axis_name="devices",
        in_axes=(0, 0, 0),
        out_axes=(0),
        devices=self._devices,
    )

    self._forward_emission_mp = jax.pmap(
        self.__forward_emission_mp,
        in_axes=(0, 0, 0, 0, 0),
        out_axes=(0, 0),
    )

    self._backtrace_emission = jax.pmap(
        self.__backtrace_emission,
        in_axes=(0, 0, 0, 0),
        out_axes=(0),
    )

    self._transition_matrix = self._update_transition_matrix(
        self._counts_matrix, self._pseudocount
    )

  @property
  def implementation(self) -> str:
    return "he"

  @property
  def batched(self) -> int:
    return self._batched

  @property
  def num_states(self) -> int:
    return self._num_states

  @property
  def num_emissions(self) -> int:
    return self._num_emissions

  @property
  def emission_matrix_default(self) -> np.ndarray:
    return cscg_utils.get_default_emission_matrix(
        n_clones=self._n_clones, dtype=self._dtype
    )

  @property
  def counts_matrix(self) -> np.ndarray:
    return jax.device_get(self._counts_matrix[0])

  @property
  def transition_matrix(self) -> np.ndarray:
    return jax.device_get(self._transition_matrix[0])

  @property
  def pseudocount(self) -> float:
    return self._pseudocount

  @property
  def n_clones(self) -> np.ndarray:
    return self._n_clones

  @property
  def n_clones_matrix(self) -> np.ndarray:
    if self._n_clones_matrix is None:
      raise ValueError
    return self._n_clones_matrix

  def set_counts_matrix(self, counts_matrix: np.ndarray):
    """Set the counts matrix."""

    assert counts_matrix.shape == (
        self._num_actions,
        self._num_states,
        self._num_states,
    ), (
        "Counts matrix dimensions are invalid, expected shape"
        f" {self._num_actions, self._num_states, self._num_states}"
    )
    self._counts_matrix = cscg_utils.bcast_local_devices(
        counts_matrix, devices=self._local_devices
    )
    self._transition_matrix = self._update_transition_matrix(
        self._counts_matrix, self._pseudocount
    )

  def set_pseudocount(self, pseudocount: float) -> None:
    """Set the pseudocount."""
    assert pseudocount >= 0.0, "The pseudocount should be non-negative"
    self._pseudocount = pseudocount
    self._transition_matrix = self._update_transition_matrix(
        self._counts_matrix, self._pseudocount
    )

  def __update_transition_matrix(self, counts_matrix, pseudocount):
    """Update the transition matrix given the accumulated counts matrix."""
    transition_matrix = counts_matrix + pseudocount
    norm = transition_matrix.sum(2, keepdims=True)
    norm = jnp.where(norm == 0, 1, norm)
    transition_matrix /= norm
    return transition_matrix

  def __update_emission_matrix(
      self, emission_counts, pseudocount, pseudocount_extra
  ):
    """Update the emission matrix."""
    emission_matrix = emission_counts + pseudocount + pseudocount_extra
    norm = emission_matrix.sum(1, keepdims=True)
    norm = jnp.where(norm == 0, 1, norm)
    emission_matrix /= norm
    return emission_matrix

  def create_n_clones_matrix(self):
    """Create a block-diagonal matrix version of n_clones."""
    clone_idxs = np.hstack(([0], self._n_clones.cumsum()))
    n_clones_matrix = np.zeros((self._num_states, self._num_states), dtype=int)

    for idx in range(1, len(clone_idxs)):
      st, en = clone_idxs[idx - 1], clone_idxs[idx]
      n_clones_matrix[st:en, st:en] = np.ones(
          (self._n_clones[idx - 1], self._n_clones[idx - 1])
      )
    return n_clones_matrix

  def bridge(
      self,
      state1: int,
      state2: int,
      max_steps: int = 100,
      min_prob: float = 0.0,
  ) -> tuple[np.ndarray | jax.Array, np.ndarray | jax.Array]:
    """Find the path between state1 and state2."""

    pi_states = np.zeros(self._num_states, dtype=self._dtype)
    pi_states[state1] = 1

    _, mess_fwd = self.__forward_mp_all(
        self._transition_matrix[0],
        pi_states,
        self._pi_actions[0],
        state2,
        max_steps,
        min_prob,
    )

    s_a = self.__backtrace_all(
        self._transition_matrix[0], mess_fwd, self._pi_actions[0], state2
    )

    return s_a

  def sample(self, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample observations and actions from the CHMM."""

    random_state = np.random.RandomState()
    assert length > 0
    state_loc = np.hstack(([0], self._n_clones)).cumsum(0)
    sample_x = np.zeros(length, dtype=int)

    p_a = jax.device_get(self._pi_actions[0])
    sample_a = random_state.choice(len(p_a), size=length, p=p_a)
    # Sample
    transition_matrix = jax.device_get(self._transition_matrix[0])
    p_h = jax.device_get(self._pi_states[0])
    for t in range(length):
      h = random_state.choice(len(p_h), p=p_h)
      sample_x[t] = np.digitize(h, state_loc) - 1
      p_h = transition_matrix[sample_a[t], h]
    return np.array(sample_x), np.array(sample_a)

  def bps(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ):
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""

    if emission_matrix is not None:
      return self._bps_emission(observations, actions, emission_matrix)

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        training=False,
    )

    log2_lik, _ = self.__forward(
        self._transition_matrix[0],
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
        self._obs_to_state_start_idx[0],
        self._masked_multiplier[0],
        self._max_clones,
    )
    return -log2_lik.flatten()

  def _bps_emission(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: np.ndarray,
  ):
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        training=False,
    )

    emission_matrix = jnp.array(emission_matrix, dtype=self._dtype)

    log2_lik, _ = self.__forward_emission(
        self._transition_matrix[0],
        emission_matrix,
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
    )
    return -log2_lik.flatten()

  def bps_viterbi(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ):
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""

    if emission_matrix is not None:
      return self._bps_viterbi_emission(observations, actions, emission_matrix)

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        training=False,
    )

    log2_lik, _ = self.__forward_mp(
        self._transition_matrix[0],
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
        self._obs_to_state_start_idx[0],
        self._masked_multiplier[0],
        self._max_clones,
    )
    return -log2_lik.flatten()

  def _bps_viterbi_emission(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: np.ndarray,
  ):
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        training=False,
    )

    emission_matrix = jnp.array(emission_matrix, dtype=self._dtype)

    log2_lik, _ = self.__forward_emission_mp(
        self._transition_matrix[0],
        emission_matrix,
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
    )
    return -log2_lik.flatten()

  def learn_em_transition(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
      n_iter: int = 100,
      term_early: bool = True,
  ):
    """Run EM training, keeping E deterministic and fixed, learning T."""

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
    )

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -np.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward(
          self._transition_matrix,
          self._pi_states,
          observations,
          actions,
          self._obs_to_state_start_idx,
          self._masked_multiplier,
          self._max_clones,
      )

      # backward messages
      mess_bwd = self._backward(
          self._transition_matrix,
          observations,
          actions,
          self._obs_to_state_start_idx,
          self._masked_multiplier,
          self._max_clones,
      )
      mess_bwd = jnp.flip(mess_bwd, axis=1)

      # # compute backward messages and update counts matrix
      self._counts_matrix = self._update_transition_counts(
          self._transition_matrix,
          mess_fwd,
          mess_bwd,
          observations,
          actions,
          self._obs_to_state_start_idx,
          self._max_clones,
      )

      # M step ----
      self._transition_matrix = self._update_transition_matrix(
          self._counts_matrix, self._pseudocount
      )
      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        if term_early:
          break
      log2_lik_old = log2_lik.mean()
    return convergence

  def learn_em_emission(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      n_iter: int = 100,
      keep_clone_structure: bool = False,
      emission_matrix_init: Optional[np.ndarray] = None,
      random_init: bool = False,
      noise_seed: int = 0,
      pseudocount_extra: float = 1e-20,
  ):
    """Run EM training, keeping E deterministic and fixed, learning T."""
    if keep_clone_structure:
      self._n_clones_matrix = self.create_n_clones_matrix()

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
    )

    if emission_matrix_init is None:
      if random_init:
        emission_matrix = jax.random.uniform(
            jax.random.PRNGKey(noise_seed),
            shape=(self._num_states, self._num_emissions),
            dtype=self._dtype,
        )
      else:
        emission_matrix = jnp.ones(
            (self._num_states, self._num_emissions), dtype=self._dtype
        )
      emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)
    else:
      emission_matrix = jnp.array(emission_matrix_init, dtype=self._dtype)

    emission_matrix = cscg_utils.bcast_local_devices(
        emission_matrix, devices=self._local_devices
    )

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -np.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward_emission(
          self._transition_matrix,
          emission_matrix,
          self._pi_states,
          observations,
          actions,
      )

      mess_bwd = self._backward_emission(
          self._transition_matrix, emission_matrix, observations, actions
      )
      mess_bwd = jnp.flip(mess_bwd, axis=1)

      # Update emission counts
      emission_counts = self._update_emission_counts(
          emission_matrix,
          mess_fwd,
          mess_bwd,
          observations,
          keep_clone_structure,
      )
      # Get the first index as the matrix is copied across all devices

      # M step ----
      emission_matrix = self._update_emission_matrix(
          emission_counts, self._pseudocount, pseudocount_extra
      )

      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        break
      log2_lik_old = log2_lik.mean()
    return convergence, jax.device_get(emission_matrix[0])

  def learn_viterbi_emission(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      n_iter: int = 100,
      keep_clone_structure: bool = False,
      emission_matrix_init: Optional[np.ndarray] = None,
      random_init: bool = False,
      noise_seed: int = 0,
      pseudocount_extra: float = 1e-20,
  ):
    """Run EM training, keeping E deterministic and fixed, learning T."""
    if keep_clone_structure:
      self._n_clones_matrix = self.create_n_clones_matrix()

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
    )

    if emission_matrix_init is None:
      if random_init:
        emission_matrix = jax.random.uniform(
            jax.random.PRNGKey(noise_seed),
            shape=(self._num_states, self._num_emissions),
            dtype=self._dtype,
        )
      else:
        emission_matrix = jnp.ones(
            (self._num_states, self._num_emissions), dtype=self._dtype
        )
      emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)
    else:
      emission_matrix = jnp.array(emission_matrix_init, dtype=self._dtype)

    emission_matrix = cscg_utils.bcast_local_devices(
        emission_matrix, devices=self._local_devices
    )

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -np.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward_emission(
          self._transition_matrix,
          emission_matrix,
          self._pi_states,
          observations,
          actions,
      )

      states = self._backtrace_emission(
          self._transition_matrix, mess_fwd, observations, actions
      )

      states = jnp.flip(states, axis=1)

      # Update emission counts
      emission_counts = self._update_emission_counts_mp(
          emission_matrix, states, observations)

      # M step ----
      emission_matrix = self._update_emission_matrix(
          emission_counts, self._pseudocount, pseudocount_extra
      )

      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        break
      log2_lik_old = log2_lik.mean()
    return convergence, jax.device_get(emission_matrix[0])

  def learn_viterbi_transition(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
      n_iter: int = 100,
  ):
    """Run Viterbi training, keeping E deterministic and fixed, learning T."""

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
    )

    self.set_pseudocount(0.0)

    convergence = []
    pbar = tqdm.trange(n_iter, position=0)
    log2_lik_old = -jnp.inf

    for _ in pbar:
      # E step ----
      # forward messages
      log2_lik, mess_fwd = self._forward_mp(
          self._transition_matrix,
          self._pi_states,
          observations,
          actions,
          self._obs_to_state_start_idx,
          self._masked_multiplier,
          self._max_clones,
      )

      # backtrace
      states = self._backtrace(
          self._transition_matrix,
          mess_fwd,
          observations,
          actions,
          self._obs_to_state_start_idx,
          self._masked_multiplier,
          self._max_clones,
      )
      states = jnp.flip(states, axis=1)

      # update counts matrix
      self._counts_matrix = self._update_transition_counts_mp(
          self._transition_matrix,
          actions,
          states,
      )

      # M step ----
      self._transition_matrix = self._update_transition_matrix(
          self._counts_matrix, self._pseudocount
      )
      # Convergence check
      if self._use_bfloat16:
        log2_lik = log2_lik.astype(np.float32)
      convergence.append(-log2_lik.mean())
      pbar.set_postfix(train_bps=convergence[-1])
      if log2_lik.mean() <= log2_lik_old:
        break
      log2_lik_old = log2_lik.mean()
    return convergence

  def decode(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ):
    """Compute the MAP assignment of latent variables using max-product message passing."""

    if emission_matrix is not None:
      return self._decode_emission(observations, actions, emission_matrix)

    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        training=False,
    )

    # forward messages
    log2_lik, mess_fwd = self.__forward_mp(
        self._transition_matrix[0],
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
        self._obs_to_state_start_idx[0],
        self._masked_multiplier[0],
        self._max_clones,
    )

    states = self.__backtrace(
        self._transition_matrix[0],
        mess_fwd,
        jnp.array(observations),
        jnp.array(actions),
        self._obs_to_state_start_idx[0],
        self._masked_multiplier[0],
        self._max_clones,
    )
    states = jnp.flip(states, axis=0)

    return -log2_lik, states.flatten()

  def _decode_emission(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: np.ndarray,
  ):
    """Compute the MAP assignment of latent variables using max-product message passing."""
    observations, actions = cscg_utils.prepare_obs_act(
        observations=observations,
        actions=actions,
        num_devices=self._num_devices,
        devices=self._local_devices,
        training=False,
    )

    emission_matrix = jnp.array(emission_matrix, dtype=self._dtype)

    # forward messages
    log2_lik, mess_fwd = self.__forward_emission_mp(
        self._transition_matrix[0],
        emission_matrix,
        self._pi_states[0],
        jnp.array(observations),
        jnp.array(actions),
    )

    states = self.__backtrace_emission(
        self._transition_matrix[0],
        mess_fwd,
        jnp.array(observations),
        jnp.array(actions),
    )
    states = jnp.flip(states, axis=0)

    return -log2_lik, states.flatten()

  def __forward(
      self,
      transition_matrices: jnp.ndarray,
      pi: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
      obs_to_start_state_index: jnp.ndarray,
      masked_multiplier: jnp.ndarray,
      max_clones: int,
  ):
    """Compute the forward messages."""

    transition_matrices = transition_matrices.transpose(0, 2, 1)
    sequence_len = observations.shape[0]
    transition_matrices = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )

    initial_message = jax.lax.dynamic_slice(
        pi, (obs_to_start_state_index[observations[0]],), (max_clones,)
    )

    initial_message = jnp.multiply(
        initial_message, masked_multiplier[observations[0]]
    )
    p_obs_0 = initial_message.sum()
    initial_message /= p_obs_0

    def one_step(message, n):
      transition_slice = jax.lax.dynamic_slice(
          transition_matrices[actions[n - 1], :, :],
          (
              obs_to_start_state_index[observations[n]],
              obs_to_start_state_index[observations[n - 1]],
          ),
          (max_clones, max_clones),
      )
      new_message = jnp.matmul(transition_slice, message)
      new_message = jnp.multiply(
          new_message, masked_multiplier[observations[n]]
      )
      p_obs = new_message.sum()
      new_message /= p_obs
      return new_message, (new_message, p_obs)

    _, (messages, p_obs) = jax.lax.scan(
        one_step, initial_message, jnp.arange(1, sequence_len)
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    p_obs = jnp.hstack((p_obs_0, p_obs))
    log2_lik = jnp.log2(p_obs)

    return log2_lik, messages

  def __backward(
      self,
      transition_matrices: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
      obs_to_start_state_index: jnp.ndarray,
      masked_multiplier: jnp.ndarray,
      max_clones: int,
  ):
    """Compute the backward messages."""

    sequence_len = observations.shape[0]
    transition_matrices = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )

    initial_message = jnp.ones(max_clones, dtype=self._dtype)
    initial_message = jnp.multiply(
        initial_message, masked_multiplier[observations[sequence_len - 1]]
    )
    initial_message /= initial_message.sum()

    def one_step(message, n):
      transition_slice = jax.lax.dynamic_slice(
          transition_matrices[actions[n], :, :],
          (
              obs_to_start_state_index[observations[n]],
              obs_to_start_state_index[observations[n + 1]],
          ),
          (max_clones, max_clones),
      )
      new_message = jnp.matmul(transition_slice, message)
      new_message = jnp.multiply(
          new_message, masked_multiplier[observations[n]]
      )
      new_message /= new_message.sum()
      return new_message, new_message

    _, messages = jax.lax.scan(
        one_step, initial_message, jnp.arange(sequence_len - 2, -1, -1)
    )

    messages = jnp.concatenate((initial_message[None, :], messages))

    return messages

  def __update_transition_counts(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      mess_bwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
      obs_to_start_state_index: jnp.ndarray,
      max_clones: int,
  ):
    """Update the counts matrix."""

    sequence_len = observations.shape[0]
    transition_matrices = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )
    counts_matrix_init = jnp.zeros(transition_matrices.shape, dtype=self._dtype)

    def one_step(counts_matrix, n):
      transition_slice = jax.lax.dynamic_slice(
          transition_matrices[actions[n], :, :],
          (
              obs_to_start_state_index[observations[n]],
              obs_to_start_state_index[observations[n + 1]],
          ),
          (max_clones, max_clones),
      )
      count_slice = jax.lax.dynamic_slice(
          counts_matrix[actions[n], :, :],
          (
              obs_to_start_state_index[observations[n]],
              obs_to_start_state_index[observations[n + 1]],
          ),
          (max_clones, max_clones),
      )

      q = transition_slice * (
          mess_fwd[n, :][:, None] * mess_bwd[n + 1, :][None, :]
      )
      q /= q.sum()
      update_slice = count_slice + q
      update_slice = update_slice[None, :, :]

      updated_counts_matrix = jax.lax.dynamic_update_slice(
          counts_matrix,
          update_slice,
          (
              actions[n],
              obs_to_start_state_index[observations[n]],
              obs_to_start_state_index[observations[n + 1]],
          ),
      )

      return updated_counts_matrix, None

    final_counts_matrix, _ = jax.lax.scan(
        one_step,
        counts_matrix_init,
        jnp.arange(sequence_len - 1),
    )

    final_counts_matrix = jax.lax.psum(final_counts_matrix, axis_name="devices")
    final_counts_matrix = final_counts_matrix[:, :-max_clones, :-max_clones]

    return final_counts_matrix

  def __update_transition_counts_given_emission(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """this function combines backward message passing with update counts."""
    sequence_len = observations.shape[0]

    transition_matrices = jnp.pad(transition_matrices, ((0, 0), (0, 1), (0, 1)))
    counts_matrix = jnp.zeros_like(transition_matrices)
    mess_fwd = jnp.pad(mess_fwd, ((0, 0), (0, 1)))
    emission_matrix = jnp.pad(emission_matrix, ((0, 1), (0, 0)))

    initial_message = jnp.ones(emission_matrix.shape[0], dtype=self._dtype)
    initial_message /= initial_message.sum()

    def one_step(inputs, n):
      counts_matrix, message = inputs

      aij = actions[n - 1]

      m_f = mess_fwd[n - 1]
      m_b = message * emission_matrix[:, observations[n]]

      q = m_f.reshape(-1, 1) * transition_matrices[aij] * m_b.reshape(1, -1)
      q /= q.sum()

      updated_counts_matrix = counts_matrix.at[aij].add(q)

      # update message
      new_message = jnp.dot(transition_matrices[aij, :, :], m_b)
      p_obs = new_message.sum()
      new_message /= p_obs

      return (updated_counts_matrix, new_message), None

    (final_counts_matrix, _), _ = jax.lax.scan(
        one_step,
        (counts_matrix, initial_message),
        jnp.arange(sequence_len - 1, 0, -1),
    )

    final_counts_matrix = jax.lax.psum(final_counts_matrix, axis_name="devices")
    final_counts_matrix = final_counts_matrix[:, :-1, :-1]

    return final_counts_matrix

  def __forward_mp(
      self,
      transition_matrices: jnp.ndarray,
      pi: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
      obs_to_start_state_index: jnp.ndarray,
      masked_multiplier: jnp.ndarray,
      max_clones: int,
  ):
    """Compute the forward messages."""
    transition_matrices = transition_matrices.transpose(0, 2, 1)
    sequence_len = observations.shape[0]
    transition_matrices = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )
    initial_message = jax.lax.dynamic_slice(
        pi, (obs_to_start_state_index[observations[0]],), (max_clones,)
    )
    initial_message = jnp.multiply(
        initial_message, masked_multiplier[observations[0]]
    )
    p_obs_0 = initial_message.max()
    initial_message /= p_obs_0

    def one_step(message, n):
      transition_slice = jax.lax.dynamic_slice(
          transition_matrices[actions[n - 1], :, :],
          (
              obs_to_start_state_index[observations[n]],
              obs_to_start_state_index[observations[n - 1]],
          ),
          (max_clones, max_clones),
      )
      new_message = (transition_slice * message).max(axis=1)
      new_message = jnp.multiply(
          new_message, masked_multiplier[observations[n]]
      )
      p_obs = new_message.max()
      new_message /= p_obs
      return new_message, (new_message, p_obs)

    _, (messages, p_obs) = jax.lax.scan(
        one_step, initial_message, jnp.arange(1, sequence_len)
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    p_obs = jnp.hstack((p_obs_0, p_obs))
    log2_lik = jnp.log2(p_obs)

    return log2_lik, messages

  def __backtrace(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
      obs_to_start_state_index: jnp.ndarray,
      masked_multiplier: jnp.ndarray,
      max_clones: int,
  ):
    """Compute the backtrace."""
    sequence_len = observations.shape[0]
    transition_matrices = jnp.pad(
        transition_matrices, ((0, 0), (0, max_clones), (0, max_clones))
    )
    initial_state = (
        mess_fwd[sequence_len - 1].argmax()
        + obs_to_start_state_index[observations[sequence_len - 1]]
    )

    def one_step(state, n):
      transition_slice = jax.lax.dynamic_slice(
          transition_matrices[actions[n], :, state],
          (obs_to_start_state_index[observations[n]],),
          (max_clones,),
      )

      belief = mess_fwd[n] * transition_slice
      belief = jnp.multiply(belief, masked_multiplier[observations[n]])
      new_state = belief.argmax() + obs_to_start_state_index[observations[n]]
      return new_state, new_state

    _, states = jax.lax.scan(
        one_step, initial_state, jnp.arange(sequence_len - 2, -1, -1)
    )
    states = jnp.hstack((initial_state, states))

    return states

  def __update_transition_counts_mp(
      self,
      transition_matrices: jnp.ndarray,
      actions: jnp.ndarray,
      states: jnp.ndarray,
  ):
    """Update the counts matrix."""

    sequence_len = actions.shape[0]
    counts_matrix_init = jnp.zeros(transition_matrices.shape, dtype=self._dtype)

    def one_step(counts_matrix, n):
      aij = actions[n - 1]
      i, j = states[n - 1], states[n]
      count_slice = jax.lax.dynamic_slice(
          counts_matrix[aij, :, :], (i, j), (1, 1)
      )
      update_slice = count_slice + 1.0
      update_slice = update_slice[None, :, :]
      updated_counts_matrix = jax.lax.dynamic_update_slice(
          counts_matrix,
          update_slice,
          (
              aij,
              i,
              j,
          ),
      )
      return updated_counts_matrix, None

    final_counts_matrix, _ = jax.lax.scan(
        one_step,
        counts_matrix_init,
        jnp.arange(1, sequence_len),
    )

    final_counts_matrix = jax.lax.psum(final_counts_matrix, axis_name="devices")

    return final_counts_matrix

  def __forward_emission(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      pi: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the forward messages with give emission matrix."""
    transition_matrices = transition_matrices.transpose(0, 2, 1)
    sequence_len = observations.shape[0]
    initial_message = jnp.multiply(pi, emission_matrix[:, observations[0]])
    p_obs_0 = initial_message.sum()
    initial_message /= p_obs_0

    def one_step(message, n):
      new_message = jnp.matmul(transition_matrices[actions[n - 1]], message)
      new_message = jnp.multiply(
          new_message, emission_matrix[:, observations[n]]
      )
      p_obs = new_message.sum()
      new_message /= p_obs
      return new_message, (new_message, p_obs)

    _, (messages, p_obs) = jax.lax.scan(
        one_step, initial_message, jnp.arange(1, sequence_len)
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    p_obs = jnp.hstack((p_obs_0, p_obs))
    log2_lik = jnp.log2(p_obs)

    return log2_lik, messages

  def __backward_emission(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the backward messages with give emission matrix."""
    sequence_len = observations.shape[0]
    initial_message = jnp.ones(emission_matrix.shape[0], dtype=self._dtype)
    initial_message /= initial_message.sum()

    def one_step(message, n):
      new_message = jnp.multiply(
          message, emission_matrix[:, observations[n + 1]]
      )
      new_message = jnp.matmul(transition_matrices[actions[n]], new_message)
      new_message /= new_message.sum()

      return new_message, new_message

    _, messages = jax.lax.scan(
        one_step, initial_message, jnp.arange(sequence_len - 2, -1, -1)
    )

    messages = jnp.concatenate((initial_message[None, :], messages))

    return messages

  def __update_emission_counts(
      self,
      emission_matrix: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      mess_bwd: jnp.ndarray,
      observations: jnp.ndarray,
      keep_clone_structure: bool = False,
  ):
    """Update emission counts."""
    sequence_len = observations.shape[0]

    gamma = mess_fwd * mess_bwd
    gamma /= gamma.sum(axis=1, keepdims=True)  # sum over latent states

    if keep_clone_structure:
      gamma = jnp.dot(gamma, self.n_clones_matrix)

    emission_counts_init = jnp.zeros(emission_matrix.shape)

    def one_step(counts_matrix, n):
      update_slice = counts_matrix[:, observations[n]] + gamma[n]

      updated_counts_matrix = jax.lax.dynamic_update_slice(
          counts_matrix,
          update_slice[:, None],
          (counts_matrix.shape[0], observations[n]),
      )

      return updated_counts_matrix, None

    emission_counts, _ = jax.lax.scan(
        one_step, emission_counts_init, jnp.arange(sequence_len)
    )

    emission_counts = jax.lax.psum(emission_counts, axis_name="devices")

    return emission_counts

  def __update_emission_counts_mp(
      self,
      emission_matrix: jnp.ndarray,
      states: jnp.ndarray,
      observations: jnp.ndarray,
  ):
    """Update the emission counts matrix."""
    sequence_len = observations.shape[0]
    emission_counts_init = jnp.zeros(emission_matrix.shape)

    def one_step(counts_matrix, n):
      i, j = states[n], observations[n]
      count_slice = jax.lax.dynamic_slice(
          counts_matrix, (i, j), (1, 1)
      )
      update_slice = count_slice + 1.0
      # update_slice = update_slice[None, :, :]
      updated_counts_matrix = jax.lax.dynamic_update_slice(
          counts_matrix,
          update_slice,
          (
              i,
              j,
          ),
      )
      return updated_counts_matrix, None

    final_counts_matrix, _ = jax.lax.scan(
        one_step,
        emission_counts_init,
        jnp.arange(1, sequence_len),
    )

    final_counts_matrix = jax.lax.psum(final_counts_matrix, axis_name="devices")

    return final_counts_matrix

  def __forward_emission_mp(
      self,
      transition_matrices: jnp.ndarray,
      emission_matrix: jnp.ndarray,
      pi: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the forward messages with give emission matrix."""
    transition_matrices = transition_matrices.transpose(0, 2, 1)
    sequence_len = observations.shape[0]
    initial_message = jnp.multiply(pi, emission_matrix[:, observations[0]])
    p_obs_0 = initial_message.sum()
    initial_message /= p_obs_0

    def one_step(message, n):
      new_message = (transition_matrices[actions[n - 1]] * message).max(axis=1)
      new_message = jnp.multiply(
          new_message, emission_matrix[:, observations[n]]
      )
      p_obs = new_message.max()
      new_message /= p_obs
      return new_message, (new_message, p_obs)

    _, (messages, p_obs) = jax.lax.scan(
        one_step, initial_message, jnp.arange(1, sequence_len)
    )
    messages = jnp.concatenate((initial_message[None, :], messages))
    p_obs = jnp.hstack((p_obs_0, p_obs))
    log2_lik = jnp.log2(p_obs)

    return log2_lik, messages

  def __backtrace_emission(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      observations: jnp.ndarray,
      actions: jnp.ndarray,
  ):
    """Compute the backtrace."""
    sequence_len = observations.shape[0]

    initial_state = mess_fwd[sequence_len - 1].argmax()

    def one_step(state, n):
      belief = mess_fwd[n] * transition_matrices[actions[n], :, state]
      new_state = belief.argmax()
      return new_state, new_state

    _, states = jax.lax.scan(
        one_step, initial_state, jnp.arange(sequence_len - 2, -1, -1)
    )
    states = jnp.hstack((initial_state, states))

    return states

  def __forward_mp_all(
      self,
      transition_matrices: jnp.ndarray,
      pi: jnp.ndarray,
      pi_actions: jnp.ndarray,
      target_state: int,
      max_steps: int,
      min_prob: float = 0.0,
  ):
    """Compute the forward messages."""
    transition_matrices = transition_matrices.transpose(0, 2, 1)

    message = pi
    p_obs_0 = message.max()
    message /= p_obs_0

    log2_lik = [np.log2(p_obs_0)]
    mess_fwd = [message]

    transition_matrices_maxa = (
        transition_matrices * pi_actions.reshape(-1, 1, 1)
    ).max(axis=0)

    for _ in range(1, max_steps):
      message = (transition_matrices_maxa * message.reshape(1, -1)).max(axis=1)
      p_obs = message.max()
      assert p_obs > 0
      message /= p_obs
      log2_lik.append(np.log2(p_obs))
      mess_fwd.append(message)
      if message[target_state] > min_prob:
        break
    else:
      assert False, "Unable to find a bridging path"

    return jnp.array(log2_lik), jnp.array(mess_fwd, dtype=self._dtype)

  def __backtrace_all(
      self,
      transition_matrices: jnp.ndarray,
      mess_fwd: jnp.ndarray,
      pi_actions: jnp.ndarray,
      target_state: int,
  ):
    """Compute the backtrace."""

    states = jnp.zeros(mess_fwd.shape[0], dtype=int)
    actions = jnp.zeros(mess_fwd.shape[0], dtype=int)

    # backward pass
    t = mess_fwd.shape[0] - 1
    # last action is irrelevant, use an invalid value
    actions = actions.at[t].set(-1)
    states = states.at[t].set(target_state)

    for t in range(mess_fwd.shape[0] - 2, -1, -1):
      belief = (
          mess_fwd[t].reshape(1, -1)
          * transition_matrices[:, :, states[t + 1]]
          * pi_actions.reshape(-1, 1)
      )
      a_s = jnp.argmax(belief.flatten())
      actions = actions.at[t].set(a_s // self._num_states)
      states = states.at[t].set(a_s % self._num_states)
    return actions, states
