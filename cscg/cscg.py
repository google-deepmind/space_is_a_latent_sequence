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

"""Interface for the CSCG code."""

import abc
from typing import Optional

import jax
import numpy as np


class CSCG(abc.ABC):
  """Interface for CSCG."""

  @property
  @abc.abstractmethod
  def implementation(self) -> str:
    """Returns the implementation type."""

  @property
  @abc.abstractmethod
  def batched(self) -> bool:
    """Returns True if batched training else False."""

  @property
  @abc.abstractmethod
  def num_states(self) -> int:
    """Returns number of states."""

  @property
  @abc.abstractmethod
  def num_emissions(self) -> int:
    """Returns number of unique emissions."""

  @property
  @abc.abstractmethod
  def emission_matrix_default(self) -> np.ndarray:
    """Returns the default emission matrix."""

  @property
  @abc.abstractmethod
  def counts_matrix(self) -> np.ndarray:
    """Returns the counts matrix."""

  @property
  @abc.abstractmethod
  def transition_matrix(self) -> np.ndarray:
    """Returns the transition matrix."""

  @property
  @abc.abstractmethod
  def pseudocount(self) -> float:
    """Returns the pseudocount."""

  @property
  @abc.abstractmethod
  def n_clones(self) -> np.ndarray:
    """Returns n_clones."""

  @abc.abstractmethod
  def set_counts_matrix(self, counts_matrix: np.ndarray) -> None:
    """Set the counts matrix."""

  @abc.abstractmethod
  def set_pseudocount(self, pseudocount: float) -> None:
    """Set the pseudocount."""

  @abc.abstractmethod
  def bridge(
      self, state1: int, state2: int, max_steps: int = 100
  ) -> tuple[np.ndarray | jax.Array, np.ndarray | jax.Array]:
    """Find the path between state1 and state2."""

  @abc.abstractmethod
  def sample(self, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample observations and actions from the CHMM."""

  @abc.abstractmethod
  def bps(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""

  @abc.abstractmethod
  def bps_viterbi(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    """Compute the log likelihood (log base 2) of a sequence of observations and actions."""

  @abc.abstractmethod
  def learn_em_transition(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
      n_iter: int = 100,
      term_early: bool = True,
  ) -> list[float]:
    """Run EM training, keeping E deterministic and fixed, learning T."""

  @abc.abstractmethod
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
  ) -> tuple[list[float], np.ndarray]:
    """Run EM training, keeping T deterministic and fixed, learning E."""

  @abc.abstractmethod
  def learn_viterbi_transition(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
      n_iter: int = 100,
  ) -> list[float]:
    """Run Viterbi training, keeping E deterministic and fixed, learning T."""

  @abc.abstractmethod
  def decode(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      emission_matrix: Optional[np.ndarray] = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the MAP assignment of latent variables using max-product message passing."""
