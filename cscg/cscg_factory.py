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

"""Factory for CSCG implementations."""

from typing import Any

from cscg import cscg
from cscg import cscg_he
from cscg import cscg_se


def build_cscg(
    implementation: str,
    seed: int,
    kwargs: dict[str, Any],
) -> cscg.CSCG:
  """Build the CSCG.

  Args:
    implementation: Type of implementation. One of he (hard evidence) or se
      (soft evidence).
    seed: The seed for the random number generator.
    kwargs: A dict for additional args e.g. `kwargs = dict(n_clones=[2, 2],
      pseudocount=2e-4, n_actions=4, batched=True, use_bfloat16=False,)`

  Returns:
    cscg: The CSCG implementation

  Raises:
    ValueError: For invalid implementation
  """

  if implementation == 'he':
    return cscg_he.CSCG(seed=seed, **kwargs)
  elif implementation == 'se':
    return cscg_se.CSCG(seed=seed, **kwargs)
  else:
    raise ValueError(f'Invalid implementation type: {implementation}')
