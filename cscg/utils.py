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

"""Utility functions for CSCG JAX."""

from typing import Mapping

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm

from cscg import cscg


def load_arrays(load_path: str) -> dict[str, np.ndarray]:
  """Load episode data from the given path."""
  npz_data = np.load(load_path)
  data = {key: value for key, value in npz_data.items()}
  return data


class OnlineClustering:
  """An online clustering algorithm based on Euclidean distancethreshold."""

  def __init__(
      self,
      threshold: int = 500,
      max_centers: int = 2_000,
  ):
    self._threshold = threshold
    self._max_centers = max_centers

  def get_hard_observations(self, images):
    """Given images of shape (N, H, W, 3); return embeddings (N, 1)."""

    cluster_indices, cluster_centers = self.cluster(images)
    aux = {}
    aux['cluster_centers'] = cluster_centers
    aux['cluster_indices'] = cluster_indices
    return cluster_indices.astype(int), aux

  def cluster(self, images):
    """Perform online clustering."""

    @jax.jit
    def find_cluster_idx(x):
      dims = tuple(np.arange(1, len(x.shape)))

      def add_vector(carry, i):
        one_member, used = carry
        d = jnp.sqrt(((x[one_member] - x[i]) ** 2).sum(dims))
        idx = d.argmin()
        add = (d[idx] > self._threshold) * (used < self._max_centers)
        cluster_idx = used * add + idx * (1 - add)
        one_member = one_member.at[used].set(i * add)
        used += add
        return (one_member, used), cluster_idx

      one_member, used = jnp.zeros(self._max_centers, int), 1
      _, cluster_idx = jax.lax.scan(
          add_vector, (one_member, used), jnp.arange(1, len(x))
      )
      cluster_idx = jnp.hstack((0, cluster_idx))
      return cluster_idx

    cluster_idx = np.array(find_cluster_idx(images))
    centers = np.zeros((cluster_idx.max() + 1,) + images.shape[1:])
    for i in range(len(centers)):
      centers[i] = images[cluster_idx == i].mean(0)

    centers = centers.astype(np.uint8)
    return cluster_idx, centers


def plot_graph(
    counts_matrix: np.ndarray | jnp.ndarray,
    states: np.ndarray | jnp.ndarray,
    n_clones: np.ndarray,
    x_max: int | np.int64,
    ax: plt.Axes,
    pos: Mapping[int, tuple[float, float]] | None = None,
    threshold: float = 0.0,
    node_size: int = 300,
    use_colors: bool = True,
    cmap: str = 'Spectral',
    node_labels: np.ndarray | None = None,
):
  """Function to plot the transition graph of a CSCG.

  Args:
    counts_matrix: The counts matrix of the CSCG.
    states: An array of decoded states.
    n_clones: Clone allocation for each observation.
    x_max: The maximum observation index.
    ax: The matplotlib axis to plot on.
    pos: A dictionary of node indices and positions. If None, positions are
      calculated using the kamada-kawai layout.
    threshold: A float value used to threshold edges below this value.
    node_size: The size of the nodes.
    use_colors: A boolean indicating whether to use colors for the nodes.
    cmap: The colormap to use for node coloring.
    node_labels: An array of node labels. Can be a NumPy or JAX NumPy array. If
      None, node indices are used as labels.

  Returns:
    The graph object.
  """
  cmap = matplotlib.cm.get_cmap(cmap)

  # Generate graph
  v = np.unique(states)
  amat = counts_matrix[:, v][:, :, v]

  amat = amat.sum(0)
  amat /= amat.sum(1, keepdims=True)

  if node_labels is None:
    node_labels = np.arange(x_max + 1).repeat(n_clones)[v]
  if use_colors:
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
  else:
    colors = 'gray'

  g = nx.from_numpy_array(
      amat > threshold, parallel_edges=False, create_using=nx.DiGraph
  )
  if pos is None:
    pos = nx.kamada_kawai_layout(g)

  # Plot graph
  nx.draw_networkx(
      g,
      pos=pos,
      ax=ax,
      node_color=colors,
      node_size=node_size,
      with_labels=False,
  )
  return g


def extract_egocentric_observations(
    room_layout: np.ndarray,
    fov_h: int,
    fov_w: int,
    observation_dict: Mapping[tuple[tuple[int, ...], ...], int],
):
  """Extract all possible egocentric observations from a 2D environment.

  Args:
    room_layout: A 2D array representing the room layout.
    fov_h: The height of the field of view.
    fov_w: The width of the field of view.
    observation_dict: A dictionary mapping egocentric observation patches to
      integers.

  Returns:
    observation_map: A 3D array representing the egocentric observation map.
    observation_dict: The updated dictionary mapping egocentric observation
      patches to integers.
  """

  n_headings = 4  # 0: north, 1: east, 2: south, 3: west

  room_h, room_w = room_layout.shape

  observation_map = (
      np.zeros([n_headings, room_h, room_w], int) - 1
  )  # -1 marks regions that are inaccessible

  if not observation_dict:
    observation_count = 0
  else:
    observation_count = max(list(observation_dict.values())) + 1

  for h in range(n_headings):

    for c in range(room_w):
      for r in range(room_h):
        if room_layout[r, c] > -1:
          if h == 0:  # north
            patch = room_layout[
                r - fov_h + 2 : r + 2, c - fov_w // 2 : c + fov_w // 2 + 1
            ]
          elif h == 1:  # east
            patch = np.rot90(
                room_layout[
                    r - fov_w // 2 : r + fov_w // 2 + 1, c - 1 : c - 1 + fov_h
                ],
                k=-1,
                axes=(1, 0),
            )
          elif h == 2:  # south
            patch = np.fliplr(
                np.flipud(
                    room_layout[
                        r - 1 : r + fov_h - 1,
                        c - fov_w // 2 : c + fov_w // 2 + 1,
                    ]
                )
            )
          else:  # west
            patch = np.rot90(
                room_layout[
                    r - fov_w // 2 : r + fov_w // 2 + 1, c - fov_h + 2 : c + 2
                ],
                k=-1,
                axes=(0, 1),
            )

          # Make all values in each column of the patch
          # beyond barriers (-1) invisible (set to -1).
          patch_copy = patch.copy()

          for i in range(fov_w):
            for j in range(fov_h - 2, 0, -1):
              if patch_copy[j, i] == -1:
                patch_copy[0:j, i] = -1

          patch_tuple = tuple([tuple(patch_copy[i]) for i in range(fov_h)])

          if patch_tuple not in observation_dict:
            observation_dict[patch_tuple] = observation_count
            observation_count += 1

          observation_map[h, r, c] = observation_dict[patch_tuple]

  return observation_map, observation_dict


def generate_egocentric_random_walk(
    room_observation_map: np.ndarray,
    r_init: int,
    c_init: int,
    h_init: int,
    length: int,
    directionality: str | None = None,
    avoided_locations: set[tuple[int, int, int]] | None = None,
    seed: int = 0,
):
  """Generate an egocentric random walk in the given room.

  Args:
    room_observation_map: A 3D array representing the egocentric observation
      map.
    r_init: The initial row.
    c_init: The initial column.
    h_init: The initial heading.
    length: The length of the random walk.
    directionality: The directionality of the random walk.
    avoided_locations: A set of locations in the layout to avoid.
    seed: The seed for generating the random walk.

  Returns:
    actions: An array of actions taken by the agent.
    observations: An array of observations at each timestep.
    positions: An array of positions at each timestep.
    headings: An array of headings at each timestep.
  """
  rng = np.random.default_rng(seed)
  room_h, room_w = room_observation_map.shape[1:]

  if avoided_locations is None:
    avoided_locations = set([])

  actions = np.zeros(length, int)
  observations = np.zeros(length, int)  # observations
  positions = np.zeros((length, 2), int)  # positions
  headings = np.zeros(length, int)  # headings

  r, c = r_init, c_init
  heading = h_init  # 0: north, 1: east, 2: south, 3: west
  observations[0] = room_observation_map[heading, r, c]
  positions[0] = r, c
  headings[0] = heading
  room_layout = (room_observation_map[0] >= 0) * 1

  count = 0
  itercount = 0

  while count < length - 1:

    itercount += 1
    if itercount > 5 * length:
      break

    prev_r = r
    prev_c = c

    a = rng.integers(0, 3)  # 0: go straight, 1: turn right, 2: turn left

    if a == 0:  # go straight

      # check if a particular location needs to be avoided
      if (heading, prev_r, prev_c) in avoided_locations:
        continue

      # Check for actions taking out of the matrix boundary.
      if heading == 0 and 0 < r:
        r -= 1
      elif heading == 1 and c < room_w - 1:
        c += 1
      elif heading == 2 and r < room_h - 1:
        r += 1
      elif heading == 3 and 0 < c:
        c -= 1

      # restrict movement to given directionality
      if (
          (directionality == 'north' and r > prev_r)
          or (directionality == 'south' and r < prev_r)
          or (directionality == 'east' and c < prev_c)
          or (directionality == 'west' and c > prev_c)
      ):
        r = prev_r
        c = prev_c
        continue

      # check whether action is taking to inaccessible states
      temp_x = room_layout[r, c]
      if temp_x == 0:
        r = prev_r
        c = prev_c
        continue

    elif a == 1:  # turn right
      heading = np.mod(heading + 1, 4)
      r, c = prev_r, prev_c
    else:  # turn left
      heading = np.mod(heading - 1, 4)
      r, c = prev_r, prev_c

    actions[count] = a
    observations[count + 1] = room_observation_map[heading, r, c]
    positions[count + 1] = r, c
    headings[count + 1] = heading

    count += 1

  return (
      actions[0 : count + 1],
      observations[0 : count + 1],
      positions[0 : count + 1],
      headings[0 : count + 1],
  )


def generate_rectangular_layout(
    room_h: int,
    room_w: int,
    padlen: int = 1,
    distinct_edges: bool = True,
    distinct_corners: bool = True,
):
  """Generate a rectangular layout of size room_h x room_w with padding."""

  room_layout = (
      np.zeros(shape=(room_h + 2 * padlen, room_w + 2 * padlen), dtype=np.int_)
      - 1
  )
  room_layout[padlen : padlen + room_h, padlen : padlen + room_w] = 1
  room_layout[
      padlen + 1 : padlen + room_h - 1, padlen + 1 : padlen + room_w - 1
  ] = 0

  a = 1
  if distinct_edges:
    room_layout[padlen, padlen : padlen + room_w] = a
    room_layout[padlen + room_h - 1, padlen : padlen + room_w] = a + 1
    room_layout[padlen : padlen + room_h, padlen] = a + 2
    room_layout[padlen : padlen + room_h, padlen + room_w - 1] = a + 3
    a += 3

  if distinct_corners:
    room_layout[padlen, padlen] = a + 1
    room_layout[padlen + room_h - 1, padlen] = a + 2
    room_layout[padlen, padlen + room_w - 1] = a + 3
    room_layout[padlen + room_h - 1, padlen + room_w - 1] = a + 4

  return room_layout


def generate_circular_layout(
    arena_size: int, radius: float, wall_thickness: float, cue_rows: int
):
  """Generate a circular layout."""
  arena_size = int(arena_size)
  room_layout = np.zeros([arena_size, arena_size])
  for r in range(arena_size):
    for c in range(arena_size):
      distance_squared = (r - arena_size // 2) ** 2 + (c - arena_size // 2) ** 2
      if distance_squared >= radius**2:
        room_layout[r, c] = -1
      elif distance_squared >= (radius - wall_thickness) ** 2:
        room_layout[r, c] = 1

  # find starting row for the circular layout
  r_start = 0
  for r in range(arena_size):
    if np.any(np.array(room_layout[r]) > -1):
      r_start = r
      break

  # Set the cue
  for ri in range(cue_rows):
    r = ri + r_start
    for c in range(arena_size):
      if room_layout[r, c] == 1:
        room_layout[r, c] = 2

  # make the room integer
  room_layout = room_layout.astype(int)

  return room_layout


@jax.jit
def forward(
    transition_matrices: jnp.ndarray,
    emission_matrix: jnp.ndarray,
    pi: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
):
  """Standalone function for computing forward messages.

  Used for computing receptive fields.

  Args:
    transition_matrices: The transition matrices of the CSCG.
    emission_matrix: The emission matrix of the CSCG.
    pi: The initial probability distribution.
    observations: Sequence of observations.
    actions: Sequence of actions.

  Returns:
    log2_lik: The log2 likelihood of the sequence.
    messages: The forward messages.
  """

  transition_matrices = transition_matrices.transpose(0, 2, 1)
  sequence_len = observations.shape[0]
  initial_message = pi * jnp.dot(emission_matrix, observations[0])
  p_obs_0 = initial_message.sum()
  initial_message /= p_obs_0

  def one_step(message, n):
    new_message = jnp.dot(
        transition_matrices[actions[n - 1], :, :], message
    ) * jnp.dot(emission_matrix, observations[n])
    p_obs = new_message.sum() + 1e-32
    new_message /= p_obs
    return new_message, (new_message, p_obs)

  _, (messages, p_obs) = jax.lax.scan(
      one_step, initial_message, jnp.arange(1, sequence_len)
  )
  messages = jnp.concatenate((initial_message[None, :], messages))
  p_obs = jnp.hstack((p_obs_0, p_obs))
  log2_lik = jnp.log2(p_obs)

  return log2_lik, messages


def compute_receptive_fields(
    state_idx: np.ndarray,
    model: cscg.CSCG,
    emission_matrix: np.ndarray,
    room_observation_map: np.ndarray,
    trial_len: int = 100,
    num_trials: int = 10,
    gaussian_half_window_size: int = 7,
    gaussian_scaling: int = 16,
    directionality: bool = None,
    avoided_locations: set[tuple[int, int, int]] | None = None,
    possible_starting_positions: list[tuple[int, int]] | None = None,
    x_soft: np.ndarray | None = None,
    seed: int = 0,
):
  """Function to compute receptive fields of clones in a CSCG."""
  rng = np.random.default_rng(seed)

  transition_matrix = jnp.array(model.transition_matrix)
  pi = jnp.array(
      jnp.ones(transition_matrix.shape[1]) / transition_matrix.shape[1]
  )

  dtype = emission_matrix.dtype
  room_h, room_w = room_observation_map.shape[1:]

  # Initialize receptive fields array.
  rfs = np.zeros((len(state_idx), room_h, room_w), dtype=dtype)
  # Initialize location visitation counts array.
  counts = np.zeros((room_h, room_w), dtype=dtype)

  if possible_starting_positions is None:
    possible_starting_positions = []
    for r in range(room_h):
      for c in range(room_w):
        if room_observation_map[0, r, c] >= 0:
          possible_starting_positions.append((r, c))

  for _ in tqdm.tqdm(range(num_trials)):
    # Sample a starting position and heading.
    sr, sc = possible_starting_positions[
        rng.integers(0, len(possible_starting_positions))
    ]
    sh = rng.integers(0, 4)

    # Generate a random walk.
    actions, _, positions, headings = generate_egocentric_random_walk(
        room_observation_map=room_observation_map,
        r_init=sr,
        c_init=sc,
        h_init=sh,
        length=trial_len,
        directionality=directionality,
        avoided_locations=avoided_locations,
        seed=rng.integers(0, 100_000),
    )

    # Initialize soft observations for all timesteps.
    x_vec = np.zeros((trial_len, emission_matrix.shape[1]), dtype=dtype)

    for t in range(trial_len):
      h_t = headings[t]
      r_t, c_t = positions[t]
      if x_soft is not None:
        x_t = np.zeros(emission_matrix.shape[1])
      else:
        x_t = None

      for dx in range(
          -gaussian_half_window_size, gaussian_half_window_size + 1
      ):
        for dy in range(
            -gaussian_half_window_size, gaussian_half_window_size + 1
        ):
          r_new, c_new = r_t + dx, c_t + dy
          if 0 <= r_new < room_h and 0 <= c_new < room_w:
            if room_observation_map[0, r_new, c_new] > -1:
              o_new = room_observation_map[h_t, r_new, c_new]
              if x_soft is None:
                x_vec[t, o_new] = np.exp(-(dx**2 + dy**2) / gaussian_scaling)
              else:
                x_t += (
                    np.exp(-(dx**2 + dy**2) / gaussian_scaling) * x_soft[o_new]
                )

      if x_soft is not None:
        x_vec[t, :] = x_t

      x_vec[t] /= x_vec[t].sum()

    # Compute forward messages
    _, messages = forward(
        transition_matrix,
        emission_matrix,
        pi,
        jnp.array(x_vec),
        jnp.array(actions),
    )

    # Aggregate activations and state visitation counts for this trial.
    activations = messages[:, state_idx]
    for j in range(positions.shape[0]):
      counts[positions[j][0], positions[j][1]] += 1
      rfs[:, positions[j][0], positions[j][1]] += activations[j]

  rfs = rfs / counts[None, :, :]
  return rfs


def nan_outside(z: np.ndarray):
  z1 = z.astype(float)
  z1[z1 == -1] = np.nan
  return z1


def findmax(z: np.ndarray):
  z = z[np.isfinite(z)]
  return z.max()
