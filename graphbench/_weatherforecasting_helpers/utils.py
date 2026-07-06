"""Spatial features and dataset utilities for grid-mesh weather forecasting graphs.

Provides coordinate transformations, spatial feature computation, temporal dataset
splitting, and a PyG InMemoryDataset for GraphCast-style weather models.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Sequence, Tuple, Optional
import xarray as xr
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable
from dataclasses import dataclass


def get_weather_variables() -> List[str]:
    return [
        '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
        '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature',
        'geopotential', 'u_component_of_wind', 'v_component_of_wind',
        'vertical_velocity', 'specific_humidity'
    ]

def get_pressure_levels() -> List[int]:
    return [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def lat_lon_deg_to_spherical(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon in degrees to spherical angles (phi, theta)."""
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90 - lat)
    return phi, theta


def spherical_to_cartesian(phi: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map spherical angles to unit-sphere Cartesian coordinates (x, y, z)."""
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return x, y, z


def get_relative_positions(senders_pos: np.ndarray, receivers_pos: np.ndarray) -> np.ndarray:
    """Return receiver_pos - sender_pos for each edge."""
    return receivers_pos - senders_pos


def get_graph_spatial_features(
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool = False,
    add_node_latitude: bool = True,
    add_node_longitude: bool = True,
    add_relative_positions: bool = True,
    edge_normalization_factor: Optional[float] = None,
    sine_cosine_encoding: bool = False,
    encoding_num_frequencies: int = 10,
    encoding_multiplier: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute node and edge spatial features for a homogeneous graph.

    Node features: optional Cartesian position, cos(lat), cos/sin(lon).
    Edge features: L2-normalised relative position vector (and its norm) in 3-D space.
    Optionally applies a multi-frequency sine/cosine encoding to both.
    """
    num_nodes = node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = node_lat.dtype
    node_phi, node_theta = lat_lon_deg_to_spherical(node_lat, node_lon)

    node_features = []
    if add_node_positions:
        node_features.extend(spherical_to_cartesian(node_phi, node_theta))
    if add_node_latitude:
        # cos(theta): 1 at north pole, -1 at south pole
        node_features.append(np.cos(node_theta))
    if add_node_longitude:
        node_features.append(np.cos(node_phi))
        node_features.append(np.sin(node_phi))

    if not node_features:
        node_features = np.zeros([num_nodes, 0], dtype=dtype)
    else:
        node_features = np.stack(node_features, axis=-1)

    edge_features = []
    if add_relative_positions:
        relative_position = get_relative_position_in_receiver_local_coordinates(
            node_phi=node_phi,
            node_theta=node_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=None,
            longitude_local_coordinates=None,
        )
        # L2 distance in 3-D space (not geodesic)
        relative_edge_distances = np.linalg.norm(relative_position, axis=-1, keepdims=True)
        if edge_normalization_factor is None:
            edge_normalization_factor = relative_edge_distances.max()
        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    if sine_cosine_encoding:
        def sine_cosine_transform(x: np.ndarray) -> np.ndarray:
            freqs = encoding_multiplier**np.arange(encoding_num_frequencies)
            phases = freqs * x[..., None]
            x_cat = np.concatenate([np.sin(phases), np.cos(phases)], axis=-1)
            return x_cat.reshape([x.shape[0], -1])

        node_features = sine_cosine_transform(node_features)
        edge_features = sine_cosine_transform(edge_features)

    return node_features, edge_features

def get_bipartite_graph_spatial_features(
    senders_node_lat: np.ndarray,
    senders_node_lon: np.ndarray,
    receivers_node_lat: np.ndarray,
    receivers_node_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool = False,
    add_node_latitude: bool = True,
    add_node_longitude: bool = True,
    add_relative_positions: bool = True,
    edge_normalization_factor: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial features for a bipartite graph (e.g. grid-to-mesh).

    Returns separate feature arrays for sender nodes, receiver nodes, and edges.
    Edge features are L2-normalised relative positions in 3-D Cartesian space.
    """
    num_senders = senders_node_lat.shape[0]
    num_receivers = receivers_node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = senders_node_lat.dtype

    senders_phi, senders_theta = lat_lon_deg_to_spherical(senders_node_lat, senders_node_lon)
    receivers_phi, receivers_theta = lat_lon_deg_to_spherical(receivers_node_lat, receivers_node_lon)

    # --- sender node features ---
    senders_features = []
    if add_node_positions:
        senders_features.extend(spherical_to_cartesian(senders_phi, senders_theta))
    if add_node_latitude:
        senders_features.append(np.cos(senders_theta))
    if add_node_longitude:
        senders_features.append(np.cos(senders_phi))
        senders_features.append(np.sin(senders_phi))
    senders_features = (
        np.zeros([num_senders, 0], dtype=dtype) if not senders_features
        else np.stack(senders_features, axis=-1)
    )

    # --- receiver node features ---
    receivers_features = []
    if add_node_positions:
        receivers_features.extend(spherical_to_cartesian(receivers_phi, receivers_theta))
    if add_node_latitude:
        receivers_features.append(np.cos(receivers_theta))
    if add_node_longitude:
        receivers_features.append(np.cos(receivers_phi))
        receivers_features.append(np.sin(receivers_phi))
    receivers_features = (
        np.zeros([num_receivers, 0], dtype=dtype) if not receivers_features
        else np.stack(receivers_features, axis=-1)
    )

    # --- edge features ---
    edge_features = []
    if add_relative_positions:
        #TODO: fix this
        relative_positions = get_bipartite_relative_position_in_receiver_local_coordinates(
            senders_node_phi=senders_phi,
            senders_node_theta=senders_theta,
            senders=senders,
            receivers_node_phi=receivers_phi,
            receivers_node_theta=receivers_theta,
            receivers=receivers,
            latitute_local_coordinates=None,
            longitude_local_coordinates=None,
        )
        distances = np.linalg.norm(relative_positions, axis=-1, keepdims=True)
        if edge_normalization_factor is None:
            edge_normalization_factor = distances.max()
        edge_features.append(distances / edge_normalization_factor)
        edge_features.append(relative_positions / edge_normalization_factor)

    edge_features = (
        np.zeros([num_edges, 0], dtype=dtype) if not edge_features
        else np.concatenate(edge_features, axis=-1)
    )

    return senders_features, receivers_features, edge_features

def get_bipartite_relative_position_in_receiver_local_coordinates(
        senders_node_phi,
        senders_node_theta,
        senders,
        receivers_node_phi,
        receivers_node_theta,
        receivers,
        latitute_local_coordinates,
        longitude_local_coordinates,
):
    """Return sender_pos - receiver_pos in 3-D Cartesian space for each bipartite edge.

    Local-coordinate rotation is not yet implemented; pass None for both coordinate args.
    """
    s_x, s_y, s_z = spherical_to_cartesian(senders_node_phi, senders_node_theta)
    r_x, r_y, r_z = spherical_to_cartesian(receivers_node_phi, receivers_node_theta)
    senders_node_pos = np.stack([s_x, s_y, s_z], axis=-1)
    receivers_node_pos = np.stack([r_x, r_y, r_z], axis=-1)
    if not (latitute_local_coordinates or longitude_local_coordinates):
        return senders_node_pos[senders.cpu()] - receivers_node_pos[receivers.cpu()]

def get_relative_position_in_receiver_local_coordinates(
        node_phi,
        node_theta,
        senders,
        receivers,
        latitude_local_coordinates,
        longitude_local_coordinates,
):
    """Return sender_pos - receiver_pos in 3-D Cartesian space for each homogeneous edge.

    Local-coordinate rotation is not yet implemented; pass None for both coordinate args.
    """
    x, y, z = spherical_to_cartesian(node_phi, node_theta)
    node_pos = np.stack([x, y, z], axis=-1)
    if not (latitude_local_coordinates or longitude_local_coordinates):
        return node_pos[senders.cpu()] - node_pos[receivers.cpu()]

@dataclass(frozen=True)
class TemporalSplits:
    """Immutable container for train/val/test index arrays."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def as_slices(self) -> Tuple[slice, slice, slice]:
        """Convert index arrays to contiguous slices (assumes sorted, contiguous indices)."""
        def to_slice(indices: np.ndarray) -> slice:
            if indices.size == 0:
                return slice(0, 0)
            return slice(int(indices[0]), int(indices[-1]) + 1)

        return to_slice(self.train_idx), to_slice(self.val_idx), to_slice(self.test_idx)


def _find_first_index_at_or_after(datetimes: np.ndarray, threshold: np.datetime64) -> int:
    # datetimes must be sorted ascending
    return int(np.searchsorted(datetimes, threshold, side="left"))


def compute_temporal_splits(
    datetimes: Sequence[np.datetime64],
) -> TemporalSplits:
    """Split datetimes into train (1979-2015), val (2016-2017), test (2018-2021).

    Falls back to 80/10/10 proportional split when any year-based split is empty.
    """
    return compute_fixed_year_splits(
        datetimes,
        train_years=(1979, 2015),
        val_years=(2016, 2017),
        test_years=(2018, 2021),
    )


def slice_sequence_by_indices(sequence: Sequence, indices: np.ndarray) -> List:
    """Select elements from a sequence by integer index array."""
    return [sequence[int(i)] for i in indices]


def compute_fixed_year_splits(
    datetimes: Sequence[np.datetime64],
    *,
    train_years: tuple[int, int] = (1979, 2015),
    val_years: tuple[int, int] = (2016, 2017),
    test_years: tuple[int, int] = (2018, 2021),
) -> TemporalSplits:
    """Split a datetime sequence by calendar-year ranges.

    Sorts unsorted input internally and maps indices back to the original order.
    Falls back to 80/10/10 proportional split when any year-based split is empty.
    """
    dts = np.asarray(datetimes)
    if dts.ndim != 1 or dts.size == 0:
        return TemporalSplits(
            train_idx=np.arange(dts.size, dtype=np.int64),
            val_idx=np.array([], dtype=np.int64),
            test_idx=np.array([], dtype=np.int64),
        )

    if not np.all(dts[:-1] <= dts[1:]):
        order = np.argsort(dts, kind="stable")
        dts_sorted = dts[order]
    else:
        order = None
        dts_sorted = dts

    years = dts_sorted.astype('datetime64[Y]').astype(int) + 1970

    def _range_mask(yrs: np.ndarray, yr_range: tuple[int, int]) -> np.ndarray:
        start, end = int(yr_range[0]), int(yr_range[1])
        return (yrs >= start) & (yrs <= end)

    train_mask = _range_mask(years, train_years)
    val_mask = _range_mask(years, val_years)
    test_mask = _range_mask(years, test_years)

    train_idx_sorted = np.nonzero(train_mask)[0].astype(np.int64)
    val_idx_sorted = np.nonzero(val_mask)[0].astype(np.int64)
    test_idx_sorted = np.nonzero(test_mask)[0].astype(np.int64)

    # Proportional fallback when any split is empty
    if (
        train_idx_sorted.size < 1
        or val_idx_sorted.size < 1
        or test_idx_sorted.size < 1
    ):
        n = dts.size
        n_train = max(int(0.8 * n), 1)
        n_val = max(int(0.1 * n), 1)
        n_test = max(n - n_train - n_val, 1)
        if n_train + n_val + n_test > n:
            n_test = n - n_train - n_val
        train_idx_sorted = np.arange(0, n_train, dtype=np.int64)
        val_idx_sorted = np.arange(n_train, n_train + n_val, dtype=np.int64)
        test_idx_sorted = np.arange(n_train + n_val, n, dtype=np.int64)

    if order is not None:
        def backmap(sorted_idx: np.ndarray) -> np.ndarray:
            return order[sorted_idx]

        train_idx = backmap(train_idx_sorted)
        val_idx = backmap(val_idx_sorted)
        test_idx = backmap(test_idx_sorted)
        train_idx.sort()
        val_idx.sort()
        test_idx.sort()
    else:
        train_idx, val_idx, test_idx = train_idx_sorted, val_idx_sorted, test_idx_sorted

    return TemporalSplits(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

def load_statistics(path: str, pressure_levels, grid_variables) -> Dict[str, Any]:
    """Load per-variable normalisation statistics from xarray NetCDF files.

    Returns (mean, std, diff_std) tensors shaped [num_features], where features
    are ordered as: surface variables first, then pressure-level variables × levels.
    """
    mean = xr.open_dataset(f"{path}/stats-mean_by_level.nc").load()
    std = xr.open_dataset(f"{path}/stats-stddev_by_level.nc").load()
    diff_std = xr.open_dataset(f"{path}/stats-diffs_stddev_by_level.nc").load()

    mean_list, std_list, diff_std_list = [], [], []

    # First 5 variables are surface-level (no pressure dimension)
    for grid_var in grid_variables[:5]:
        mean_list.append(torch.from_numpy(mean[grid_var].values).reshape(1))
        std_list.append(torch.from_numpy(std[grid_var].values).reshape(1))
        diff_std_list.append(torch.from_numpy(diff_std[grid_var].values).reshape(1))

    # Remaining variables have one entry per pressure level
    for grid_var in grid_variables[5:]:
        for level in pressure_levels:
            mean_list.append(torch.from_numpy(mean[grid_var].sel(level=level).values).reshape(1))
            std_list.append(torch.from_numpy(std[grid_var].sel(level=level).values).reshape(1))
            diff_std_list.append(torch.from_numpy(diff_std[grid_var].sel(level=level).values).reshape(1))

    return torch.cat(mean_list), torch.cat(std_list), torch.cat(diff_std_list)