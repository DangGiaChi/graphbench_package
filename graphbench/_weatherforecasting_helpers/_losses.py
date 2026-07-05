"""Weighted MSE loss and metric-breakdown helpers for weather forecasting."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


"""def _weather_loss_sum(y_pred, y_true, grid_lat):
    device = y_pred.device
    if y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
    mask = torch.isfinite(y_true)
    if not mask.any():
        return torch.zeros((), device=device, dtype=y_pred.dtype)
    y_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))
    y_true = torch.where(mask, y_true, torch.zeros_like(y_true))
    lat_rad = torch.tensor(np.deg2rad(grid_lat), dtype=torch.float32, device=device)
    lat_w = torch.cos(lat_rad)
    lat_w = lat_w / lat_w.mean()
    num_nodes = y_pred.shape[1]
    num_grids = num_nodes // lat_w.shape[0]
    lat_w = lat_w.repeat(num_grids)
    sq_err = (y_pred - y_true) ** 2
    return (sq_err * lat_w.view(1, -1, 1)).mean()"""

def compute_latitude_weights(latitude_values: np.ndarray) -> torch.Tensor:
    """Compute cos(lat) area weights for grid cells, normalised to mean 1."""
    lat_rad = np.deg2rad(latitude_values)
    weights = np.cos(lat_rad)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_pressure_level_weights(pressure_levels: np.ndarray) -> torch.Tensor:
    """Compute pressure-level weights proportional to the pressure level, normalised to mean 1."""
    weights = pressure_levels / np.mean(pressure_levels)
    return torch.tensor(weights, dtype=torch.float32)


def get_default_pressure_levels() -> np.ndarray:
    """Get default pressure levels matching the GraphCast configuration."""
    return np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.float32)


def compute_weather_metric_breakdown(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    feature_group_slices: List[Tuple[int, int]],
    feature_group_names: List[str],
    multi_level_variable_names: Optional[set] = None,
    pressure_levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute per-variable-group MSE, and per-pressure-level MSE for multi-level variables.

    NaN/Inf targets are masked out of the mean rather than replaced with zero, so
    padded or missing grid points don't bias the reported MSE.

    Args:
        y_pred: Predictions [..., num_features].
        y_true: Targets, same shape as `y_pred`.
        feature_group_slices: [start, end) channel slice per variable group.
        feature_group_names: Variable name per slice in `feature_group_slices`.
        multi_level_variable_names: Names of variables to additionally break down
            by pressure level. Defaults to the standard GraphCast multi-level variables.
        pressure_levels: Pressure level values corresponding to each channel within a
            multi-level variable's slice. Defaults to `get_default_pressure_levels()`.

    Returns:
        Dict mapping ``"{name}_mse"`` and ``"{name}_z{level}_mse"`` to scalar MSE values
        (``nan`` for groups/levels with no finite targets).
    """
    if multi_level_variable_names is None:
        multi_level_variable_names = {
            "specific_humidity", "vertical_velocity", "u_component_of_wind",
            "v_component_of_wind", "geopotential", "temperature",
        }
    if pressure_levels is None:
        pressure_levels = get_default_pressure_levels()

    mask = torch.isfinite(y_true)
    sq_err = (y_pred - y_true) ** 2
    reduce_dims = tuple(range(sq_err.dim() - 1)) if sq_err.dim() > 1 else ()
    masked_sq_err = torch.where(mask, sq_err, torch.zeros_like(sq_err))
    feat_sum = masked_sq_err.sum(dim=reduce_dims).detach().cpu().to(torch.float64).numpy()
    feat_count = mask.sum(dim=reduce_dims).detach().cpu().to(torch.float64).numpy()

    def _mse(sum_val: float, count_val: float) -> float:
        return float(sum_val / count_val) if count_val > 0 else float("nan")

    metrics: Dict[str, float] = {}
    for name, (start, end) in zip(feature_group_names, feature_group_slices):
        start, end = int(start), int(end)
        metrics[f"{name}_mse"] = _mse(feat_sum[start:end].sum(), feat_count[start:end].sum())

        if name in multi_level_variable_names:
            levels_count = end - start
            for i in range(min(len(pressure_levels), levels_count)):
                level = pressure_levels[i]
                metrics[f"{name}_z{level}_mse"] = _mse(feat_sum[start + i], feat_count[start + i])

    return metrics
