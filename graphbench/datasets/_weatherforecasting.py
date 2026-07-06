"""
weather forecasting dataset loader
----------------------------------

This module implements `WeatherforecastingDataset`, a PyG `InMemoryDataset`
that prepares graph-based weather forecasting examples. It downloads preprocessed weather data
which then can be used in downstream tasks. Furthermore, support for generation of the dataset is given (currently disabled)
"""

from __future__ import annotations

import time 
from pathlib import Path
from typing import  Dict, Optional, Any, Sequence, Tuple 
from loguru import logger
from graphbench._helpers import get_logger
from torch_geometric.data import Data, InMemoryDataset
from dataclasses import dataclass
import torch 
import pickle 
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable
# (i) helper functions

# -----------------------------------------------------------------------------#
# (a) Utilities
# -----------------------------------------------------------------------------#

_logger = get_logger(__name__)



# HuggingFace download URLs for prebuilt processed files.
_PREBUILT_PT_URLS = {
    "weather": "https://huggingface.co/datasets/log-rwth-aachen/Graphbench_Weather/resolve/main/weather_64_combined.pt",
    "weather_subset": "https://huggingface.co/datasets/log-rwth-aachen/GraphBench_Weather_Subset/resolve/main/weather_64_combined.pt",
}

# HuggingFace repo IDs for raw component files.
_HF_WEATHER_REPOS = {
    "weather": "log-rwth-aachen/Graphbench_Weather",
    "weather_subset": "log-rwth-aachen/GraphBench_Weather_Subset",
}

# Static component files that are shared across tasks and always sourced from
# the original weather repo (log-rwth-aachen/Graphbench_Weather).
_HF_WEATHER_STATIC_FILES = {
    "metadata.pkl": "metadata.pkl",
    "static_components.pkl": "static_components.pkl",
    "stats-mean_by_level.nc": "stats-mean_by_level.nc",
    "stats-stddev_by_level.nc": "stats-stddev_by_level.nc",
    "stats-diffs_stddev_by_level.nc": "stats-diffs_stddev_by_level.nc",
}

# Data files specific to each task (sourced from the task-specific repo).
# weather_64.pt is the raw timestep data; saved as weather_dataset.pt which is
# what EfficientWeatherGraphDataset.process() expects.
_HF_WEATHER_DATA_FILES = {
    "weather_64.pt": "weather_dataset.pt",
}

# Combined view for the full weather task (backward-compat).
_HF_WEATHER_RAW_FILES = {**_HF_WEATHER_STATIC_FILES, **_HF_WEATHER_DATA_FILES}



def _load_static_files(root: str):
    root = Path(root)
    metadata_path = root /"weather" /"metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:

        raise FileNotFoundError(f"Missing metadata.pkl in {metadata_path}. Please ensure the weather dataset is downloaded and processed correctly.")

    static_components_path = root /"weather" /"static_components.pkl"
    if static_components_path.exists():
        with open(static_components_path, 'rb') as f:
            static_components = pickle.load(f)
    else:
        raise FileNotFoundError(f"Missing static_components.pkl in {static_components_path}. Please ensure the weather dataset is downloaded and processed correctly.")

    return metadata, static_components

def _dist_rank_world():
    """Return the current process's (rank, world_size), or (0, 1) outside of DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _wait_for_weather_cache(ready_path: Path, timeout_s: int, poll_s: float = 5.0):
    """Block until rank 0 writes the ready marker, then returns. Used for DDP processes to coordinate loading of files."""
    start = time.time()
    while not ready_path.exists():
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for weather cache marker: {ready_path}"
            )
        time.sleep(poll_s)

def _prepare_weather_cache_once(root: str, timeout_s: int, skip_cache_build: bool, task_name: str = "weather", use_prebuilt: bool = False):
    """In DDP runs, rank 0 builds the processed cache; all other ranks wait for it.

    If use_prebuilt is True, rank 0 downloads the prebuilt processed file from
    HuggingFace instead of building from raw data (no-op if already present).
    Raw component files (static_components.pkl, metadata.pkl, etc.) are
    downloaded from HuggingFace automatically when missing.
    """
    #from wf_utils import EfficientWeatherGraphDataset

    root_path = Path(root)
    processed_path = root_path /"weather" / "processed" / "weather_graph_data_processed.pt"
    ready_path = root_path /"weather" / "processed" / "weather_graph_data_processed.ready"
    rank, world_size = _dist_rank_world()

    if world_size == 1:
        if use_prebuilt:
            _ensure_prebuilt_pt(task_name, processed_path)
        elif not processed_path.exists():
            _ensure_raw_weather_files(root, task_name)
        _download_static_weather_files(root)
        return

    if rank == 0:
        if use_prebuilt:
            _ensure_prebuilt_pt(task_name, processed_path)
        elif skip_cache_build:
            logger.warning("Rank 0 skipping weather cache build due to --skip_weather_cache_build")
        elif not processed_path.exists():
            _ensure_raw_weather_files(root, task_name)
            logger.info(f"Rank 0 building weather processed cache at {processed_path}")
            EfficientWeatherGraphDataset(root=root, pre_transform=None, transform=None)
        else:
            logger.info(f"Rank 0 reusing existing weather processed cache at {processed_path}")

        if processed_path.exists():
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(f"ready\nrank=0\nts={int(time.time())}\n", encoding="ascii")
        _download_static_weather_files(root)
        return

    # Non-zero ranks skip the wait if the cache already exists without a ready marker
    if not ready_path.exists() and processed_path.exists():
        logger.info(
            f"Rank {rank} found existing weather processed cache without ready marker; proceeding."
        )
        return

    logger.info(f"Rank {rank} waiting for weather cache marker at {ready_path}")
    _wait_for_weather_cache(ready_path, timeout_s=timeout_s)
    logger.info(f"Rank {rank} detected weather cache ready marker")


def _download_prebuilt_pt(url: str, dst: Path) -> None:
    """Download a prebuilt .pt file from a url to a destination path dst with a progress indicator."""
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    logger.info(f"Downloading prebuilt processed file from {url} ...")

    def _reporthook(count, block_size, total_size):
        if total_size > 0 and count % 500 == 0:
            mb_done = count * block_size / (1024 ** 2)
            mb_total = total_size / (1024 ** 2)
            logger.info(f"  {mb_done:.1f} / {mb_total:.1f} MB")

    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_reporthook)
        tmp.rename(dst)
        logger.info(f"Saved prebuilt processed file to {dst}")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_prebuilt_pt(task_name: str, processed_path: Path) -> None:
    """Download the prebuilt .pt for task_name if it is not already at processed_path."""
    if processed_path.exists():
        logger.info(f"Prebuilt processed file already present at {processed_path}; skipping download")
        return
    url = _PREBUILT_PT_URLS[task_name]
    _download_prebuilt_pt(url, processed_path)


def _download_static_weather_files(root: str) -> None:
    """Download the named static component files from the original weather
    repo (log-rwth-aachen/Graphbench_Weather) into `root/weather`.

    These files (metadata.pkl, static_components.pkl, stats-*.nc) are shared
    across all weather tasks and are always sourced from the same repository
    as the prebuilt datasets, regardless of which task variant is used.
    """
    dest_dir = Path(root) / "weather"
    origin_repo_id = _HF_WEATHER_REPOS["weather"]

    missing = [
        hf_name for hf_name, local_name in _HF_WEATHER_STATIC_FILES.items()
        if not (dest_dir / local_name).exists()
    ]
    if not missing:
        logger.info(f"All static weather component files already present in {dest_dir}")
        return

    logger.info(f"Missing static weather files: {missing}; downloading from {origin_repo_id}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; cannot auto-download static weather files. "
            "Install it with: pip install huggingface_hub"
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    for hf_name in missing:
        local_name = _HF_WEATHER_STATIC_FILES[hf_name]
        dst = dest_dir / local_name
        if dst.exists():
            continue
        logger.info(f"  Downloading {hf_name} from {origin_repo_id} -> {dst}")
        try:
            hf_hub_download(
                repo_id=origin_repo_id,
                filename=hf_name,
                repo_type="dataset",
                local_dir=str(dest_dir),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves to local_dir/<filename>; rename if needed
            downloaded = dest_dir / hf_name
            if downloaded.exists() and downloaded != dst:
                downloaded.rename(dst)
                logger.info(f"  Renamed {hf_name} -> {local_name}")
        except Exception as exc:
            logger.warning(f"  Failed to download {hf_name}: {exc}")


def _ensure_raw_weather_files(root: str, task_name: str) -> None:
    """Download raw component files from HuggingFace if they are missing.

    For weather_subset, static component files (metadata, static_components,
    stats NetCDFs) are sourced from the original Graphbench_Weather repo since
    the subset shares the same graph structure and normalisation statistics.
    Only the timestep data file (weather_64.pt) comes from the subset repo.
    """
    root_path = Path(root)
    task_repo_id = _HF_WEATHER_REPOS.get(task_name)
    origin_repo_id = _HF_WEATHER_REPOS["weather"]

    if task_repo_id is None:
        logger.warning(f"No HF repo configured for task {task_name}; skipping raw file download")
        return

    all_files = _HF_WEATHER_RAW_FILES
    missing = [
        hf_name for hf_name, local_name in all_files.items()
        if not (root_path / local_name).exists()
    ]
    if not missing:
        logger.info(f"All raw weather component files already present in {root_path}")
        return

    logger.info(f"Missing raw weather files for '{task_name}': {missing}; downloading")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; cannot auto-download raw weather files. "
            "Install it with: pip install huggingface_hub"
        )
        return

    root_path.mkdir(parents=True, exist_ok=True)
    for hf_name in missing:
        local_name = all_files[hf_name]
        dst = root_path / local_name
        if dst.exists():
            continue
        # Static components are always sourced from the original weather repo.
        if hf_name in _HF_WEATHER_STATIC_FILES:
            repo_id = origin_repo_id
        else:
            repo_id = task_repo_id
        logger.info(f"  Downloading {hf_name} from {repo_id} -> {dst}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_name,
                repo_type="dataset",
                local_dir=str(root_path),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves to local_dir/<filename>; rename if needed
            downloaded = root_path / hf_name
            if downloaded.exists() and downloaded != dst:
                downloaded.rename(dst)
                logger.info(f"  Renamed {hf_name} -> {local_name}")
        except Exception as exc:
            logger.warning(f"  Failed to download {hf_name}: {exc}")

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

class EfficientWeatherGraphDataset(InMemoryDataset):
    """
    Weather forecasting dataset.

    Note:
        This class **should not be used directly**, please use :class:`graphbench.Loader` instead to access the provided
        datasets.
        The purpose of this page is merely to provide details on the dataset.


    Overview:
        We provide a graph-based medium-range weather forecasting dataset derived from the
        `ERA5 <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>`_ reanalysis dataset.
        We use a down-sampled version of ERA5 with a ``64 x 32`` equiangular grid and a temporal resolution of six
        hours.

        The task is to model medium-range weather evolution by predicting the residual change in the atmospheric
        state over a fixed 12-hour horizon.
        Given an initial snapshot of the current atmospheric state, the model forecasts the 12-hour future change
        in meteorological variables at each grid location.

        Since the dataset contains node features of various scales and units, we provide pre-computed mean and standard deviation values from the GraphCast team for each variable across all pressure levels.
        We refer to the `GraphCast paper <https://arxiv.org/abs/2212.12794>`_ for details on the computation and usage of these statistics. These statistics are downloaded alongside the weather dataset as  "stats-mean_by_level.nc",
        "stats-stddev_by_level.nc" and "stats-diffs_stddev_by_level.nc". 

        Additional metadata used during dataset preparation and model training such as latitude and longitude are provided in metadata.pkl and static_components.pkl.
        These metadata and static component values can also be accessed from each split of the dataset via the ``metadata`` and ``static_components`` attributes of the returned PyG Data objects.
        Example: given the training split via ``train_dataset = Loader("data", "weather").load()[0]["train"]``, the metadata can be accessed via ``train_dataset.metadata`` and the static components via ``train_dataset.static_components``.

    Graph Attributes:
        .. list-table::
           :header-rows: 1

           * - Attribute
             - Size
             - Description
           * - ``grid_x``
             - ``[num_nodes, 166]``
             - Node features: weather variables across all pressure levels for each grid coordinate of the current ([0:83]) and previous time step ([83:]).
           * - timestep_idx
             - ``[1]``
             - Given timestep index to attribute a specific time to each graph. 
           * - edge_index
             - ``[2,]``
             - Contains the complete edge index for both grid and mesh nodes. 
           * - mesh_edge_index
             - ``[2,]``
             - The edge index for the mesh graph as used in GraphCast. Contains only the edges of the mesh nodes.
           * - node_type
             - ``[4610]``
             - Denotes whether a node belongs to the grid or mesh graph. 
           * - x
             - ``[2562, 1]``
             - Placeholder weather variable for the mesh nodes computed during the model forward pass. 

        Please refer to the `GraphBench paper <https://arxiv.org/abs/2512.04475>`_ for a detailed list of the weather
        variables included in the dataset.

    Metadata Attributes: 
        .. list-table::
           :header-rows: 1

           * - Attribute
             - Size
             - Description
           * - ``grid_variables``
             - [11]
             - The variable names for all grid variables. Includes both pressure dependent and independent variables.
           * - ``mesh_splits``
             - [1]
             - The number of mesh splits.
           * - ``connectivity_radius``
             - [1]
             - The connectivity radius used during mesh generation.
           * - ``grid_lat``
             - [64]
             - Grid latitute values used for loss computation and distance computation.
           * - ``grid_lon``
             - [64]
             - Grid longitude values used for loss computation and distance computation.
           * - ``num_timesteps``
             - [1]
             - Number of total timesteps in the dataset.
           * - ``datetimes``
             - [93544]
             - Datetime objects for each timestep in the dataset. Can be used to compute temporal splits for training, validation and testing.
           * - ``variable_channel_counts``
             - [11]
             - Number of appearances of variables for each variable type.
           * - ``variable_channel_slices``
             - [11]
             - Channel slices for each variable channel in grid_x features to assign correct loss values.

    Static components Attributes: 
        .. list-table::
           :header-rows: 1

           * - ``mesh_vertices``
             - [2562,3]
             - 3D coordinates of each mesh vertex used in the mesh graph. Not used in current computations.
           * - ``variable_channel_counts``
             - [11]
             - Number of appearances of variables for each variable type.
           * - ``variable_channel_slices``
             - [11]
             - Channel slices for each variable channel in grid_x features to assign correct loss values.
           * - ``mesh_faces``
             - [2562,3]
             - Node IDs of the mesh faces used to compute the mesh edge index.
           * - ``mesh_lat``
             - [2562]
             - Mesh latitude values used for loss computation and distance computation.
           * - ``mesh_lon``
             - [2562]
             - Mesh longitude values used for loss computation and distance computation.
           * - ``grid2mesh_data``
             - NaN
             - Data object holding the information for the grid to mesh data. Used in construction from single time steps. Not used for the already combined dataset (default).
           * - ``grid_lat``
             - [64]
             - Grid latitude values used for loss computation and distance computation.
           * - ``grid_lon``
             - [64]
             - Grid longitude values used for loss computation and distance computation.
           * - ``num_grid_nodes``
             - [1]
             - The number of grid nodes.
           * - ``num_mesh_nodes``
             - [1]
             - The number of mesh nodes.

    Note: 
        Currently the weather dataset is only evaluated on a twelve hour forecast time window. The target values can be obtained by using the current time step values for each node of timestep t+2 with timestep t as an input. 
        Therefore no value is set for the graph attribute y. 

    List of Available Datasets:
        We currently provide a single dataset, called ``weather``.

        It can be loaded like this:

        .. code:: python

            from graphbench import Loader
            dataset = Loader("data", "weather").load()
    """

    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root: Root directory containing the ``weather`` folder with raw and processed files.
            transform: Optional PyG transform applied to data objects before every access.
            pre_transform: Optional PyG transform applied before saving data objects to disk.
            pre_filter: A function that indicates whether a data object should be included in the final dataset.
        """
        self.root_path = Path(root)

        self.metadata_path = self.root_path /"weather" /"metadata.pkl"
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

        self.static_components_path = self.root_path /"weather" /"static_components.pkl"
        if self.static_components_path.exists():
            with open(self.static_components_path, 'rb') as f:
                self.static_components = pickle.load(f)
        else:
            self.static_components = {}

        # Cached references to static tensors shared across all samples
        self._cached_splits: Optional[Dict[str, np.ndarray]] = None
        self._static_edge_index: Optional[torch.Tensor] = None
        self._static_mesh_edge_index: Optional[torch.Tensor] = None
        self._static_node_type: Optional[torch.Tensor] = None
        self._static_mesh_x: Optional[torch.Tensor] = None
        self._refresh_static_cache()

        super().__init__(str(self.root_path / "weather"), transform, pre_transform)
        self.load(self.processed_paths[0])
        self._refresh_static_cache()

    def load(self, path: str, data_cls=Data) -> None:
        """Load processed data, trying mmap first to reduce RSS spikes."""
        load_attempts = [
            {"map_location": "cpu", "weights_only": False, "mmap": True},
            {"map_location": "cpu", "mmap": True},
            {"map_location": "cpu", "weights_only": False},
            {"map_location": "cpu"},
        ]

        out = None
        for kwargs in load_attempts:
            try:
                out = torch.load(path, **kwargs)
                break
            except (TypeError, RuntimeError, ValueError):
                continue

        if out is None:
            out = torch.load(path)

        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        self.data = data if not isinstance(data, dict) else data_cls.from_dict(data)

    @property
    def raw_file_names(self):
        """Returns list of files that must be present to trigger process()."""
        files = ["static_components.pkl", "metadata.pkl"]
        if (self.root_path /"weather"/"weather_dataset.pt").exists():
            files.append("weather_dataset.pt")
        return files

    @property
    def processed_file_names(self):
        """Returns name of the cached file produced by process()."""
        return ['weather_graph_data_processed.pt']

    def _load_torch_payload(self, file_path: Path) -> Any:
        """Load a torch file with mmap-first fallback to reduce RSS spikes."""
        load_attempts = [
            {"map_location": "cpu", "weights_only": False, "mmap": True},
            {"map_location": "cpu", "mmap": True},
            {"map_location": "cpu", "weights_only": False},
            {"map_location": "cpu"},
        ]
        for kwargs in load_attempts:
            try:
                return torch.load(file_path, **kwargs)
            except (TypeError, RuntimeError, ValueError):
                continue
        return torch.load(file_path)

    def _refresh_static_cache(self) -> None:
        """Pre-compute and cache static graph tensors shared across all timesteps."""
        if not self.static_components:
            return

        grid2mesh_data = self.static_components.get('grid2mesh_data')
        mesh_faces = self.static_components.get('mesh_faces')
        num_mesh_nodes = int(self.static_components.get('num_mesh_nodes', 0))

        if grid2mesh_data is None or mesh_faces is None or num_mesh_nodes <= 0:
            return

        self._static_edge_index = grid2mesh_data.edge_index
        self._static_node_type = grid2mesh_data.node_type

        # Build bidirectional mesh edge index from triangular faces
        faces = torch.as_tensor(mesh_faces, dtype=torch.long)
        tri_edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], dim=0).t().contiguous()
        self._static_mesh_edge_index = torch.cat([tri_edges, tri_edges.flip(0)], dim=1)
        self._static_mesh_x = torch.zeros(num_mesh_nodes, 1, dtype=torch.float32)

    def _build_dynamic_graph(
        self,
        grid_features: torch.Tensor,
        grid_features_prev: torch.Tensor,
        timestep_idx: int,
    ) -> Data:
        """Create a minimal per-timestep Data object with only dynamic tensors."""
        return Data(
            grid_x=torch.cat([grid_features, grid_features_prev], dim=-1),
            timestep_idx=timestep_idx,
        )

    def _attach_static_fields(self, data: Data) -> Data:
        """Attach cached static tensors to a dynamic Data object at read time."""
        if self._static_edge_index is None or self._static_mesh_edge_index is None or self._static_node_type is None:
            return data

        data.edge_index = self._static_edge_index
        data.mesh_edge_index = self._static_mesh_edge_index
        data.node_type = self._static_node_type

        if self._static_mesh_x is not None:
            grid_x = getattr(data, 'grid_x', None)
            if isinstance(grid_x, torch.Tensor) and grid_x.dtype != self._static_mesh_x.dtype:
                data.x = self._static_mesh_x.to(dtype=grid_x.dtype)
            else:
                data.x = self._static_mesh_x

        return data

    def process(self):
        """Reads time step data, provides additional static and dynamic features if not prebuilt and collates data into a single processed .pt file."""
        print("Processing dataset... this might take a while.")

        with open(self.root_path / "weather" / "static_components.pkl", 'rb') as f:
            static_components = pickle.load(f)
            self.static_components = static_components
        self._refresh_static_cache()

        # When no pre_filter/pre_transform is set, store dynamic fields only to save memory
        use_dynamic_only_storage = self.pre_filter is None and self.pre_transform is None

        combined_file_path = self.root_path /"weather" / "weather_dataset.pt"
        data_list = []

        if combined_file_path.exists():
            # Load all timesteps from a single combined file [T, N, F]
            print(f"Loading raw data from {combined_file_path}...")
            loaded_raw = self._load_torch_payload(combined_file_path)
            all_grid_features = loaded_raw['grid_features']
            num_timesteps = all_grid_features.shape[0]
            prev_grid_features: Optional[torch.Tensor] = None

            for t in tqdm(range(num_timesteps), desc="Loading timesteps", unit="step"):
                grid_features = all_grid_features[t]
                if prev_grid_features is None:
                    prev_grid_features = torch.zeros_like(grid_features)

                if use_dynamic_only_storage:
                    graph = self._build_dynamic_graph(grid_features, prev_grid_features, t)
                else:
                    timestep_data = {'grid_features': grid_features}
                    timestep_data_prev = {'grid_features': prev_grid_features}
                    graph = self._combine_static_and_dynamic(static_components, timestep_data, t, timestep_data_prev)

                if self.pre_filter is not None and not self.pre_filter(graph):
                    prev_grid_features = grid_features
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)

                data_list.append(graph)
                prev_grid_features = grid_features

        else:
            # Load timesteps from individual timestep_*.pt files
            timestep_files = sorted(list(self.root_path.glob("timestep_*.pt")))
            num_timesteps = len(timestep_files)
            print(f"Loading raw data from {num_timesteps} individual files...")
            prev_grid_features: Optional[torch.Tensor] = None

            for t, file_path in enumerate(tqdm(timestep_files, desc="Loading timesteps", unit="file")):
                timestep_data = self._load_torch_payload(file_path)
                grid_features = timestep_data['grid_features']
                if prev_grid_features is None:
                    prev_grid_features = torch.zeros_like(grid_features)

                if use_dynamic_only_storage:
                    graph = self._build_dynamic_graph(grid_features, prev_grid_features, t)
                else:
                    timestep_data_prev = {'grid_features': prev_grid_features}
                    graph = self._combine_static_and_dynamic(static_components, timestep_data, t, timestep_data_prev)

                if self.pre_filter is not None and not self.pre_filter(graph):
                    prev_grid_features = grid_features
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)

                data_list.append(graph)
                prev_grid_features = grid_features

        print("Collating and saving...")
        self.save(data_list, self.processed_paths[0])
        print("Done!")

    def _combine_static_and_dynamic(self, static: Dict, timestep_data: Dict[str, Any], timestep_idx: int, timestep_data_prev: Dict[str, Any]) -> Data:
        """Builds the completed weather data object by combining static and dynamic features."""
        grid_features = timestep_data['grid_features']
        grid_features_prev = timestep_data_prev['grid_features']
        graph_data = self._build_dynamic_graph(grid_features, grid_features_prev, timestep_idx)
        return self._attach_static_fields(graph_data)

    def _get_split_indices(self) -> Dict[str, np.ndarray]:
        """Return cached train/val/test index arrays, computing them on first call."""
        if self._cached_splits is not None:
            return self._cached_splits

        num_timesteps = len(self)
        datetimes = self.metadata.get('datetimes', None)
        split_dict = {}
        computed = False

        if datetimes is not None and len(datetimes) == num_timesteps:
            try:
                splits = compute_temporal_splits(datetimes, test_duration_days=365, val_duration_days=180)
                split_dict = {
                    'train': splits.train_idx,
                    'val': splits.val_idx,
                    'test': splits.test_idx,
                }
                computed = True
            except Exception:
                pass

        if not computed:
            # Proportional 80/10/10 fallback
            n_train = max(int(0.8 * num_timesteps), 1)
            n_val = max(int(0.1 * num_timesteps), 1)
            idx = np.arange(num_timesteps, dtype=np.int64)
            split_dict = {
                'train': idx[:n_train],
                'val': idx[n_train:n_train + n_val],
                'test': idx[n_train + n_val:],
            }

        self._cached_splits = split_dict
        return split_dict

    def __getitem__(self, index):
        """Support string split keys ('train', 'val'/'valid', 'test') alongside integer indexing."""
        if isinstance(index, str):
            key = index.lower()
            if key == 'valid':
                key = 'val'
            if key in ('train', 'val', 'test'):
                splits = self._get_split_indices()
                return self.index_select(splits[key])

        # Re-attach static fields for dynamic-only storage mode
        data_or_subset = super().__getitem__(index)
        if isinstance(data_or_subset, Data):
            return self._attach_static_fields(data_or_subset)
        return data_or_subset


