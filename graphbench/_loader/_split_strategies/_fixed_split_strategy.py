from typing import Dict, Optional, TYPE_CHECKING

from ._split_strategy import DatasetFactory, SplitStrategy


if TYPE_CHECKING:
    from torch_geometric.data import InMemoryDataset


class FixedSplitStrategy(SplitStrategy):
    def __init__(self, split_map: Optional[Dict[str, str]] = None) -> None:
        self.split_map = split_map or {"train": "train", "valid": "val", "test": "test"}

    def build(self, factory: DatasetFactory, dataset_name: str) -> Dict[str, InMemoryDataset]:
        return {
            key: factory(dataset_name, split_name, None)
            for key, split_name in self.split_map.items()
        }
