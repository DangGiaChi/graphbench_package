from typing import Dict, Optional

from ._split_strategy import DatasetFactory, SplitStrategy


class FixedSplitStrategy(SplitStrategy):
    def __init__(self, split_map: Optional[Dict[str, str]] = None) -> None:
        self.split_map = split_map or {"train": "train", "valid": "val", "test": "test"}

    def build(self, factory: DatasetFactory, dataset_name: str) -> Dict[str, object]:
        return {
            key: factory(dataset_name, split_name, None)
            for key, split_name in self.split_map.items()
        }
