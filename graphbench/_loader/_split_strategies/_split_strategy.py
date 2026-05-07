from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, TypeAlias

from torch_geometric.data import InMemoryDataset


DatasetFactory: TypeAlias = Callable[[str, str, Optional[str]], InMemoryDataset]


class SplitStrategy(ABC):
    @abstractmethod
    def build(self, factory: DatasetFactory, dataset_name: str) -> Dict[str, InMemoryDataset]:
        ...
