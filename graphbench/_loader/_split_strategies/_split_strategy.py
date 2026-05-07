
from typing import Callable, Dict, Optional, TypeAlias


DatasetFactory: TypeAlias = Callable[[str, str, Optional[str]], object]


class SplitStrategy:
    def build(self, factory: DatasetFactory, dataset_name: str) -> Dict[str, object]:
        raise NotImplementedError

