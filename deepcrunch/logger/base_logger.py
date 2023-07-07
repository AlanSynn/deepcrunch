from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config

    @abstractmethod
    def log(self, data: Dict[str, Any], step: int, commit: Optional[bool] = None,
            sync: Optional[bool] = None):
        pass

    def get_attr(self, attr: str) -> Optional[str]:
        return getattr(self, attr, None)

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config}
