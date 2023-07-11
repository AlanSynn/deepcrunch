# Copyright 2023 -ignore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLogger(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config

    @abstractmethod
    def log(self, data: Dict[str, Any], step: int, commit: Optional[bool] = None, sync: Optional[bool] = None):
        pass

    def get_attr(self, attr: str) -> Optional[str]:
        return getattr(self, attr, None)

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config}
