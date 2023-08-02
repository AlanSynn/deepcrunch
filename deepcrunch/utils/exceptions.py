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


class BaseException(Exception):
    """Base exception class that other exception classes can inherit from."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class MisconfigurationException(BaseException):
    """Exception used to inform users of the given configuration is invalid."""

    def __init__(self, message="Invalid configuration."):
        super().__init__(message)


class InvalidQuantizationConfigException(BaseException):
    """Exception used to inform users of the given quantization config is invalid."""

    def __init__(self, message="Invalid quantization config."):
        super().__init__(message)


class InvalidParallelModeException(BaseException):
    """Exception used to inform users of the given parallel mode is invalid."""

    def __init__(self, message="Invalid parallel mode."):
        super().__init__(message)
