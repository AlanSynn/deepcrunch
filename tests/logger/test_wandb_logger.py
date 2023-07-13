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

from unittest import TestCase, mock
from deepcrunch.logger.wandb_logger import WandbLogger


class TestWandbLogger(TestCase):
    def setUp(self):
        self.logger = WandbLogger()

    def test_log_scalar(self):
        with mock.patch("wandb.log") as mock_log:
            self.logger.log_scalar("loss", 0.5, step=1)
            mock_log.assert_called_once_with({"loss": 0.5}, step=1)

    def test_log_image(self):
        with mock.patch("wandb.log") as mock_log:
            self.logger.log_image("image", "path/to/image.png", step=1)
            mock_log.assert_called_once_with(
                {"image": wandb.Image("path/to/image.png")}, step=1
            )

    def test_log_histogram(self):
        with mock.patch("wandb.log") as mock_log:
            self.logger.log_histogram("weights", [1, 2, 3], step=1)
            mock_log.assert_called_once_with(
                {"weights": wandb.Histogram([1, 2, 3])}, step=1
            )
