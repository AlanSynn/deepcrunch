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

import unittest
from deepcrunch.core import ModelWrapper, TrainerWrapper

class TestCore(unittest.TestCase):

    def test_model_wrapper(self):
        # Test ModelWrapper class
        model = ModelWrapper()
        self.assertIsInstance(model, ModelWrapper)

    def test_trainer_wrapper(self):
        # Test TrainerWrapper class
        trainer = TrainerWrapper()
        self.assertIsInstance(trainer, TrainerWrapper)

if __name__ == '__main__':
    unittest.main()