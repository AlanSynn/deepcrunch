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

from deepcrunch.quantization.base_quantizer import BaseQuantizer
from deepcrunch.utils.exceptions import InvalidQuantizationConfigException
from deepcrunch.utils.os_utils import LazyImport

quantize_dynamic = LazyImport("torch.quantization.quantize_dynamic")


class NaiveQuantizer(BaseQuantizer):
    @staticmethod
    def quantization_algorithm():
        return quantize_dynamic

    @staticmethod
    def quantize(model, qconfig_spec, dtype, inplace):
        model_quantized = quantize_dynamic(
            model=model,
            qconfig_spec={nn.LSTM, nn.Linear},
            dtype=torch.qint8,
            inplace=False,
        )
        return quantize_dynamic(model, qconfig_spec, dtype, inplace)
