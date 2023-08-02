import itertools
from copy import copy
from enum import IntEnum
from typing import Optional

from deepcrunch import logger
from deepcrunch.backend.engines.base_backend import (
    PostTrainingQuantizationBaseWrapper as PTQBase,
)
from deepcrunch.utils.os_utils import LazyImport
from deepcrunch.utils.time import log_elapsed_time

torch = LazyImport("torch")

QconfigMapping = LazyImport("torch.ao.quantization.QConfigMapping")
get_default_qat_qconfig_mapping = LazyImport(
    "torch.ao.quantization.get_default_qat_qconfig_mapping"
)
get_default_qconfig_mapping = LazyImport(
    "torch.ao.quantization.get_default_qconfig_mapping"
)

_activation_is_memoryless = LazyImport(
    "torch.ao.quantization.qconfig._activation_is_memoryless"
)
_add_module_to_qconfig_obs_ctr = LazyImport(
    "torch.ao.quantization.qconfig._add_module_to_qconfig_obs_ctr"
)
default_dynamic_qconfig = LazyImport(
    "torch.ao.quantization.qconfig.default_dynamic_qconfig"
)
float16_dynamic_qconfig = LazyImport(
    "torch.ao.quantization.qconfig.float16_dynamic_qconfig"
)
float_qparams_weight_only_qconfig = LazyImport(
    "torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig"
)
float_qparams_weight_only_qconfig_4bit = LazyImport(
    "torch.ao.quantization.qconfig.float_qparams_weight_only_qconfig_4bit"
)

# import torch.quantization
# from torch.quantization import QuantStub, DeQuantStub


class TORCH_PTQ(IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    FX = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str):
        return TORCH_PTQ[string.upper()]


DTYPE_MAP = {
    "qint8": torch.qint8,
    "quint8": torch.quint8,
    "qint32": torch.qint32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "quint4x2": torch.quint4x2,
}


class TorchPTQ(PTQBase):
    def __init__(
        self,
        model: torch.nn.Module = None,
        type: int = TORCH_PTQ.DYNAMIC,
        config_path: Optional[str] = None,
    ):
        """
        Args:
            model: A trained PyTorch or TensorFlow model. Default is None.
            type: The type of quantization to perform. Default is TORCH_PTQ.DYNAMIC.
            config_path: Path to the YAML configuration file for quantization. Default is None.
        """
        self.model = model
        self.type = type
        self.config_path = config_path
        super().__init__(self.model, self.config_path)

    # @log_elapsed_time(customized_msg="Quantization time: {elapsed_time:.2f} seconds")
    def quantize(self, *args, **kwargs):
        """Quantize the model and save it to the specified path.

        Args:
            type: The type of quantization to perform. Default is dynamic.
        """

        type_str = kwargs.get("type", "dynamic")
        type = TORCH_PTQ.from_str(type_str)

        rtn = None

        if type == TORCH_PTQ.DYNAMIC:
            rtn = self.quantize_dynamic(*args, **kwargs)
        elif type == TORCH_PTQ.STATIC:
            rtn = self.quantize_static(*args, **kwargs)
        elif type == TORCH_PTQ.QAT:
            rtn = self.quantize_qat(*args, **kwargs)
        elif type == TORCH_PTQ.FX:
            rtn = self.quantize_fx(*args, **kwargs)
        else:
            raise ValueError(
                f"Invalid quantization type: {type}. Please choose from: TORCH_PTQ.DYNAMIC, TORCH_PTQ.STATIC, TORCH_PTQ.QAT, TORCH_PTQ.FX"
            )

        return rtn

    def quantize_dynamic(
        self, model, output_path: Optional[str] = None, *args, **kwargs
    ):
        """Quantize the model using dynamic quantization and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """
        nn = torch.nn

        qconfig_spec = kwargs.get("qconfig_spec", None)
        dtype = kwargs.get("dtype", "bfloat16")

        dtype = DTYPE_MAP.get(dtype, None)

        if qconfig_spec is None:
            if dtype == torch.qint8:
                qconfig_spec = {
                    nn.Linear: default_dynamic_qconfig,
                    nn.LSTM: default_dynamic_qconfig,
                    nn.GRU: default_dynamic_qconfig,
                    nn.LSTMCell: default_dynamic_qconfig,
                    nn.RNNCell: default_dynamic_qconfig,
                    nn.GRUCell: default_dynamic_qconfig,
                }
            elif dtype == torch.float16:
                qconfig_spec = {
                    nn.Linear: float16_dynamic_qconfig,
                    nn.LSTM: float16_dynamic_qconfig,
                    nn.GRU: float16_dynamic_qconfig,
                    nn.LSTMCell: float16_dynamic_qconfig,
                    nn.RNNCell: float16_dynamic_qconfig,
                    nn.GRUCell: float16_dynamic_qconfig,
                }
            elif dtype == torch.quint8:
                qconfig_spec = {
                    nn.EmbeddingBag: float_qparams_weight_only_qconfig,
                    nn.Embedding: float_qparams_weight_only_qconfig,
                }
            elif dtype == torch.quint4x2:
                qconfig_spec = {
                    nn.EmbeddingBag: float_qparams_weight_only_qconfig_4bit,
                }
            else:
                raise ValueError(
                    "Don't know how to quantize with default settings for {}. Provide full qconfig please".format(
                        dtype
                    )
                )
        elif isinstance(qconfig_spec, set):
            if dtype is torch.qint8:
                default_qconfig = default_dynamic_qconfig
            elif dtype is torch.float16:
                default_qconfig = float16_dynamic_qconfig
            elif dtype is torch.quint8:
                default_qconfig = float_qparams_weight_only_qconfig
            elif dtype is torch.quint4x2:
                default_qconfig = float_qparams_weight_only_qconfig_4bit
            else:
                raise RuntimeError(
                    "Unknown dtype specified for quantize_dynamic: ", str(dtype)
                )
            qconfig_spec = dict(zip(qconfig_spec, itertools.repeat(default_qconfig)))

        # Quantize the model
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model, qconfig_spec=qconfig_spec, dtype=dtype
        )

        # Save the quantized model
        self.save_quantized_model(quantized_model, output_path)

        return quantized_model

    def quantize_static(self, model, output_path: Optional[str] = None):
        """Quantize the model using static quantization and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        quantize_list = {
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.EmbeddingBag,
            torch.nn.Embedding,
        }

        # Quantize the model
        quantized_model = torch.ao.quantization.quantize_static(
            model, quantize_list, dtype=torch.qint8
        )

        # Save the quantized model
        self.save_quantized_model(quantized_model, output_path)

        return quantized_model

    def quantize_qat(
        self,
        model,
        output_path: Optional[str] = None,
        input_dim: tuple = (1, 3, 224, 224),
    ):
        """Quantize the model using quantization-aware training and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        quantized_model = copy.deepcopy(self.model)

        # Prepare the model for quantization-aware training
        quantized_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        torch.ao.quantization.prepare_qat(quantized_model, inplace=True)

        # Train the model
        quantized_model.train()

        # Run the model on a dummy input
        dummy_input = torch.randn(input_dim)
        quantized_model(dummy_input)

        # Convert the model to a quantized model
        torch.ao.quantization.convert(quantized_model, inplace=True)

        # Save the quantized model
        self.save_quantized_model(quantized_model, output_path)

        return self.model

    def quantize_fx(
        self, output_path: Optional[str] = None, input_dim: tuple = (1, 3, 224, 224)
    ):
        """Quantize the model using FX graph mode quantization and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """
        import torch.ao.quantization.quantize_fx as quantize_fx

        # Quantize the model
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.eval()

        qconfig_mapping = QConfigMapping().set_global(
            torch.ao.quantization.default_dynamic_qconfig
        )

        example_inputs = torch.randn(input_dim)

        model_prepared = quantize_fx.prepare_fx(
            model_to_quantize, qconfig_mapping, example_inputs
        )
        # quantize
        quantized_model = quantize_fx.convert_fx(model_prepared)

        # fuse modules
        quantized_model = quantize_fx.fuse_fx(quantized_model)

        # Save the quantized model
        self.save_quantized_model(quantized_model, output_path)

        return quantized_model

    @staticmethod
    def save_quantized_model(quantized_model, output_path: Optional[str]):
        """Save the quantized model to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        if output_path is not None:
            torch.save(quantized_model, output_path)
            print(f"Quantized model saved to {output_path}")
            return quantized_model

    def get_type(self):
        """Get the type of quantization.

        Returns:
            The type of quantization.
        """
        return self.type

    def get_config_path(self):
        """Get the path to the YAML configuration file for quantization.

        Returns:
            The path to the YAML configuration file for quantization.
        """
        return self.config_path

    def get_inference_model(self, model, input_dim: tuple = (1, 3, 224, 224)):
        """Get the inference model.

        Returns:
            The inference model.
        """

        # For now, we suggest to disable the Jit Autocast Pass,
        # As the issue: https://github.com/pytorch/pytorch/issues/75956

        torch._C._jit_set_autocast_mode(False)

        with torch.cpu.amp.autocast(cache_enabled=False):
            model = torch.jit.trace(model, torch.randn(*input_dim))
        inf_model = torch.jit.freeze(model)

        return inf_model
