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

from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)

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

class TorchPTQ(PTQBase):
    def __init__(self, model: torch.nn.Module=None, type: int=TORCH_PTQ.DYNAMIC, config_path: Optional[str]=None):
        """
        Args:
            model: A trained PyTorch or TensorFlow model. Default is None.
            type: The type of quantization to perform. Default is TORCH_PTQ.DYNAMIC.
            config_path: Path to the YAML configuration file for quantization. Default is None.
        """
        self.model = model
        self.quantized_model = None
        self.type = type
        self.config_path = config_path
        super().__init__()

    @log_elapsed_time(customized_msg="Quantization time: {elapsed_time:.2f} seconds")
    def quantize(self, type: int=TORCH_PTQ.DYNAMIC, output_path: Optional[str]=None):
        """Quantize the model and save it to the specified path.

        Args:
            type: The type of quantization to perform. Default is TORCH_PTQ.DYNAMIC.
            output_path: Path to save the quantized model.
        """

        if type == TORCH_PTQ.DYNAMIC:
            self.quantize_dynamic(output_path)
        elif type == TORCH_PTQ.STATIC:
            self.quantize_static(output_path)
        elif type == TORCH_PTQ.QAT:
            self.quantize_qat(output_path)
        elif type == TORCH_PTQ.FX:
            self.quantize_fx(output_path)
        else:
            raise ValueError("Invalid quantization type.")

        return self.model

    def quantize_dynamic(self, output_path: Optional[str]=None):
        """Quantize the model using dynamic quantization and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        quantize_list = {
            torch.nn.Linear,
            torch.nn.LSTM,
            torch.nn.GRU,
            torch.nn.RNNCell,
            torch.nn.GRUCell,
            torch.nn.LSTMCell,
            torch.nn.EmbeddingBag,
        }

        # Quantize the model
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model, quantize_list, dtype=torch.qint8
        )

        # Save the quantized model
        self.save_quantized_model(output_path)

        return self.quantized_model

    def quantize_static(self, output_path: Optional[str]=None):
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
        self.quantized_model = torch.quantization.quantize_static(
            self.model, quantize_list, dtype=torch.qint8
        )

        # Save the quantized model
        self.save_quantized_model(output_path)

        return self.quantized_model

    def quantize_qat(self, output_path: Optional[str]=None):
        """Quantize the model using quantization-aware training and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        # Prepare the model for quantization-aware training
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("x86")
        torch.quantization.prepare_qat(self.model, inplace=True)

        # Train the model
        self.model.train()

        # Run the model on a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        self.model(dummy_input)

        # Convert the model to a quantized model
        torch.quantization.convert(self.model, inplace=True)

        # Save the quantized model
        self.save_quantized_model(output_path)

        return self.model

    def quantize_fx(self, output_path: Optional[str]=None, input_dim: tuple=(1, 3, 224, 224)):
        """Quantize the model using FX graph mode quantization and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """
        import torch.ao.quantization.quantize_fx as quantize_fx

        # Quantize the model
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.eval()

        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)

        example_inputs = (torch.randn(input_dim))

        model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        # quantize
        self.quantized_model = quantize_fx.convert_fx(model_prepared)

        # fuse modules
        self.quantized_model = quantize_fx.fuse_fx(self.quantized_model)

        # Save the quantized model
        self.save_quantized_model(output_path)

        return self.quantized_model

    def save_quantized_model(self, output_path: Optional[str]):
        """Save the quantized model to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        if output_path is not None:
            torch.jit.save(self.quantized_model, output_path)
            print(f"Quantized model saved to {output_path}")
            return self.quantized_model

    def get_quantized_model(self):
        """Get the quantized model.

        Returns:
            The quantized model.
        """
        return self.quantized_model

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
