import itertools
import random
import string
import tempfile
from pathlib import Path
from copy import copy
from enum import IntEnum
from pathlib import Path
from typing import Optional

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_dynamic,
    quantize_static,
)

from deepcrunch import logger
from deepcrunch.backend.engines.base_backend import (
    PostTrainingQuantizationBaseWrapper as PTQBase,
)
from deepcrunch.utils.os_utils import LazyImport
from deepcrunch.utils.time import log_elapsed_time

# onnxruntime = LazyImport("onnxruntime")
float16 = LazyImport("onnxconverter_common.float16")
# quantize_dynamic = LazyImport("onnxruntime.quantization.quantize_dynamic")
# quantize_static = LazyImport("onnxruntime.quantization.quantize_static")
# quantize_qat = LazyImport("onnxruntime.quantization.quantize_qat")
# QuantType = LazyImport("onnxruntime.quantization.QuantType")


class ONNX_PTQ_TYPE(IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    FLOAT16 = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str):
        return ONNX_PTQ_TYPE[string.upper()]


class ONNXPTQ(PTQBase):
    def __init__(
        self,
        model=None,
        type: int = ONNX_PTQ_TYPE.DYNAMIC,
        config_path: Optional[str] = None,
    ):
        """
        Args:
            model: A trained PyTorch or TensorFlow model. Default is None.
            type: The type of quantization to perform. Default is ONNX_PTQ_TYPE.DYNAMIC.
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
        import onnx
        from onnxruntime.quantization import (
            QuantType,
            quantize_dynamic,
            quantize_static,
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        type_str = kwargs.get("type", "dynamic")
        type = ONNX_PTQ_TYPE.from_str(type_str)

        rtn = None

        if type == ONNX_PTQ_TYPE.DYNAMIC:
            rtn = self.onnx_quantize_dynamic(*args, **kwargs)
        elif type == ONNX_PTQ_TYPE.STATIC:
            rtn = self.onnx_quantize_static(*args, **kwargs)
        elif type == ONNX_PTQ_TYPE.FLOAT16:
            rtn = self.onnx_quantize_float16(*args, **kwargs)
        elif type == ONNX_PTQ_TYPE.QAT:
            rtn = self.onnx_quantize_qat(*args, **kwargs)
        else:
            raise ValueError(
                f"Invalid quantization type: {type}. Please choose from: ONNX_PTQ_TYPE.DYNAMIC, ONNX_PTQ_TYPE.STATIC, ONNX_PTQ_TYPE.QAT, ONNX_PTQ_TYPE.FX"
            )

        return rtn

    def onnx_quantize_dynamic(
        self, model, output_path: Optional[str] = None, *args, **kwargs
    ):
        """Quantize the model using dynamic quantization and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        qconfig_spec = kwargs.get("qconfig_spec", None)
        dtype = kwargs.get("dtype", "bfloat16")

        if output_path is None:
            random_string = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=8)
            )
            output_path = random_string + "quantized.onnx"

        with tempfile.TemporaryDirectory(prefix="pre.quant.") as quant_tmp_dir:
            temp_path = Path(quant_tmp_dir)

            # if model size is greater than 2GB, skip model optimization
            skip_optimization = False
            if Path(model).stat().st_size > 2**31 - 1:
                logger.warning(
                    "Model size is greater than 2GB, skipping model optimization."
                )
                skip_optimization = True

            quant_pre_process(
                input_model_path = model,
                output_model_path = temp_path / "pre.quant.onnx",
                skip_optimization = skip_optimization,
                skip_onnx_shape = False,
                skip_symbolic_shape = False,
                auto_merge = False,
                int_max = 2**31 - 1,
                guess_output_rank = False,
                verbose = 0,
                save_as_external_data = False,
                all_tensors_to_one_file = False,
                external_data_location = None,
                external_data_size_threshold = 1024,
            )

            # Quantize the model
            quantize_dynamic(
                model_input=temp_path / "pre.quant.onnx",
                model_output=output_path,
                op_types_to_quantize=None,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QInt8,
                nodes_to_quantize=None,
                nodes_to_exclude=None,
                use_external_data_format=False,
                extra_options=None,
            )

        # sess_options = ort.SessionOptions()

        # # Set graph optimization level
        # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # Save the quantized model
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)

        return quantized_model

    def onnx_quantize_float16(
        self, model, output_path: Optional[str] = None, *args, **kwargs
    ):
        model_fp32 = onnx.load(model)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        if output_path is not None:
            onnx.save(model_fp16, output_path)
            print(f"Quantized model saved to {output_path}")
        return model_fp16

    def onnx_quantize_static(
        self,
        model,
        output_path: Optional[str] = None,
        calibration_dataset: Optional[str] = None,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        *args,
        **kwargs,
    ):
        if calibration_dataset is None:
            raise ValueError("Calibration dataset is required for static quantization.")

        qconfig_spec = kwargs.get("qconfig_spec", None)
        dtype = kwargs.get("dtype", QuantType.QInt8)

        if output_path is None:
            random_string = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=8)
            )
            output_path = random_string + "quantized.onnx"

        quantize_static(
            model,
            output_path,
            calibration_dataset,
            quant_format=quant_format,
            per_channel=per_channel,
            weight_type=dtype,
        )

        # Save the quantized model
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)

        return quantized_model

    def onnx_quantize_qat(
        self,
        model,
        output_path: Optional[str] = None,
        input_dim: tuple = (1, 3, 224, 224),
    ):
        pass

    @staticmethod
    def save_quantized_model(quantized_model, output_path: Optional[str]):
        """Save the quantized model to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """

        if output_path is not None:
            onnx.save(quantized_model, output_path)
            print(f"Quantized model saved to {output_path}")

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
        pass
