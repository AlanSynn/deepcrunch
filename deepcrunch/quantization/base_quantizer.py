from abc import ABC, abstractmethod

from deepcrunch.utilities.dynamic_load import ModuleLoader
from deepcrunch.utilities.exceptions import InvalidQuantizationConfigException


class BaseQuantizer(ABC):
    @abstractmethod
    def quantization_algorithm(self):
        pass

    @abstractmethod
    def quantize(self):
        pass


class QuantizerFactory:
    quantizers = {}

    @staticmethod
    def get_quantizer_type(quantizer_type: str) -> str:
        if quantizer_type in QuantizerFactory.quantizers:
            return QuantizerFactory.quantizers[quantizer_type]
        else:
            raise InvalidQuantizationConfigException("Invalid quantizer type")

    @staticmethod
    def import_quantizer(quantizer_path: str) -> Any:
        # check if quantizer_path is path of file or name of module
        quantizer = quantizer_path.split(".")[-1]

        quantizer_module = ModuleLoader.import_module_by_name("deepcrunch.quantization." + quantizer)
        quantizer_class = getattr(quantizer_module, quantizer)
        return quantizer_class

    @staticmethod
    def get_quantizer_instance(quantizer_type: str) -> Any:
        quantizer = QuantizerFactory.get_quantizer_type(quantizer_type)
        quantizer_instance = QuantizerFactory.import_quantizer(quantizer)
        return quantizer_instance

    @staticmethod
    def register_quantizer(quantizer_type: str, quantizer: str) -> None:
        QuantizerFactory.quantizers[quantizer_type] = quantizer

    @staticmethod
    def quantize(model: Any, quantizer_type: str, qconfig_spec: Any, dtype: Any, inplace: bool) -> Any:
        quantizer = QuantizerFactory.get_quantizer_instance(quantizer_type)
        return
