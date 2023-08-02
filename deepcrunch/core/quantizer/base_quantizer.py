from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from deepcrunch.utils.os_utils import LazyImport

torch = LazyImport("torch")


@dataclass
class BaseDataclass:
    def to_dict(self, format_for_wandb=False, prefix=""):
        if prefix and not prefix.endswith("/"):
            prefix += "_"

        output = {}

        for key, value in self.__dict__.items():
            if isinstance(value, BaseDataclass):
                output.update(
                    value.to_dict(
                        format_for_wandb=format_for_wandb, prefix=f"{prefix}{key}"
                    )
                )
            elif isinstance(value, IntEnum):
                output[f"{prefix}{key}"] = (
                    int(value) if format_for_wandb else str(value)
                )
            elif isinstance(value, bool):
                output[f"{prefix}{key}"] = int(value) if format_for_wandb else value
            else:
                output[f"{prefix}{key}"] = value

        return output


@dataclass
class BaseQuantizationConfig(BaseDataclass):
    def __str__(self):
        return f"QuantizationConfig({self.__dict__})"


class BaseQuantizer(ABC):
    """
    Base class for all quantizers
    It has a single method in the public interface:
    1. quantize: This is the main method that is called to quantize the model
    During initialization, it expects to receive an instance of quantization config which
    is derived from BaseQuantizationConfig
    """

    def __init__(
        self,
        quantization_config: BaseQuantizationConfig,
        debug: bool = False,
    ) -> None:
        self._config = quantization_config
        self._debug = debug

    @abstractmethod
    def quantize(self, model: torch.nn.Module) -> None:
        pass

    @abstractmethod
    def quantize_optimizer(model, optimizer, in_place=False) -> None:
        pass
