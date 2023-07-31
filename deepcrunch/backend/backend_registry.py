from deepcrunch.backend.types import BACKEND_TYPES
from deepcrunch.utils.os_utils import LazyImport

# TODO: Lazy import backends
from deepcrunch.backend.engines.neural_compressor import (
    NeuralCompressorPTQ as NeuralCompressor,
)
from deepcrunch.backend.engines.torch_ao import TorchPTQ as Torch
from deepcrunch.backend.engines.onnx import ONNXPTQ as onnx
# qd = LazyImport('qd')
# deepcrunch = LazyImport('deepcrunch.compressor')


class BackendRegistry:
    """
    Registry for neural compression backends
    Stores the mapping between backend and registry
    """

    _backends = {}

    @classmethod
    def register(cls, backend_name, backend):
        """
        Registers a neural compression backend with the given quantizer.

        :param backend_name: The name of the backend to register.
        :param quantizer: The quantizer to use with the backend.
        """
        if backend_name not in cls._backends:
            cls._backends[backend_name] = backend
        else:
            raise ValueError(
                f"Neural compression backend: {backend_name} already registered"
            )

    @classmethod
    def get_backend(cls, backend_name, *args, **kwargs):
        """
        Returns an instance of the specified backend with the given arguments.

        :param backend_name: The name of the backend to retrieve.
        :param args: Any additional arguments to pass to the backend constructor.
        :param kwargs: Any additional keyword arguments to pass to the backend constructor.
        :return: An instance of the specified backend.
        """

        backend_type = BACKEND_TYPES.from_str(backend_name)

        return cls._backends[backend_type](*args, **kwargs)

    @classmethod
    def get_backend_class(cls, backend_name):
        """
        Returns the class of the specified backend.

        :param backend_name: The name of the backend to retrieve.
        :return: The class of the specified backend.
        """

        backend_type = BACKEND_TYPES.from_str(backend_name)

        return cls._backends[backend_type]


BackendRegistry.register(BACKEND_TYPES.NEURAL_COMPRESSOR, NeuralCompressor)
BackendRegistry.register(BACKEND_TYPES.TORCH, Torch)
BackendRegistry.register(BACKEND_TYPES.ONNX, onnx)
# BackendRegistry.register(BACKEND_TYPES.DEEPCRUNCH, deepcrunch)
