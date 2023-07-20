from deepcrunch.backend.types import BACKEND_TYPES
from deepcrunch.utils.os_utils import LazyImport

# Lazy importing all the dnn compression backends
neural_compressor = LazyImport("neural_compressor")
torch = LazyImport("torch")
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
        return cls._backends[backend_name](*args, **kwargs)

    @classmethod
    def get_backend_class(cls, backend_name):
        """
        Returns the class of the specified backend.

        :param backend_name: The name of the backend to retrieve.
        :return: The class of the specified backend.
        """
        return cls._backends[backend_name]


BackendRegistry.register(BACKEND_TYPES.NEURAL_COMPRESSOR, neural_compressor)
BackendRegistry.register(BACKEND_TYPES.TORCH, torch)
# BackendRegistry.register(BACKEND_TYPES.QD, qd)
# BackendRegistry.register(BACKEND_TYPES.DEEPCRUNCH, deepcrunch)
