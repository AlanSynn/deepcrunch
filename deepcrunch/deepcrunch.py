from typing import Optional

from deepcrunch.backend.backend_registry import BackendRegistry


class _Config:
    def __init__(self):
        self.framework = "pytorch"
        self.mode = "inference"
        self.backend = "torch"

    def __repr__(self):
        return f"Config(framework={self.framework}, mode={self.mode}, backend={self.backend})"

    def __str__(self):
        return f"Config(framework={self.framework}, mode={self.mode}, backend={self.backend})"

    def __setattr__(self, name, value):
        if name == "framework":
            if value not in ["pytorch", "torch", "tensorflow", "tf", "onnx"]:
                raise ValueError(f"Invalid framework: {value}")
        elif name == "mode":
            if value not in ["inference", "training"]:
                raise ValueError(f"Invalid mode: {value}")
        elif name == "backend":
            if value not in ["torch", "onnx", "neural_compressor"]:
                raise ValueError(f"Invalid backend: {value}")
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if name == "framework":
            return self.framework
        elif name == "mode":
            return self.mode
        elif name == "backend":
            return self.backend
        else:
            raise AttributeError(f"Invalid attribute: {name}")


def config(
    framework: Optional[str] = None,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
):
    """
    Configures the framework and mode for the DeepCrunch library.

    Parameters
    ----------
    framework : str, optional
        The framework to use for the library. The default is "pytorch".
    mode : str, optional
        The mode to use for the library. The default is "inference".

    Returns
    -------
    None.

    """

    # Check model file suffix
    if isinstance(framework, str):
        if framework.endswith(".onnx") or framework == "onnx":
            framework = "onnx"
        elif (
            framework.endswith(".pt")
            or framework.endswith(".pth")
            or framework == "pytorch"
            or framework == "torch"
        ):
            framework = "pytorch"
        elif (
            framework.endswith(".pb") or framework == "tensorflow" or framework == "tf"
        ):
            framework = "tensorflow"
        else:
            raise ValueError(f"Invalid model file: {framework}")

    # Check backend
    if backend is None:
        if framework == "pytorch":
            backend = "torch"
        elif framework == "tensorflow":
            backend = "tensorflow"
        elif framework == "onnx":
            backend = "onnx"
        else:
            raise ValueError(f"Invalid framework: {framework}")

    # Check mode
    if mode is None:
        mode = "inference"

    global _CONFIG
    _CONFIG = _Config()
    _CONFIG.backend = backend
    _CONFIG.framework = framework
    _CONFIG.mode = mode


def quantize(model, *args, **kwargs):
    """
    Quantizes the given model using the specified backend.

    Parameters
    ----------
    model : torch.nn.Module
        The model to quantize.
    backend : str
        The backend to use for quantization.
    *args
        Any additional arguments to pass to the backend.
    **kwargs
        Any additional keyword arguments to pass to the backend.

    Returns
    -------
    torch.nn.Module
        The quantized model.

    """

    if "_CONFIG" not in globals():
        config(framework=model)

    registered_backend = BackendRegistry.get_backend(_CONFIG.backend)
    backend_instance = registered_backend

    if _CONFIG.framework == "pytorch" or _CONFIG.framework == "torch":
        # list all args
        # print(args)
        # print(kwargs)
        # raise ValueError()
        return backend_instance.quantize(model, *args, **kwargs)
    elif _CONFIG.framework == "tensorflow" or _CONFIG.framework == "tf":
        raise NotImplementedError("TensorFlow backend not implemented yet")
    elif _CONFIG.framework == "onnx":
        return backend_instance.quantize(model, *args, **kwargs)
    else:
        raise ValueError(f"Not supported framework: {_CONFIG.framework}")


def save(quantized_model, output_path: Optional[str] = None):
    backend_instance = BackendRegistry.get_backend(_CONFIG.backend)
    return backend_instance.save_quantized_model(quantized_model, output_path)


def clear_globals():
    """
    Clears the global variables used by the library.

    Returns
    -------
    None.

    """
    global _CONFIG
    global _BACKEND
    _CONFIG = None
    _BACKEND = None
