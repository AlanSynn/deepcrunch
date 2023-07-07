class MisconfigurationException(Exception):
    """Exception used to inform users of the given configuration is invalid."""

class InvalidQuantizationConfigException(Exception):
    """Exception used to inform users of the given quantization config is invalid."""

class InvalidParallelModeException(Exception):
    """Exception used to inform users of the given parallel mode is invalid."""