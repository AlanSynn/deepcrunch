from enum import IntEnum


class BACKEND_TYPES(IntEnum):
    NEURAL_COMPRESSOR = 0
    TORCH = 1
    ONNX = 2
    DISABLED = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str):
        return BACKEND_TYPES[string.upper()]
