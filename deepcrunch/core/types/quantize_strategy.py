from enum import IntEnum


class QUANTIZATION_STRATEGY(IntEnum):
    UNIFORM = 0
    DEEPCRUNCH = 1

    QD = 2
    DISABLED = 3

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str):
        return QUANTIZATION_STRATEGY[string.upper()]
