from enum import IntEnum

class QUANTIZATION_STRATEGY(IntEnum):
    UNIFORM = 0
    DEEPCRUNCH = 1
    QD = 2
    DISABLED = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(string):
        return QUANTIZATION_STRATEGY[string.upper()]