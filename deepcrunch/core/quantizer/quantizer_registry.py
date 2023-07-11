from deepcrunch.core.types.quantize_strategy import QUANTIZATION_STRATEGY

class QuantizerRegistry:
    """
    Registry for quantizers
    Stores the mapping between system and quantizer
    """

    _quantizers = {}

    @classmethod
    def register(cls, system, quantizer):
        if system not in cls._quantizers:
            cls._quantizers[system] = quantizer
        else:
            raise ValueError(f"System {system} already registered")

    @classmethod
    def get_quantizer(cls, system, *args, **kwargs):
        return cls._quantizers[system](*args, **kwargs)

    @classmethod
    def get_quantizer_class(cls, system):
        return cls._quantizers[system]


QuantizerRegistry.register(QUANTIZATION_STRATEGY.UNIFORM, UniformQuantizer)
# QuantizerRegistry.register(QUANTIZATION_STRATEGY.QD, QDQuantizer)
# QuantizerRegistry.register(QUANTIZATION_STRATEGY.DEEPCRUNCH, deepcrunchQuantizer)
