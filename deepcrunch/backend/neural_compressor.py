import os

from neural_compressor.experimental import Quantization, common


class PostTrainingQuantizationWrapper:
    """A wrapper for post-training quantization with Intel Neural Compressor."""

    def __init__(self, model, config_path):
        """
        Args:
            model: A trained PyTorch or TensorFlow model.
            config_path: Path to the YAML configuration file for quantization.
        """
        self.model = model
        self.config_path = config_path

    @log_elapsed_time
    def quantize(self, output_path):
        """Quantize the model and save it to the specified path.

        Args:
            output_path: Path to save the quantized model.
        """
        # Create a quantizer
        quantizer = Quantization(self.config_path)

        # Set the model to be quantized
        quantizer.model = common.Model(self.model)

        # Perform quantization
        q_model = quantizer()

        # Save the quantized model
        q_model.save(output_path)
        print(f"Quantized model saved to {output_path}")

        return q_model
