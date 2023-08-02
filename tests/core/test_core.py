import unittest
from deepcrunch.core.model import ModelWrapper
from deepcrunch.core.trainer import TrainerWrapper


class TestCore(unittest.TestCase):
    def test_model_wrapper(self):
        # Test ModelWrapper class
        model = ModelWrapper()
        self.assertIsInstance(model, ModelWrapper)

    def test_trainer_wrapper(self):
        # Test TrainerWrapper class
        trainer = TrainerWrapper()
        self.assertIsInstance(trainer, TrainerWrapper)


if __name__ == "__main__":
    unittest.main()
