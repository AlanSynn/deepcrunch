from abc import ABC, abstractmethod


class PostTrainingQuantizationBaseWrapper(ABC):
    def __init__(self, model, config_path) -> None:
        super().__init__()
