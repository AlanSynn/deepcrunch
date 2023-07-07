"""
DeepCrunch: An automated library for practical and efficient deep learning model compression.
This project is a part of the LG U+ Global Summer Internship.

Created by Alan Synn (alan@alansynn.com) MLOps team.
"""

from deepcrunch.core.trainer import TrainerWrapper
from deepcrunch.core.model import ModelWrapper

__all__ = ["TrainerWrapper", "ModelWrapper"]