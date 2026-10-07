from .base import ResearchTask
from .classification import ClassificationTask
from .generation import AutoencoderTask, DiffusionTask

__all__ = ["ResearchTask", "ClassificationTask", "AutoencoderTask", "DiffusionTask"]
