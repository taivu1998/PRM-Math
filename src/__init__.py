"""
PRM-Math source package.
Contains core modules for Process Reward Model training and inference.
"""

from .utils import seed_everything, setup_logging, get_device
from .config_parser import ConfigParser
from .dataset import PRMDatasetBuilder

# PRMModelLoader requires unsloth which needs CUDA
# Import it conditionally to allow package to work on non-CUDA systems
try:
    from .model import PRMModelLoader, UNSLOTH_AVAILABLE
except ImportError:
    PRMModelLoader = None
    UNSLOTH_AVAILABLE = False

__all__ = [
    "seed_everything",
    "setup_logging",
    "get_device",
    "ConfigParser",
    "PRMDatasetBuilder",
    "PRMModelLoader",
    "UNSLOTH_AVAILABLE",
]
