"""compression-lm: general-purpose compressors as language models."""

from . import backends
from .engine import GenerationConfig, GenerationResult, generate

__version__ = "0.1.0"

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "__version__",
    "backends",
    "generate",
]
