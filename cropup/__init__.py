"""CropUp — crop understanding grounded in Earth observation.

Importing this package must stay cheap: no ee, no onnxruntime, no numpy.
Entrypoints import ``cropup.bootstrap`` first, then whatever they need.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
