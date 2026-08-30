from importlib.metadata import PackageNotFoundError, version

from .core import (
    deviation_step,
    forward,
    forward_points,
    forward_step,
    infer_step,
    inverse,
    inverse_points,
    inverse_step,
    simplify_points,
    simplify_step,
)

try:
    __version__ = version("xinterp")
except PackageNotFoundError:  # pragma: no cover
    # Imported from a source tree that was never installed; the compiled
    # extension is present but the distribution metadata is not.
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "deviation_step",
    "forward",
    "forward_points",
    "forward_step",
    "infer_step",
    "inverse",
    "inverse_points",
    "inverse_step",
    "simplify_points",
    "simplify_step",
]
