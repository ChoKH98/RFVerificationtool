"""RFVerificationtool electromagnetic solver package."""

from .fdtd_core import FDTD
from .geometry import patch_dimensions, create_patch_geometry
from .benchmark_cases import BENCHMARK_CASES

__all__ = ["FDTD", "patch_dimensions", "create_patch_geometry", "BENCHMARK_CASES"]
