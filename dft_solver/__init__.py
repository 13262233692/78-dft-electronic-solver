"""DFT Electronic Solver - Python package"""

try:
    import _dft_core as core
    _USING_CPP_EXT = True
except ImportError:
    try:
        from . import _dft_core_fallback as core
        _USING_CPP_EXT = False
    except ImportError:
        raise ImportError(
            "Neither the compiled C++ extension (_dft_core) nor the "
            "pure Python fallback (_dft_core_fallback) is available. "
            "Please build the extension or ensure the fallback module is present."
        )

from .scf import SCFSolver, SCFParams, SCFResult
from .hamiltonian_wrapper import HamiltonianWrapper
from .kpoints import KPoints
from .mixing import (
    DensityMixer, LinearMixer, BroydenMixer, DIISMixer,
    kerker_preconditioner, create_mixer,
)
from .dos import DOSCalculator, DOSResult, compute_dos, plot_dos

__version__ = "0.1.0"
__all__ = [
    "SCFSolver", "SCFParams", "SCFResult",
    "HamiltonianWrapper", "KPoints",
    "DensityMixer", "LinearMixer", "BroydenMixer", "DIISMixer",
    "kerker_preconditioner", "create_mixer",
    "DOSCalculator", "DOSResult", "compute_dos", "plot_dos",
    "core", "_USING_CPP_EXT",
]
