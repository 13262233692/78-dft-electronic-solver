"""K-point generation utilities."""

import numpy as np
from typing import List, Tuple


Vector3 = np.ndarray


class KPoints:
    """Manages k-point sampling of the Brillouin zone."""

    def __init__(self, kpoints: np.ndarray, weights: np.ndarray):
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (N, 3)")
        if weights.shape[0] != kpoints.shape[0]:
            raise ValueError("weights and kpoints must have same length")
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
        self._kpoints = np.asarray(kpoints, dtype=float)
        self._weights = np.asarray(weights, dtype=float)

    @property
    def nk(self) -> int:
        return self._kpoints.shape[0]

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def kpoint(self, ik: int) -> np.ndarray:
        return self._kpoints[ik]

    def weight(self, ik: int) -> float:
        return self._weights[ik]

    @classmethod
    def gamma(cls) -> "KPoints":
        return cls(np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))

    @classmethod
    def monkhorst_pack(cls, grid: Tuple[int, int, int],
                        cell: np.ndarray = None,
                        shift: Tuple[float, float, float] = (0, 0, 0)) -> "KPoints":
        ng = np.array(grid, dtype=int)
        shift = np.array(shift, dtype=float)
        kpts = []
        for i in range(ng[0]):
            for j in range(ng[1]):
                for k in range(ng[2]):
                    frac = (np.array([i, j, k], dtype=float) + 0.5 + shift) / ng
                    kpts.append(frac - np.round(frac))

        kpoints = np.array(kpts)
        weights = np.ones(len(kpoints)) / len(kpoints)
        return cls(kpoints, weights)

    def to_cartesian(self, reciprocal_cell: np.ndarray) -> np.ndarray:
        return self._kpoints @ reciprocal_cell.T
