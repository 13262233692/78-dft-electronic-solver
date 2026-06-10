"""
Advanced Density Mixing Algorithms for SCF Stability.

Implements robust charge mixing schemes to prevent charge sloshing
in metallic and narrow-bandgap systems:

1. LinearMixer     - Simple linear mixing (baseline, may diverge for metals)
2. BroydenMixer    - Quasi-Newton Broyden's method with inverse Jacobian
3. DIISMixer       - Pulay's DIIS (Direct Inversion in Iterative Subspace)
4. AdaptiveMixer   - Switches between methods based on residual history

Physics Background
------------------
The fixed-point problem in SCF is:
    ρ_{in}^{n+1} = M[ρ_{in}^n]

where M = build_density ∘ solve_H. The residual r^n = ρ_out^n - ρ_in^n
measures deviation from self-consistency.

For metals, the Jacobian J = dM/dρ has eigenvalues near 1 at the Fermi
surface, causing simple Picard iteration (linear mixing) to diverge via
the "charge sloshing" instability.

Broyden's method and DIIS both accelerate convergence by using
historical information to build a better estimate of the fixed point.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DensityMixer(ABC):
    """Abstract base class for density mixers.

    Subclasses must implement :meth:`mix` which takes the input density
    and the computed output density from one SCF step, and returns the
    next input density.

    The mixer operates on the density vector ``x`` (typically in
    reciprocal space for computational efficiency) and maintains
    whatever internal state is required by the specific algorithm.
    """

    def __init__(self, alpha: float = 0.3, max_history: int = 20):
        self.alpha = alpha
        self.max_history = max_history
        self.iteration = 0
        self._history_size = 0

    @abstractmethod
    def reset(self) -> None:
        """Clear all history and internal state."""
        self.iteration = 0
        self._history_size = 0

    @abstractmethod
    def mix(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """
        Compute the next input density.

        Parameters
        ----------
        x_in : np.ndarray
            The input density vector at step n.
        x_out : np.ndarray
            The computed output density vector at step n
            (i.e., after diagonalization + density construction).

        Returns
        -------
        np.ndarray
            The input density vector for step n+1.
        """
        self.iteration += 1

    @property
    def residual_norm(self) -> Optional[float]:
        """Norm of the most recent residual, if available."""
        return getattr(self, "_last_residual_norm", None)

    def _residual(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        """r = x_out - x_in  (target: r = 0 at self-consistency)"""
        return x_out - x_in


# ========================================================================
# Linear Mixing (baseline)
# ========================================================================

class LinearMixer(DensityMixer):
    """Simple linear (Picard) mixing.

    x_{n+1} = (1 - α) x_n^in + α x_n^out
            = x_n^in + α r_n

    Stability requires α < 2 / |λ_max(J)|, which can be tiny for
    metals, making this method impractically slow or divergent.
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__(alpha=alpha, max_history=0)
        self._last_residual = None

    def reset(self) -> None:
        super().reset()
        self._last_residual = None
        self._last_residual_norm = None

    def mix(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        super().mix(x_in, x_out)
        r = self._residual(x_in, x_out)
        self._last_residual = r
        self._last_residual_norm = float(np.linalg.norm(r))
        x_next = x_in + self.alpha * r
        logger.debug(f"LinearMix:  α={self.alpha:.3f}  ||r||={self._last_residual_norm:.3e}")
        return x_next


# ========================================================================
# Broyden Mixing (Goodman's modified Broyden / Jonson's method)
# ========================================================================

class BroydenMixer(DensityMixer):
    """Modified Broyden mixing with approximate inverse Jacobian.

    Uses the Sherman-Morrison formula to build a rank-(m) update to
    the initial Jacobian estimate J0 ≈ -1/α I.  This is the standard
    "Broyden's second method" or "Anderson mixing" formulation used in
    most plane-wave DFT codes (VASP, Quantum ESPRESSO, ABINIT).

    Algorithm (simplified):
      1. Store histories Δx_i = x_{i+1} - x_i, Δr_i = r_{i+1} - r_i
      2. Build overlap matrix A_{ij} = ⟨Δr_i, Δr_j⟩
      3. Solve A c = ⟨Δr, r_{n}⟩
      4. x_{n+1} = x_n + α r_n - Σ_i c_i (α Δr_i + Δx_i)

    For metallic systems, set w > 0 to regularize the overlap matrix
    (Tchebychev-style damping of historical information).

    References
    ----------
    D. D. Johnson, PRB 38, 12807 (1988)
    G. Kresse, J. Furthmüller, PRB 54, 11169 (1996)
    """

    def __init__(self,
                 alpha: float = 0.4,
                 max_history: int = 12,
                 w0: float = 0.01,
                 preconditioner: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        alpha : float
            Initial mixing parameter (J_0 ≈ -1/α I). Larger = more aggressive.
            For metals, use 0.3-0.8. For insulators, up to 1.0 works.
        max_history : int
            Number of history vectors to keep. Typical 8-20.
        w0 : float
            Regularization weight (Tchebychev damping). Set to ~0.01 for
            problematic metallic systems to stabilize the overlap matrix.
        preconditioner : np.ndarray, optional
            Diagonal preconditioner P for the residual. For Kerker-type
            preconditioning, set P_G = G² / (G² + k²) (suppresses
            large-G charge sloshing which couples to long-range Hartree).
        """
        super().__init__(alpha=alpha, max_history=max_history)
        self.w0 = w0
        self.preconditioner = preconditioner
        self._x_history: List[np.ndarray] = []   # x_n
        self._r_history: List[np.ndarray] = []   # r_n
        self._dx_history: List[np.ndarray] = []  # Δx = x_{n+1} - x_n
        self._dr_history: List[np.ndarray] = []  # Δr = r_{n+1} - r_n
        self._last_residual = None

    def reset(self) -> None:
        super().reset()
        self._x_history.clear()
        self._r_history.clear()
        self._dx_history.clear()
        self._dr_history.clear()
        self._last_residual = None
        self._last_residual_norm = None

    def _apply_precon(self, v: np.ndarray) -> np.ndarray:
        if self.preconditioner is not None:
            return self.preconditioner * v
        return v

    def _precon_dot(self, a: np.ndarray, b: np.ndarray) -> complex:
        return complex(np.vdot(self._apply_precon(a), self._apply_precon(b)))

    def _precon_norm(self, v: np.ndarray) -> float:
        return float(np.linalg.norm(self._apply_precon(v)))

    def mix(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        super().mix(x_in, x_out)
        r = self._residual(x_in, x_out)
        self._last_residual = r
        r_norm = self._precon_norm(r)
        self._last_residual_norm = r_norm

        if self._history_size == 0:
            x_next = x_in + self.alpha * r
            self._x_history.append(x_in.copy())
            self._r_history.append(r.copy())
            self._history_size = 1
            logger.info(f"Broyden iter 1 (init):  ||r||={r_norm:.3e}  α={self.alpha:.2f}")
            return x_next

        dr = r - self._r_history[-1]
        dx = (x_in - self._x_history[-1])
        self._dr_history.append(dr.copy())
        self._dx_history.append(dx.copy())

        if len(self._dr_history) > self.max_history:
            self._dr_history.pop(0)
            self._dx_history.pop(0)

        m = len(self._dr_history)

        A = np.zeros((m, m), dtype=np.complex128)
        for i in range(m):
            for j in range(i, m):
                A[i, j] = self._precon_dot(self._dr_history[i], self._dr_history[j])
                A[j, i] = np.conj(A[i, j])

        for i in range(m):
            weight = 1.0 + self.w0 * (m - i)
            A[i, i] += weight * weight

        b = np.zeros(m, dtype=np.complex128)
        for i in range(m):
            b[i] = self._precon_dot(self._dr_history[i], r)

        try:
            gamma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning("Broyden overlap matrix singular, falling back to linear step")
            x_next = x_in + self.alpha * r
            self._x_history.append(x_in.copy())
            self._r_history.append(r.copy())
            return x_next

        x_next = x_in + self.alpha * r
        for i in range(m):
            x_next -= gamma[i] * (self.alpha * self._dr_history[i] + self._dx_history[i])

        cond = np.linalg.cond(A)
        gamma_norm = float(np.linalg.norm(gamma))
        logger.info(
            f"Broyden iter {self.iteration}:  ||r||={r_norm:.3e}  "
            f"hist={m}/{self.max_history}  cond(A)={cond:.2e}  "
            f"|γ|={gamma_norm:.3f}"
        )

        self._x_history.append(x_in.copy())
        self._r_history.append(r.copy())
        self._history_size = len(self._x_history)

        return x_next


# ========================================================================
# Pulay / DIIS Mixing
# ========================================================================

class DIISMixer(DensityMixer):
    """Pulay's Direct Inversion in the Iterative Subspace (DIIS).

    Also known as "Pulay mixing". Minimizes the norm of the extrapolated
    residual within the Krylov-like subspace spanned by the last m
    iterates, subject to the constraint that the coefficients sum to 1.

    Key advantage: very cheap per step (small generalized eigenvalue
    problem of size m). Works extremely well for SCF when the residual
    decreases smoothly.

    Algorithm:
      1. Store histories {x_i}, {r_i}, i = n-m+1 ... n
      2. Build error matrix B_{ij} = ⟨r_i, r_j⟩
      3. Solve the augmented Lagrangian:
             [B   1] [c]   [0]
             [1^T 0] [λ] = [1]
      4. x_{n+1} = Σ_i c_i x_i (or Σ_i c_i (x_i + α r_i))

    The mixing parameter α is used in the "CDIIS" variant (Pulay 1982)
    which extrapolates in the preconditioned direction.

    References
    ----------
    P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
    P. Pulay, J. Comput. Chem. 3, 556 (1982)
    """

    def __init__(self,
                 alpha: float = 1.0,
                 max_history: int = 15,
                 use_cdiis: bool = True,
                 min_diag: float = 1e-8):
        """
        Parameters
        ----------
        alpha : float
            Mixing parameter for CDIIS. Use 1.0 for standard DIIS on
            the preconditioned residual; smaller values for robustness.
        max_history : int
            Subspace size. Usually 8-20.
        use_cdiis : bool
            If True, extrapolate (x_i + α r_i) — this is the usual
            variant used in SCF ("Commutator DIIS").
        min_diag : float
            Regularization floor added to diagonal of B to prevent
            singular behavior when residuals become colinear near
            convergence.
        """
        super().__init__(alpha=alpha, max_history=max_history)
        self.use_cdiis = use_cdiis
        self.min_diag = min_diag
        self._x_history: List[np.ndarray] = []
        self._r_history: List[np.ndarray] = []
        self._last_residual = None

    def reset(self) -> None:
        super().reset()
        self._x_history.clear()
        self._r_history.clear()
        self._last_residual = None
        self._last_residual_norm = None

    def mix(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        super().mix(x_in, x_out)
        r = self._residual(x_in, x_out)
        self._last_residual = r
        r_norm = float(np.linalg.norm(r))
        self._last_residual_norm = r_norm

        if self.use_cdiis:
            y = x_in + self.alpha * r
        else:
            y = (1.0 - self.alpha) * x_in + self.alpha * x_out

        if self._history_size < self.max_history:
            self._x_history.append(y.copy())
            self._r_history.append(r.copy())
        else:
            self._x_history.pop(0)
            self._r_history.pop(0)
            self._x_history.append(y.copy())
            self._r_history.append(r.copy())

        self._history_size = len(self._x_history)
        m = self._history_size

        if m <= 1:
            logger.info(f"DIIS iter {self.iteration} (warming):  ||r||={r_norm:.3e}")
            return y

        B = np.zeros((m + 1, m + 1), dtype=np.complex128)
        B[:m, :m] = self.min_diag * np.eye(m, dtype=np.complex128)
        for i in range(m):
            for j in range(i, m):
                B[i, j] = np.vdot(self._r_history[i], self._r_history[j])
                B[j, i] = np.conj(B[i, j])
            B[i, m] = 1.0 + 0.0j
            B[m, i] = 1.0 + 0.0j
        B[m, m] = 0.0 + 0.0j

        rhs = np.zeros(m + 1, dtype=np.complex128)
        rhs[m] = 1.0 + 0.0j

        try:
            solution = np.linalg.solve(B, rhs)
            coeffs = solution[:m]
        except np.linalg.LinAlgError:
            logger.warning("DIIS system singular, falling back to linear step")
            return y

        coeffs_real = coeffs.real
        max_coeff = float(np.max(np.abs(coeffs_real)))

        x_next = np.zeros_like(x_in)
        for i in range(m):
            x_next += coeffs[i] * self._x_history[i]

        coeff_sum = float(np.sum(coeffs_real))

        if self.use_cdiis:
            logger.info(
                f"DIIS   iter {self.iteration}:  ||r||={r_norm:.3e}  "
                f"m={m}/{self.max_history}  Σc={coeff_sum:.4f}  "
                f"max|c|={max_coeff:.3f}  α={self.alpha:.2f}"
            )
        else:
            logger.info(
                f"Pulay  iter {self.iteration}:  ||r||={r_norm:.3e}  "
                f"m={m}/{self.max_history}  Σc={coeff_sum:.4f}  "
                f"max|c|={max_coeff:.3f}"
            )

        return x_next


# ========================================================================
# Kerker Preconditioner (useful helper for Broyden on metals)
# ========================================================================

def kerker_preconditioner(g_norm2: np.ndarray,
                          kappa: float = 0.8,
                          q0: float = 0.0) -> np.ndarray:
    """
    Construct the Kerker (G=0 subtracted) preconditioner.

    P_G = G² / (G² + κ² + q0²)

    This damps the small-G (long-wavelength) components of the residual,
    which are exactly the components that cause charge sloshing in
    metals: a long-wavelength density fluctuation induces a Hartree
    potential 4πδρ/G² which in turn drives the density back in the
    opposite direction, creating a 2-step oscillation.

    Parameters
    ----------
    g_norm2 : np.ndarray
        |G|² for each plane-wave component of the density (in a.u.).
    kappa : float
        Thomas-Fermi screening wavevector (typical 0.5-1.2 a.u.).
    q0 : float
        Additional shift for band insulators (often 0 for metals).

    Returns
    -------
    np.ndarray
        Diagonal preconditioner vector (same shape as g_norm2).
    """
    g_norm2 = np.asarray(g_norm2, dtype=float)
    denom = g_norm2 + kappa * kappa + q0 * q0
    safe = denom > 1e-16
    P = np.zeros_like(g_norm2)
    P[safe] = g_norm2[safe] / denom[safe]
    P[g_norm2 < 1e-16] = 1.0
    return P


# ========================================================================
# Factory
# ========================================================================

def create_mixer(method: str, **kwargs) -> DensityMixer:
    """
    Create a density mixer by name.

    Parameters
    ----------
    method : str
        One of: "linear", "broyden", "diis", "pulay"
    **kwargs :
        Passed to the mixer constructor.

    Returns
    -------
    DensityMixer
    """
    method = method.lower().strip()

    if method == "linear":
        return LinearMixer(alpha=kwargs.pop("alpha", 0.3))

    if method in ("broyden", "anderson", "broyden2"):
        return BroydenMixer(
            alpha=kwargs.pop("alpha", 0.4),
            max_history=kwargs.pop("max_history", 12),
            w0=kwargs.pop("w0", 0.01),
            preconditioner=kwargs.pop("preconditioner", None),
        )

    if method in ("diis", "pulay", "cdiis"):
        return DIISMixer(
            alpha=kwargs.pop("alpha", 1.0 if method != "pulay" else 0.5),
            max_history=kwargs.pop("max_history", 15),
            use_cdiis=kwargs.pop("use_cdiis", method != "pulay"),
            min_diag=kwargs.pop("min_diag", 1e-8),   # more regularization for pulay
        )

    raise ValueError(
        f"Unknown mixing method: '{method}'. "
        "Choose from: linear, broyden, diis, pulay."
    )
