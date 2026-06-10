"""Self-Consistent Field (SCF) solver for DFT.

Supports advanced density mixing algorithms for metallic/narrow-gap
systems to prevent charge-sloshing divergence:
  - Linear    (baseline)
  - Broyden   (Quasi-Newton, inverse-Jacobian estimation)
  - DIIS      (Pulay's Direct Inversion in Iterative Subspace)
  - Pulay     (original Pulay variant, less aggressive than DIIS)
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import logging

from . import core
from .hamiltonian_wrapper import HamiltonianWrapper
from .kpoints import KPoints
from .mixing import (
    DensityMixer, LinearMixer, BroydenMixer, DIISMixer,
    kerker_preconditioner, create_mixer,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SCFParams:
    """Parameters controlling the SCF iteration."""
    max_steps: int = 100
    energy_tol: float = 1e-6
    density_tol: float = 1e-6

    mixing_method: str = "broyden"
    """linear | broyden | diis | pulay"""

    mixing_alpha: float = 0.4
    """
    Initial mixing parameter.
      - Linear:   the full mixing coefficient (use small ~0.1 for metals!)
      - Broyden:  initial inverse-Jacobian scaling. 0.3-0.8 for metals
      - DIIS:     CDIIS extrapolation factor (typically 1.0)
      - Pulay:    pre-mixing coefficient (typically 0.5)
    """

    mixing_max_history: int = 12
    """Number of history vectors for Broyden / DIIS."""

    mixing_w0: float = 0.01
    """Tchebychev-style regularization for Broyden overlap matrix."""

    use_kerker_preconditioner: bool = True
    """
    Apply Kerker (G² / (G² + κ²)) preconditioner when using Broyden.
    *Essential* for metals to kill the long-range charge-sloshing mode.
    """

    kerker_kappa: float = 0.8
    """Thomas-Fermi screening wavevector in a.u. for Kerker preconditioner."""

    diagonalization_method: str = "eigsh"
    """eigsh (ARPACK) | lobpcg"""

    n_extra_bands: int = 2
    smearing: float = 0.01
    smearing_type: str = "fermi"
    verbose: bool = True


# ============================================================================
# Result container
# ============================================================================

@dataclass
class SCFResult:
    """Result of an SCF calculation."""
    converged: bool = False
    n_iter: int = 0
    total_energy: float = 0.0
    eigenvalues: Dict[int, np.ndarray] = field(default_factory=dict)
    eigenvectors: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    occupations: Dict[int, np.ndarray] = field(default_factory=dict)
    energies_per_iter: List[float] = field(default_factory=list)
    density_errors: List[float] = field(default_factory=list)
    residual_norms: List[float] = field(default_factory=list)
    density: Optional[core.Density] = None
    mixing_history: Dict[str, List[Any]] = field(default_factory=dict)


# ============================================================================
# Smearing helpers
# ============================================================================

def smearing_function(energies: np.ndarray, fermi_level: float,
                      sigma: float, kind: str = "fermi") -> np.ndarray:
    """Compute occupation numbers via smearing."""
    x = (energies - fermi_level) / sigma
    if kind == "fermi":
        occ = np.zeros_like(x, dtype=float)
        neg_mask = x < 0
        pos_mask = ~neg_mask
        occ[neg_mask] = 1.0 / (1.0 + np.exp(x[neg_mask]))
        exp_neg_x = np.exp(-np.clip(x[pos_mask], None, 500))
        occ[pos_mask] = exp_neg_x / (1.0 + exp_neg_x)
    elif kind == "gaussian":
        from scipy.special import erfc
        occ = 0.5 * erfc(x / np.sqrt(2))
    else:
        raise ValueError(f"Unknown smearing: {kind}")
    return occ


def find_fermi_level(energies: np.ndarray, n_electrons: float,
                     sigma: float, kind: str = "fermi") -> float:
    """Find Fermi level by bisection."""
    from scipy.optimize import bisect

    def f(mu):
        occ = smearing_function(energies, mu, sigma, kind)
        return 2.0 * occ.sum() - n_electrons

    e_min = energies.min() - 10.0
    e_max = energies.max() + 10.0
    if f(e_min) * f(e_max) > 0:
        return (e_min + e_max) / 2
    return bisect(f, e_min, e_max, xtol=1e-12)


# ============================================================================
# SCF Solver
# ============================================================================

class SCFSolver:
    """Self-Consistent Field solver for plane-wave DFT.

    The SCF loop is the standard fixed-point iteration:

        1. Given ρ_in, build H[ρ_in]
        2. Diagonalize H → {ψ_i, ε_i}
        3. Build ρ_out = Σ f_i |ψ_i|²
        4. Use mixer to produce ρ_{in,next} = Mix(ρ_in, ρ_out)
        5. Check convergence; repeat if not.

    For systems with very high density-of-states at the Fermi level
    (simple metals, near-gap states), the bare fixed-point map has
    eigenvalues near -1, causing the infamous "charge sloshing"
    oscillation when using naive linear mixing.

    The Broyden / DIIS mixers in this class solve this problem by
    using the last ~10 iterations to build an optimal estimate of the
    fixed point, dramatically improving both stability and convergence
    rate for metallic systems.
    """

    def __init__(self, basis: core.PlaneWaveBasis, atoms: core.Atoms,
                 kpoints: Optional[KPoints] = None,
                 params: Optional[SCFParams] = None):
        self._basis = basis
        self._atoms = atoms
        self._kpoints = kpoints or KPoints.gamma()
        self._params = params or SCFParams()

        self._hamiltonian_cpp = core.Hamiltonian(basis, atoms)
        self._hamiltonian = HamiltonianWrapper(self._hamiltonian_cpp)

        self._n_electrons = atoms.nelectrons()
        self._density = core.Density(basis)
        self._density.set_n_electrons(self._n_electrons)

        n_bands = max(int(np.ceil(self._n_electrons / 2)) + self._params.n_extra_bands, 4)
        self._n_bands = n_bands

        self._mixer: DensityMixer = self._build_mixer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def hamiltonian(self) -> HamiltonianWrapper:
        return self._hamiltonian

    @property
    def density(self) -> core.Density:
        return self._density

    @property
    def mixer(self) -> DensityMixer:
        return self._mixer

    @property
    def params(self) -> SCFParams:
        return self._params

    @params.setter
    def params(self, p: SCFParams):
        self._params = p
        self._mixer = self._build_mixer()

    def reset_mixer(self) -> None:
        """Reset the mixer state (e.g., when restarting SCF)."""
        self._mixer = self._build_mixer()

    # ------------------------------------------------------------------
    # Mixer construction
    # ------------------------------------------------------------------

    def _build_mixer(self) -> DensityMixer:
        p = self._params

        precond = None
        if (p.use_kerker_preconditioner
                and p.mixing_method.lower() in ("broyden", "anderson", "broyden2")):
            g_norm2 = np.array(
                [gv.norm2 for gv in self._basis.g_vectors], dtype=float
            )
            precond = kerker_preconditioner(g_norm2, kappa=p.kerker_kappa)
            if p.verbose:
                logger.info(
                    f"Kerker preconditioner enabled, κ={p.kerker_kappa:.2f} a.u.  "
                    f"range(P) = [{precond.min():.3f}, {precond.max():.3f}]"
                )

        mixer = create_mixer(
            p.mixing_method,
            alpha=p.mixing_alpha,
            max_history=p.mixing_max_history,
            w0=p.mixing_w0,
            preconditioner=precond,
        )
        if p.verbose:
            logger.info(
                f"Density mixer: {type(mixer).__name__}  "
                f"(α={p.mixing_alpha:.2f}, hist={p.mixing_max_history})"
            )
        return mixer

    # ------------------------------------------------------------------
    # Density helpers
    # ------------------------------------------------------------------

    def _density_to_vector(self, density: core.Density) -> np.ndarray:
        """Flatten density into a complex vector (reciprocal space)."""
        rho_g = np.asarray(density.rho_g)
        return rho_g.astype(np.complex128, copy=True)

    def _vector_to_density(self, vec: np.ndarray) -> core.Density:
        """Reconstruct a Density object from a reciprocal-space vector."""
        density = core.Density(self._basis)
        density.set_n_electrons(self._n_electrons)

        vec = np.asarray(vec, dtype=np.complex128)
        ngx, ngy, ngz = self._basis.ngx, self._basis.ngy, self._basis.ngz

        g_vecs = self._basis.g_vectors
        a1 = self._basis.cell[:, 0]
        a2 = self._basis.cell[:, 1]
        a3 = self._basis.cell[:, 2]
        nr = ngx * ngy * ngz

        rho_r = np.zeros(nr, dtype=float)
        for ix in range(ngx):
            for iy in range(ngy):
                for iz in range(ngz):
                    ir = ix * ngy * ngz + iy * ngz + iz
                    r = (ix / ngx) * a1 + (iy / ngy) * a2 + (iz / ngz) * a3
                    total = 0.0 + 0.0j
                    for ig, g in enumerate(g_vecs):
                        phase = float(np.dot(g.cartesian, r))
                        total += vec[ig] * complex(
                            float(np.cos(phase)), float(np.sin(phase))
                        )
                    rho_r[ir] = total.real

        density.set_from_r_space(rho_r)
        return density

    def _initial_density(self) -> core.Density:
        """Initialize density from superposition of atomic densities."""
        density = core.Density(self._basis)
        density.set_n_electrons(self._n_electrons)

        ngx = self._basis.ngx
        ngy = self._basis.ngy
        ngz = self._basis.ngz
        nr = ngx * ngy * ngz
        rho_r = np.zeros(nr)

        cell = np.asarray(self._basis.cell)
        a1, a2, a3 = cell[:, 0], cell[:, 1], cell[:, 2]

        for atom in self._atoms.atoms:
            for ix in range(ngx):
                for iy in range(ngy):
                    for iz in range(ngz):
                        ir = ix * ngy * ngz + iy * ngz + iz
                        r = (ix / ngx) * a1 + (iy / ngy) * a2 + (iz / ngz) * a3
                        dist = float(np.linalg.norm(r - np.array(atom.position)))
                        r_c = 1.5
                        Z = atom.atomic_number
                        prefactor = Z / ((np.sqrt(2 * np.pi) * r_c) ** 3)
                        rho_r[ir] += prefactor * np.exp(-dist ** 2 / (2 * r_c ** 2))

        density.set_from_r_space(rho_r)
        return density

    # ------------------------------------------------------------------
    # Band computation
    # ------------------------------------------------------------------

    def _compute_bands(self, density: core.Density
                      ) -> Tuple[Dict[int, np.ndarray], Dict[int, List[np.ndarray]],
                                 Dict[int, np.ndarray]]:
        """Compute eigenvalues and eigenvectors at all k-points."""
        self._hamiltonian.update_density(density)

        eigenvalues = {}
        eigenvectors = {}
        occupations = {}

        for ik in range(self._kpoints.nk):
            kpt = self._kpoints.kpoint(ik)

            if self._params.verbose:
                logger.info(f"  Diagonalizing at k-point {ik + 1}/{self._kpoints.nk}: {kpt}")

            if self._params.diagonalization_method == "lobpcg":
                precon = self._hamiltonian.kinetic_preconditioner(kpt)
                evals, evecs = self._hamiltonian.diagonalize(
                    kpt, self._n_bands, method="lobpcg",
                    preconditioner=precon,
                )
            else:
                evals, evecs = self._hamiltonian.diagonalize(
                    kpt, self._n_bands, method="eigsh",
                )

            eigenvalues[ik] = evals

            psi_list = [evecs[:, i].copy() for i in range(self._n_bands)]
            norm = np.sqrt(self._basis.cell_volume)
            for i in range(self._n_bands):
                psi_list[i] /= np.linalg.norm(psi_list[i]) * norm
            eigenvectors[ik] = psi_list

            occupations[ik] = np.zeros(self._n_bands)

        # --------------- occupations via smearing ----------------
        all_evals = np.concatenate([eigenvalues[ik] for ik in range(self._kpoints.nk)])
        all_weights = np.concatenate([
            np.full_like(eigenvalues[ik], self._kpoints.weight(ik))
            for ik in range(self._kpoints.nk)
        ])

        if self._params.smearing > 0 and self._n_electrons > 0:
            mu = find_fermi_level(all_evals, self._n_electrons,
                                   self._params.smearing, self._params.smearing_type)
            for ik in range(self._kpoints.nk):
                occ = smearing_function(eigenvalues[ik], mu,
                                         self._params.smearing,
                                         self._params.smearing_type)
                occupations[ik] = 2.0 * self._kpoints.weight(ik) * occ
        else:
            nocc = int(np.ceil(self._n_electrons / 2))
            for ik in range(self._kpoints.nk):
                occupations[ik][:nocc] = 2.0 * self._kpoints.weight(ik)

        return eigenvalues, eigenvectors, occupations

    def _compute_density(self, eigenvalues: Dict[int, np.ndarray],
                         eigenvectors: Dict[int, List[np.ndarray]],
                         occupations: Dict[int, np.ndarray]) -> core.Density:
        """Build new density from wavefunctions."""
        new_density = core.Density(self._basis)
        new_density.set_n_electrons(self._n_electrons)

        all_psi = []
        all_occ = []
        all_k = np.zeros(3)

        for ik in range(self._kpoints.nk):
            kpt = self._kpoints.kpoint(ik)
            if self._kpoints.nk == 1:
                all_k = kpt
            for ib in range(self._n_bands):
                all_psi.append(np.asarray(eigenvectors[ik][ib]))
                all_occ.append(occupations[ik][ib])

        new_density.set_from_eigenstates(all_psi, all_occ, all_k)
        return new_density

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def _compute_total_energy(self, eigenvalues, eigenvectors, occupations,
                               density) -> float:
        E_band = 0.0
        for ik in range(self._kpoints.nk):
            kpt = self._kpoints.kpoint(ik)
            e_kin = self._hamiltonian_cpp.compute_kinetic_energy(
                [np.asarray(v) for v in eigenvectors[ik]],
                occupations[ik].tolist(), kpt,
            )
            E_band += e_kin

        E_ewald = self._hamiltonian_cpp.ewald_energy()
        E_hartree = density.e_hartree()
        E_xc = density.e_xc()

        return E_band + E_ewald - E_hartree + E_xc

    # ------------------------------------------------------------------
    # Main SCF loop
    # ------------------------------------------------------------------

    def solve(self) -> SCFResult:
        """Run the SCF iteration.

        Replaces the old ``_mix_density`` with the configured
        :class:`DensityMixer`, enabling Broyden / DIIS convergence
        acceleration for metallic systems.
        """
        result = SCFResult()
        p = self._params

        if p.verbose:
            logger.info("=" * 60)
            logger.info("Starting SCF calculation")
            logger.info(f"  Atoms: {self._atoms.natoms}, Electrons: {self._n_electrons}")
            logger.info(f"  Plane waves: {self._basis.npw}, Bands: {self._n_bands}")
            logger.info(f"  k-points: {self._kpoints.nk}")
            logger.info(f"  Mixing: {p.mixing_method} (α={p.mixing_alpha:.2f})")
            if p.use_kerker_preconditioner:
                logger.info(f"  Kerker preconditioner: ON (κ={p.kerker_kappa:.2f} a.u.)")
            logger.info("=" * 60)

        # ----- state initialization --------------------------------
        self._mixer.reset()
        density = self._initial_density()
        self._density = density
        prev_energy = None
        converged = False

        # --- diagnostics for detecting charge sloshing -------------
        recent_residuals: List[float] = []
        recent_energies: List[float] = []

        step = 0
        for step in range(1, p.max_steps + 1):
            if p.verbose:
                logger.info(f"\nSCF step {step}/{p.max_steps}")

            # 1) Diagonalize, get bands
            eigenvalues, eigenvectors, occupations = self._compute_bands(density)

            # 2) Build output density
            new_density = self._compute_density(eigenvalues, eigenvectors, occupations)

            # 3) Compute residual (||ρ_out - ρ_in||)
            rho_g_in = np.asarray(density.rho_g)
            rho_g_out = np.asarray(new_density.rho_g)
            rho_diff = float(np.linalg.norm(rho_g_out - rho_g_in))

            # 4) Advanced density mixing
            x_in = self._density_to_vector(density)
            x_out = self._density_to_vector(new_density)

            x_next = self._mixer.mix(x_in, x_out)
            residual_norm = self._mixer.residual_norm or rho_diff

            # 5) Reconstruct density from mixed reciprocal-space vector
            try:
                density = self._vector_to_density(x_next)
            except Exception as exc:
                logger.warning(
                    f"  Vector→density reconstruction failed ({exc}); "
                    f"falling back to linear mixing for this step."
                )
                alpha_fallback = min(p.mixing_alpha, 0.2)
                x_fallback = x_in + alpha_fallback * (x_out - x_in)
                density = self._vector_to_density(x_fallback)

            # 6) Total energy
            E_total = self._compute_total_energy(
                eigenvalues, eigenvectors, occupations, density,
            )

            # --- bookkeeping -----------------------------------------
            result.energies_per_iter.append(E_total)
            result.density_errors.append(rho_diff)
            result.residual_norms.append(residual_norm)
            recent_energies.append(E_total)
            recent_residuals.append(residual_norm)
            if len(recent_residuals) > 8:
                recent_residuals.pop(0)
                recent_energies.pop(0)

            # --- charge-sloshing diagnostic --------------------------
            oscillating = False
            if len(recent_residuals) >= 4 and p.verbose:
                sign_changes = 0
                for i in range(1, len(recent_energies)):
                    dE = recent_energies[i] - recent_energies[i - 1]
                    if i >= 2:
                        prev_dE = recent_energies[i - 1] - recent_energies[i - 2]
                        if dE * prev_dE < 0 and abs(dE) > 1e-4 and abs(prev_dE) > 1e-4:
                            sign_changes += 1
                if sign_changes >= 2:
                    oscillating = True
                    logger.warning(
                        f"  ⚠ CHARGE SLOSHING DETECTED ({sign_changes} sign changes in energy)."
                    )
                    if not isinstance(self._mixer, (BroydenMixer, DIISMixer)):
                        logger.warning(
                            f"  ⚠ Switching from {type(self._mixer).__name__} to Broyden!"
                        )
                        self._params.mixing_method = "broyden"
                        self._params.use_kerker_preconditioner = True
                        self._mixer = self._build_mixer()
                        for ig in range(len(x_in)):
                            self._mixer.mix(x_in, x_out)  # seed mixer

            if p.verbose:
                logger.info(f"  Total energy: {E_total:.8f} Ha")
                logger.info(f"  Density Δ||ρ||: {rho_diff:.3e}")
                if oscillating:
                    logger.info(f"  (stabilizing charge oscillation via advanced mixer)")

            # --- convergence check -----------------------------------
            if prev_energy is not None:
                dE = abs(E_total - prev_energy)
                if dE < p.energy_tol and residual_norm < p.density_tol:
                    converged = True
                    if p.verbose:
                        logger.info(f"\n✓ SCF converged in {step} steps!")
                        logger.info(f"  Final total energy: {E_total:.8f} Ha")
                        logger.info(f"  Final residual:    {residual_norm:.3e}")
                    break

            prev_energy = E_total

        # --- finalization --------------------------------------------
        result.converged = converged
        result.n_iter = step
        result.total_energy = E_total
        result.eigenvalues = eigenvalues
        result.eigenvectors = eigenvectors
        result.occupations = occupations
        result.density = density
        self._density = density

        if not converged and p.verbose:
            logger.warning(
                f"\n✗ SCF did NOT converge in {p.max_steps} steps. "
                f"Consider increasing max_steps, reducing mixing_alpha, "
                f"or switching to 'broyden'/'diis' with Kerker preconditioner."
            )

        return result
