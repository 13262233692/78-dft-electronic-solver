"""Self-Consistent Field (SCF) solver for DFT."""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

from . import core
from .hamiltonian_wrapper import HamiltonianWrapper
from .kpoints import KPoints

logger = logging.getLogger(__name__)


@dataclass
class SCFParams:
    """Parameters controlling the SCF iteration."""
    max_steps: int = 100
    energy_tol: float = 1e-6
    density_tol: float = 1e-6
    mixing: float = 0.3
    diagonalization_method: str = "eigsh"
    n_extra_bands: int = 2
    smearing: float = 0.01
    smearing_type: str = "fermi"
    verbose: bool = True


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
    density: Optional[core.Density] = None


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
    """Find Fermi level by bisection to ensure correct electron count."""
    from scipy.optimize import bisect

    def f(mu):
        occ = smearing_function(energies, mu, sigma, kind)
        return 2.0 * occ.sum() - n_electrons

    e_min = energies.min() - 10.0
    e_max = energies.max() + 10.0
    if f(e_min) * f(e_max) > 0:
        return (e_min + e_max) / 2
    return bisect(f, e_min, e_max)


class SCFSolver:
    """Self-Consistent Field solver for plane-wave DFT."""

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

    @property
    def hamiltonian(self) -> HamiltonianWrapper:
        return self._hamiltonian

    @property
    def density(self) -> core.Density:
        return self._density

    @property
    def params(self) -> SCFParams:
        return self._params

    @params.setter
    def params(self, p: SCFParams):
        self._params = p

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
                        dist = np.linalg.norm(r - np.array(atom.position))
                        r_c = 1.5
                        Z = atom.atomic_number
                        prefactor = Z / ((np.sqrt(2 * np.pi) * r_c) ** 3)
                        rho_r[ir] += prefactor * np.exp(-dist ** 2 / (2 * r_c ** 2))

        density.set_from_r_space(rho_r)
        return density

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
                    preconditioner=precon
                )
            else:
                evals, evecs = self._hamiltonian.diagonalize(
                    kpt, self._n_bands, method="eigsh"
                )

            eigenvalues[ik] = evals

            psi_list = [evecs[:, i].copy() for i in range(self._n_bands)]
            norm = np.sqrt(self._basis.cell_volume)
            for i in range(self._n_bands):
                psi_list[i] /= np.linalg.norm(psi_list[i]) * norm
            eigenvectors[ik] = psi_list

            occupations[ik] = np.zeros(self._n_bands)

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

    def _mix_density(self, rho_old: core.Density, rho_new: core.Density
                     ) -> core.Density:
        """Simple linear mixing of densities."""
        alpha = self._params.mixing
        rho_g_old = np.asarray(rho_old.rho_g)
        rho_g_new = np.asarray(rho_new.rho_g)
        rho_g_mixed = (1 - alpha) * rho_g_old + alpha * rho_g_new

        result = core.Density(self._basis)
        result.set_n_electrons(self._n_electrons)
        result.set_from_r_space(np.real(np.fft.ifftn(rho_g_mixed)))
        return result

    def _compute_total_energy(self, eigenvalues, eigenvectors, occupations,
                               density) -> float:
        """Compute the DFT total energy."""
        E_band = 0.0
        for ik in range(self._kpoints.nk):
            kpt = self._kpoints.kpoint(ik)
            e_kin = self._hamiltonian_cpp.compute_kinetic_energy(
                [np.asarray(v) for v in eigenvectors[ik]],
                occupations[ik].tolist(), kpt
            )
            E_band += e_kin

        E_ewald = self._hamiltonian_cpp.ewald_energy()
        E_hartree = density.e_hartree()
        E_xc = density.e_xc()

        E_total = E_band + E_ewald - E_hartree + E_xc
        return E_total

    def solve(self) -> SCFResult:
        """Run the SCF iteration."""
        result = SCFResult()

        if self._params.verbose:
            logger.info("=" * 60)
            logger.info("Starting SCF calculation")
            logger.info(f"  Atoms: {self._atoms.natoms}, Electrons: {self._n_electrons}")
            logger.info(f"  Plane waves: {self._basis.npw}, Bands: {self._n_bands}")
            logger.info(f"  k-points: {self._kpoints.nk}")
            logger.info("=" * 60)

        density = self._initial_density()
        self._density = density

        prev_energy = None
        converged = False

        for step in range(1, self._params.max_steps + 1):
            if self._params.verbose:
                logger.info(f"\nSCF step {step}/{self._params.max_steps}")

            eigenvalues, eigenvectors, occupations = self._compute_bands(density)

            new_density = self._compute_density(eigenvalues, eigenvectors, occupations)

            rho_diff = np.linalg.norm(
                np.asarray(new_density.rho_g) - np.asarray(density.rho_g)
            )

            density = self._mix_density(density, new_density)

            E_total = self._compute_total_energy(
                eigenvalues, eigenvectors, occupations, density
            )

            result.energies_per_iter.append(E_total)
            result.density_errors.append(rho_diff)

            if self._params.verbose:
                logger.info(f"  Total energy: {E_total:.8f} Ha")
                logger.info(f"  Density change: {rho_diff:.2e}")

            if prev_energy is not None:
                dE = abs(E_total - prev_energy)
                if dE < self._params.energy_tol and rho_diff < self._params.density_tol:
                    converged = True
                    if self._params.verbose:
                        logger.info(f"\nSCF converged in {step} steps!")
                        logger.info(f"  Final total energy: {E_total:.8f} Ha")
                    break

            prev_energy = E_total

        result.converged = converged
        result.n_iter = step
        result.total_energy = E_total
        result.eigenvalues = eigenvalues
        result.eigenvectors = eigenvectors
        result.occupations = occupations
        result.density = density
        self._density = density

        if not converged and self._params.verbose:
            logger.warning(f"\nSCF did NOT converge in {self._params.max_steps} steps!")

        return result
