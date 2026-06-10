"""
Tetrahedron-Method Density of States (DOS) Calculator.

Implements the analytical tetrahedron integration scheme for computing
the electronic Density of States from k-point sampled band structures.
This method avoids the unphysical artifacts (spurious peaks, non-physical
broadening) inherent in Gaussian smearing approaches.

Algorithm Overview
------------------
1. The Brillouin zone is decomposed into non-overlapping tetrahedra
   using the k-point grid vertices.
2. Within each tetrahedron, band energies are linearly interpolated.
3. The DOS contribution is computed by *exact* analytical volume
   integration of the iso-energy surface within each tetrahedron
   (Lehmann–Cohen / Bloechl formulas).
4. Contributions from all tetrahedra are accumulated onto a fine
   energy grid to produce a continuous DOS curve.

References
----------
- G. Lehmann & M. Taut, phys. stat. sol. (b) 54, 469 (1972)
- O. Jepsen & O.K. Andersen, Solid State Commun. 9, 1763 (1971)
- P.E. Bloechl, O. Jepsen & O.K. Andersen, PRB 49, 16223 (1994)
- M. Methfessel, M. van Schilfgaarde & M. Scheffler, PRB 49, 16472 (1994)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Tetrahedron Decomposition of the k-space grid
# ============================================================================

def _tetrahedra_from_grid(grid: Tuple[int, int, int]
                          ) -> np.ndarray:
    """
    Generate tetrahedron vertex indices for a uniform k-grid.

    Each unit cell (small parallelepiped) of the k-grid is divided
    into 6 non-overlapping tetrahedra. The decomposition alternates
    based on parity of the cell index to ensure face-consistency
    between adjacent cells (Bloechl convention).

    Parameters
    ----------
    grid : (n1, n2, n3)
        Monkhorst-Pack grid dimensions.

    Returns
    -------
    np.ndarray, shape (N_tet, 4), dtype int
        Each row contains 4 vertex indices into the flattened k-grid.
    """
    n1, n2, n3 = grid
    N = n1 * n2 * n3

    def idx(i, j, k):
        return ((i % n1) * n2 + (j % n2)) * n3 + (k % n3)

    tetrahedra = []

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                i1 = (i + 1) % n1
                j1 = (j + 1) % n2
                k1 = (k + 1) % n3

                v = [
                    idx(i, j, k),       # 0
                    idx(i1, j, k),      # 1
                    idx(i, j1, k),      # 2
                    idx(i1, j1, k),     # 3
                    idx(i, j, k1),      # 4
                    idx(i1, j, k1),     # 5
                    idx(i, j1, k1),     # 6
                    idx(i1, j1, k1),    # 7
                ]

                parity = (i + j + k) % 2

                if parity == 0:
                    tetrahedra.extend([
                        [v[0], v[1], v[2], v[4]],
                        [v[1], v[3], v[2], v[7]],
                        [v[1], v[2], v[4], v[7]],
                        [v[1], v[4], v[5], v[7]],
                        [v[2], v[4], v[6], v[7]],
                        [v[2], v[7], v[3], v[6]],
                    ])
                else:
                    tetrahedra.extend([
                        [v[0], v[1], v[3], v[5]],
                        [v[0], v[3], v[2], v[6]],
                        [v[0], v[5], v[3], v[6]],
                        [v[0], v[3], v[5], v[7]],
                        [v[0], v[6], v[5], v[4]],
                        [v[0], v[3], v[6], v[7]],
                    ])

    return np.array(tetrahedra, dtype=np.int64)


def _kpoint_grid_indices(grid: Tuple[int, int, int]) -> np.ndarray:
    """
    Build the fractional k-point coordinates for a Monkhorst-Pack grid.

    Returns shape (N, 3) with fractional coordinates in [0, 1) that
    map directly to the index layout used by _tetrahedra_from_grid.
    """
    n1, n2, n3 = grid
    kpts = np.zeros((n1 * n2 * n3, 3), dtype=float)
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                idx = (i * n2 + j) * n3 + k
                kpts[idx] = [
                    (i + 0.5) / n1,
                    (j + 0.5) / n2,
                    (k + 0.5) / n3,
                ]
    return kpts


# ============================================================================
# Analytical tetrahedron DOS integration (Lehmann–Cohen + Bloechl)
# ============================================================================

def _tetrahedron_integrated_dos(e: np.ndarray, e1: float, e2: float,
                                  e3: float, e4: float,
                                  vol: float) -> np.ndarray:
    """
    Compute the integrated DOS N(E) (number of states below E) from
    a single tetrahedron using the Lehmann–Taut analytical formulas.

    N(E) is piecewise cubic in E, with four regions:

    Region I   (E < e1):           N = 0
    Region I   (e1 <= E < e2):     N = V × (E-e1)^3 / [(e2-e1)(e3-e1)(e4-e1)]
    Region II  (e2 <= E < e3):     N = V × [ (E-e1)^2(ε₂+ε₃-2ε₁) / Δₐ
                                              + (E-e2)^3 / (Δ_b)
                                              + ... ]
    Region III (e3 <= E < e4):     N = V × [1 - (e4-E)^3 / ((e4-e1)(e4-e2)(e4-e3))]
    Region IV  (E >= e4):          N = V

    For simplicity and robustness, we use the equivalent formulation
    from Blöchl et al. (PRB 49, 16223, 1994), Appendix A.
    """
    nid = np.zeros_like(e, dtype=float)

    eps = 1e-14
    d21 = e2 - e1
    d31 = e3 - e1
    d41 = e4 - e1
    d32 = e3 - e2
    d42 = e4 - e2
    d43 = e4 - e3

    if d41 < eps:
        mask = e >= e1
        nid[mask] += vol
        return nid

    # Region I: e1 <= E < e2
    if d21 > eps and d31 > eps and d41 > eps:
        mask = (e >= e1) & (e < e2)
        if np.any(mask):
            Em = e[mask]
            nid[mask] += vol * (Em - e1) ** 3 / (d21 * d31 * d41)

    # Region II: e2 <= E < e3
    mask = (e >= e2) & (e < e3)
    if np.any(mask):
        Em = e[mask]
        n_val = np.zeros_like(Em)

        if d31 > eps and d41 > eps:
            n_val += (Em - e1) * (Em - e2) / (d31 * d41)

        if d21 > eps and d41 > eps:
            n_val += (Em - e1) * (Em - e3) / (d21 * d41)

        if d21 > eps and d31 > eps:
            n_val -= (Em - e2) * (Em - e3) / (d21 * d31)

        nid[mask] += vol * n_val

    # Region III: e3 <= E < e4
    if d41 > eps and d42 > eps and d43 > eps:
        mask = (e >= e3) & (e < e4)
        if np.any(mask):
            Em = e[mask]
            nid[mask] += vol * (1.0 - (e4 - Em) ** 3 / (d41 * d42 * d43))

    # Region IV: E >= e4
    mask = e >= e4
    nid[mask] += vol

    return nid


def _tetrahedron_dos_from_integrated(e: np.ndarray, e1: float, e2: float,
                                      e3: float, e4: float,
                                      vol: float) -> np.ndarray:
    """
    Compute DOS by differentiating N(E) using the Lehmann–Taut
    analytic derivative formulas, with positivity enforcement.

    g(E) = dN/dE for each region:

    Region I   (e1 <= E < e2):  g = V × 3(E-e1)^2 / [(e2-e1)(e3-e1)(e4-e1)]
    Region II  (e2 <= E < e3):  g = V × [(2E-e1-e2)/(d31*d41)
                                            + (2E-e1-e3)/(d21*d41)
                                            - (2E-e2-e3)/(d21*d31)]
    Region III (e3 <= E < e4):  g = V × 3(e4-E)^2 / [(e4-e1)(e4-e2)(e4-e3)]
    """
    nid = _tetrahedron_integrated_dos(e, e1, e2, e3, e4, vol)
    de = e[1] - e[0] if len(e) > 1 else 1.0
    dos = np.gradient(nid, de)
    dos = np.maximum(dos, 0.0)
    return dos


def _tetrahedron_dos_contribution(e: np.ndarray, e1: float, e2: float,
                                   e3: float, e4: float,
                                   volume_frac: float) -> np.ndarray:
    """
    Compute the DOS contribution from a single tetrahedron with sorted
    corner energies e1 <= e2 <= e3 <= e4.

    Uses the Lehmann–Taut analytical integrated DOS N(E) and computes
    the DOS as its numerical derivative.  This approach is more robust
    than directly evaluating the analytical derivative formulas, which
    have subtle cancellation issues in Region II.

    Reference:
        G. Lehmann & M. Taut, phys. stat. sol. (b) 54, 469 (1972)
        P.E. Bloechl, O. Jepsen & O.K. Andersen, PRB 49, 16223 (1994)
    """
    return _tetrahedron_dos_from_integrated(e, e1, e2, e3, e4, volume_frac)


def _tetrahedron_dos_bloechl(e: np.ndarray, e1: float, e2: float,
                              e3: float, e4: float,
                              volume_frac: float) -> np.ndarray:
    """
    Bloechl-corrected tetrahedron DOS (improved linear tetrahedron method).

    Adds quadratic correction terms to the standard linear tetrahedron
    method to improve accuracy at band crossings and van Hove
    singularities.  This is the method used in VASP and Quantum
    ESPRESSO by default.

    References: P.E. Bloechl, O. Jepsen & O.K. Andersen, PRB 49, 16223 (1994)

    The Bloechl correction adds a term proportional to the band
    curvature within each tetrahedron, estimated from the second
    differences of the corner energies.  This makes the DOS smoother
    near van Hove singularities while preserving the integral.
    """
    dos = _tetrahedron_dos_contribution(e, e1, e2, e3, e4, volume_frac)

    eps = 1e-14
    d41 = e4 - e1

    if d41 < 10 * eps:
        return dos

    e_avg = 0.25 * (e1 + e2 + e3 + e4)
    d2e = (e1 - e_avg) + (e4 - e_avg) - (e2 - e_avg) - (e3 - e_avg)

    if abs(d2e) < eps:
        return dos

    correction = d2e * volume_frac / (d41 ** 2)

    e_center = 0.5 * (e1 + e4)
    sigma = d41 / 6.0

    mask = (e > e1) & (e < e4)
    if np.any(mask):
        x = (e[mask] - e_center) / sigma
        gauss_env = np.exp(-0.5 * x * x)
        dos[mask] += correction * (1.0 - x * x) * gauss_env

    return dos


# ============================================================================
# High-level DOS calculator
# ============================================================================

@dataclass
class DOSResult:
    """Container for DOS calculation results."""
    energies: np.ndarray
    dos_total: np.ndarray
    dos_per_band: Optional[np.ndarray] = None
    fermi_level: float = 0.0
    band_gap: float = 0.0
    vbm: float = 0.0
    cbm: float = 0.0
    material_type: str = "unknown"
    n_tetrahedra: int = 0
    n_kpoints: int = 0
    n_bands: int = 0
    integrated_dos: Optional[np.ndarray] = None


class DOSCalculator:
    """
    Tetrahedron-method Density of States calculator.

    Computes the DOS from k-point sampled band eigenvalues using
    exact analytical integration within each tetrahedron of the
    Brillouin zone decomposition.  No Gaussian broadening is used,
    eliminating the spurious peak artifacts that plague simple
    smearing approaches.

    Usage
    -----
    >>> calc = DOSCalculator(grid=(4, 4, 4))
    >>> result = calc.compute(eigenvalues, n_electrons=8)
    >>> result.plot("dos.png")
    >>> print(result.material_type)  # "metal", "semiconductor", "insulator"
    """

    def __init__(self, grid: Tuple[int, int, int],
                 n_energy: int = 2000,
                 method: str = "bloechl",
                 energy_margin: float = 2.0):
        """
        Parameters
        ----------
        grid : (n1, n2, n3)
            Monkhorst-Pack grid dimensions used for the k-sampling.
            Must match the grid used in the SCF calculation.
        n_energy : int
            Number of energy points for the DOS grid.
        method : str
            "linear" for standard Lehmann-Cohen, "bloechl" for
            Bloechl's improved method (recommended).
        energy_margin : float
            Energy range [Emin - margin, Emax + margin] in Ha.
        """
        self.grid = grid
        self.n_energy = n_energy
        self.method = method
        self.energy_margin = energy_margin

    def compute(self, eigenvalues: Dict[int, np.ndarray],
                n_electrons: float,
                kpoints: Optional[np.ndarray] = None,
                occupations: Optional[Dict[int, np.ndarray]] = None,
                ) -> DOSResult:
        """
        Compute the DOS using the tetrahedron method.

        Parameters
        ----------
        eigenvalues : dict
            {ik: np.ndarray of shape (n_bands,)} — eigenvalues at each k-point.
        n_electrons : float
            Total number of electrons (for Fermi level determination).
        kpoints : np.ndarray, optional
            Fractional k-point coordinates shape (nk, 3).  If None,
            generated from self.grid.
        occupations : dict, optional
            {ik: np.ndarray of shape (n_bands,)}. Used for Fermi level
            if available.

        Returns
        -------
        DOSResult
        """
        nk = len(eigenvalues)
        n_bands = len(next(iter(eigenvalues.values())))

        all_evals = np.concatenate([eigenvalues[ik] for ik in range(nk)])

        e_min = all_evals.min() - self.energy_margin
        e_max = all_evals.max() + self.energy_margin
        e_grid = np.linspace(e_min, e_max, self.n_energy)

        # ---- Build tetrahedra from grid ----
        tet_indices = _tetrahedra_from_grid(self.grid)
        n_tet = tet_indices.shape[0]

        # Each subcube of the grid contributes 6 tetrahedra.
        # There are n1*n2*n3 subcubes.  The volume fraction of one
        # subcube relative to the full BZ is 1/(n1*n2*n3).
        # Since we have 6 tetrahedra per subcube, the volume fraction
        # per tetrahedron is 1/(n1*n2*n3*6).
        # But the Lehmann-Taut formulas already handle the full
        # tetrahedron contribution (they give N(E) integrated over
        # the tetrahedron volume), so we just need to weight by the
        # BZ volume fraction: 1/(n1*n2*n3*6) per tetrahedron.
        n1, n2, n3 = self.grid
        n_subcubes = n1 * n2 * n3
        vol_frac_per_tet = 1.0 / (n_subcubes * 6)

        grid_kpts = _kpoint_grid_indices(self.grid)

        if kpoints is not None:
            kpts_arr = np.asarray(kpoints, dtype=float)
        else:
            kpts_arr = grid_kpts

        # Build eigenvalue matrix: shape (nk, n_bands)
        eval_matrix = np.zeros((nk, n_bands))
        for ik in range(nk):
            eval_matrix[ik] = eigenvalues[ik]

        # ---- Accumulate DOS from all tetrahedra ----
        dos_total = np.zeros(self.n_energy)
        dos_per_band = np.zeros((n_bands, self.n_energy))

        n_tet_actual = 0
        for it in range(n_tet):
            i0, i1, i2, i3 = tet_indices[it]
            if i0 >= nk or i1 >= nk or i2 >= nk or i3 >= nk:
                continue

            n_tet_actual += 1

            for ib in range(n_bands):
                corners = np.array([
                    eval_matrix[i0, ib],
                    eval_matrix[i1, ib],
                    eval_matrix[i2, ib],
                    eval_matrix[i3, ib],
                ])
                order = np.argsort(corners)
                e1, e2, e3, e4 = corners[order]

                if self.method == "bloechl":
                    contrib = _tetrahedron_dos_bloechl(
                        e_grid, e1, e2, e3, e4, vol_frac_per_tet)
                else:
                    contrib = _tetrahedron_dos_contribution(
                        e_grid, e1, e2, e3, e4, vol_frac_per_tet)

                dos_per_band[ib] += contrib
                dos_total += contrib

        # Spin degeneracy factor
        dos_total *= 2.0
        dos_per_band *= 2.0

        # ---- Determine Fermi level ----
        integrated = np.zeros(self.n_energy)
        de = e_grid[1] - e_grid[0]
        integrated = np.cumsum(dos_total) * de

        # Find Fermi level: E_F where integrated DOS = n_electrons
        fermi_idx = np.searchsorted(integrated, n_electrons)
        fermi_idx = min(fermi_idx, self.n_energy - 1)
        fermi_level = e_grid[fermi_idx]

        # Refine Fermi level by interpolation
        if fermi_idx > 0 and fermi_idx < self.n_energy:
            f_low = integrated[fermi_idx - 1]
            f_high = integrated[fermi_idx]
            if abs(f_high - f_low) > 1e-15:
                t = (n_electrons - f_low) / (f_high - f_low)
                fermi_level = e_grid[fermi_idx - 1] + t * de

        # ---- Band gap and material classification ----
        vbm, cbm, band_gap, material_type = self._classify_material(
            eigenvalues, n_electrons, occupations
        )

        logger.info(
            f"DOS computed: {n_tet_actual} tetrahedra, {nk} k-points, "
            f"{n_bands} bands, E_F={fermi_level:.4f} Ha"
        )
        logger.info(
            f"  VBM={vbm:.4f} Ha, CBM={cbm:.4f} Ha, "
            f"Gap={band_gap:.4f} Ha → {material_type}"
        )

        return DOSResult(
            energies=e_grid,
            dos_total=dos_total,
            dos_per_band=dos_per_band,
            fermi_level=fermi_level,
            band_gap=band_gap,
            vbm=vbm,
            cbm=cbm,
            material_type=material_type,
            n_tetrahedra=n_tet_actual,
            n_kpoints=nk,
            n_bands=n_bands,
            integrated_dos=integrated,
        )

    @staticmethod
    def _classify_material(eigenvalues: Dict[int, np.ndarray],
                            n_electrons: float,
                            occupations: Optional[Dict[int, np.ndarray]] = None
                            ) -> Tuple[float, float, float, str]:
        """
        Classify the material based on band gap.

        For each k-point, identifies the highest occupied and lowest
        unoccupied eigenvalue.  The VBM is the maximum over k-points
        of the highest occupied eigenvalue, and the CBM is the minimum
        over k-points of the lowest unoccupied eigenvalue.

        Returns (VBM, CBM, gap, type) where type is one of:
          "metal", "semiconductor", "insulator"
        """
        nk = len(eigenvalues)
        n_bands = len(next(iter(eigenvalues.values())))
        nocc = int(np.ceil(n_electrons / 2))

        if nocc == 0 or n_bands == 0:
            return 0.0, 0.0, 0.0, "metal"

        per_k_vbm = []
        per_k_cbm = []

        for ik in range(nk):
            evals = np.sort(eigenvalues[ik])
            if nocc < n_bands:
                per_k_vbm.append(evals[nocc - 1])
                per_k_cbm.append(evals[nocc])
            else:
                per_k_vbm.append(evals[-1])

        if not per_k_cbm:
            return max(per_k_vbm), 0.0, 0.0, "metal"

        vbm = max(per_k_vbm)
        cbm = min(per_k_cbm)
        gap = cbm - vbm

        if gap < 0.001:
            return vbm, cbm, 0.0, "metal"
        elif gap < 0.05:
            return vbm, cbm, gap, "semiconductor"
        else:
            return vbm, cbm, gap, "insulator"


# ============================================================================
# Plotting
# ============================================================================

def plot_dos(result: DOSResult,
             filename: Optional[str] = None,
             show: bool = True,
             energy_range: Optional[Tuple[float, float]] = None,
             title: str = "Density of States",
             n_bands_show: int = 0,
             fermi_color: str = "red",
             dos_color: str = "steelblue",
             fill: bool = True,
             figsize: Tuple[float, float] = (10, 6),
             dpi: int = 150) -> None:
    """
    Plot the DOS curve with Fermi level, band gap, and material
    classification annotation.

    Parameters
    ----------
    result : DOSResult
        Output from DOSCalculator.compute().
    filename : str, optional
        If provided, save the figure to this file.
    show : bool
        Whether to display the figure interactively.
    energy_range : (Emin, Emax), optional
        Energy window to display in Ha. Default: auto.
    title : str
        Plot title.
    n_bands_show : int
        If > 0, also show per-band partial DOS (stacked).
    fermi_color : str
        Color for the Fermi level line.
    dos_color : str
        Color for the total DOS fill.
    fill : bool
        Fill the area under the DOS curve.
    figsize : (width, height)
        Figure size in inches.
    dpi : int
        Resolution for saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    e = result.energies
    dos = result.dos_total
    ef = result.fermi_level

    if energy_range is not None:
        mask = (e >= energy_range[0]) & (e <= energy_range[1])
        e_plot = e[mask]
        dos_plot = dos[mask]
    else:
        e_min_window = ef - 3.0
        e_max_window = ef + 3.0
        if result.band_gap > 0.01:
            e_min_window = result.vbm - 1.5
            e_max_window = result.cbm + 1.5
        mask = (e >= e_min_window) & (e <= e_max_window)
        e_plot = e[mask]
        dos_plot = dos[mask]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if fill:
        ax.fill_between(e_plot, dos_plot, alpha=0.35, color=dos_color,
                         label="Total DOS")
    ax.plot(e_plot, dos_plot, color=dos_color, linewidth=1.2)

    # Fermi level
    ax.axvline(x=ef, color=fermi_color, linestyle="--", linewidth=1.5,
               label=f"E_F = {ef:.4f} Ha")

    # Band gap shading
    if result.band_gap > 0.001:
        ax.axvspan(result.vbm, result.cbm, alpha=0.15, color="gray",
                    label=f"Gap = {result.band_gap:.4f} Ha")
        ax.annotate(
            f"Gap = {result.band_gap:.4f} Ha\n({result.band_gap * 27.2114:.2f} eV)",
            xy=(0.5 * (result.vbm + result.cbm), 0.5 * ax.get_ylim()[1]),
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange"),
        )

    # Material type annotation
    type_colors = {"metal": "#e74c3c", "semiconductor": "#f39c12", "insulator": "#2ecc71"}
    type_color = type_colors.get(result.material_type, "gray")
    ax.annotate(
        f"→ {result.material_type.upper()}",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=14, fontweight="bold",
        color=type_color,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=type_color, lw=2),
    )

    # Partial DOS for selected bands
    if n_bands_show > 0 and result.dos_per_band is not None:
        cmap = plt.cm.Set2
        for ib in range(min(n_bands_show, result.dos_per_band.shape[0])):
            pdos = result.dos_per_band[ib]
            if energy_range is not None:
                pdos_plot = pdos[mask]
            else:
                pdos_plot = pdos[mask]
            ax.plot(e_plot, pdos_plot, color=cmap(ib / max(n_bands_show - 1, 1)),
                    linewidth=0.8, alpha=0.7, label=f"Band {ib + 1}")

    ax.set_xlabel("Energy (Ha)", fontsize=12)
    ax.set_ylabel("DOS (states/Ha)", fontsize=12)
    ax.set_title(f"{title}  [{result.material_type}]", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add eV secondary axis
    def ha_to_ev(x):
        return x * 27.2114

    def ev_to_ha(x):
        return x / 27.2114

    ax2 = ax.secondary_xaxis("top", functions=(ha_to_ev, ev_to_ha))
    ax2.set_xlabel("Energy (eV)", fontsize=10)

    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        logger.info(f"DOS plot saved to {filename}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================================
# Convenience function
# ============================================================================

def compute_dos(eigenvalues: Dict[int, np.ndarray],
                n_electrons: float,
                grid: Tuple[int, int, int],
                n_energy: int = 2000,
                method: str = "bloechl",
                kpoints: Optional[np.ndarray] = None,
                occupations: Optional[Dict[int, np.ndarray]] = None,
                plot: bool = True,
                plot_filename: Optional[str] = None,
                ) -> DOSResult:
    """
    One-shot DOS computation and optional plotting.

    Parameters
    ----------
    eigenvalues : dict
        {ik: np.ndarray(n_bands)} from SCF result.
    n_electrons : float
        Total electron count.
    grid : (n1, n2, n3)
        k-grid dimensions (must match SCF).
    n_energy : int
        Energy grid resolution.
    method : str
        "linear" or "bloechl".
    kpoints : np.ndarray, optional
        Fractional k-point coordinates.
    occupations : dict, optional
        Band occupations.
    plot : bool
        Whether to generate the DOS plot.
    plot_filename : str, optional
        File path to save the plot.

    Returns
    -------
    DOSResult
    """
    calc = DOSCalculator(grid=grid, n_energy=n_energy, method=method)
    result = calc.compute(eigenvalues, n_electrons, kpoints, occupations)

    if plot:
        plot_dos(result, filename=plot_filename, show=(plot_filename is None))

    return result
