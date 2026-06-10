"""
Example 4: Tetrahedron-Method Density of States (DOS) Computation

Demonstrates the analytical tetrahedron integration method for
computing the electronic Density of States from k-point sampled
band structures.  This approach avoids Gaussian smearing artifacts
and can precisely classify materials as metal/semiconductor/insulator.

We compute DOS for:
  1. H atom in a large box (insulator-like, large gap)
  2. Free-electron model (metal, parabolic DOS)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dft_solver import (
    core, SCFSolver, SCFParams, KPoints,
    DOSCalculator, DOSResult, compute_dos, plot_dos,
)

# ============================================================================
# Part 1: H atom — insulator
# ============================================================================
print("=" * 65)
print("  Part 1: H atom DOS (tetrahedron method)")
print("=" * 65)

cell = np.eye(3) * 10.0
ecut = 5.0
basis = core.PlaneWaveBasis(cell, ecut)
atoms = core.Atoms(cell)
atoms.add_atom("H", np.array([5.0, 5.0, 5.0]))

kpoints = KPoints.monkhorst_pack((3, 3, 3))
params = SCFParams(
    max_steps=12,
    energy_tol=1e-3,
    density_tol=1e-3,
    mixing_method="broyden",
    use_kerker_preconditioner=True,
    smearing=0.03,
    verbose=True,
)

solver = SCFSolver(basis, atoms, kpoints, params)
result = solver.solve()

status = "✓ CONVERGED" if result.converged else "~ NOT CONVERGED"
print(f"\nSCF: {status} in {result.n_iter} steps, E = {result.total_energy:.6f} Ha")
print(f"Grid: {result.kpoint_grid}")

dos_result = result.compute_dos(
    n_energy=1500,
    method="bloechl",
    plot=False,
)

print(f"\n  DOS Results:")
print(f"  ─────────────────────────────────────")
print(f"  Fermi level  : {dos_result.fermi_level:.6f} Ha ({dos_result.fermi_level * 27.2114:.4f} eV)")
print(f"  VBM          : {dos_result.vbm:.6f} Ha ({dos_result.vbm * 27.2114:.4f} eV)")
print(f"  CBM          : {dos_result.cbm:.6f} Ha ({dos_result.cbm * 27.2114:.4f} eV)")
print(f"  Band gap     : {dos_result.band_gap:.6f} Ha ({dos_result.band_gap * 27.2114:.4f} eV)")
print(f"  Material type: {dos_result.material_type.upper()}")
print(f"  Tetrahedra   : {dos_result.n_tetrahedra}")
print(f"  k-points     : {dos_result.n_kpoints}")
print(f"  Bands        : {dos_result.n_bands}")

plot_file = os.path.join(os.path.dirname(__file__), "dos_h_atom.png")
plot_dos(dos_result, filename=plot_file, show=False,
         title="H Atom DOS (Tetrahedron Method)",
         n_bands_show=min(4, dos_result.n_bands))
print(f"\n  Plot saved: {plot_file}")

# ============================================================================
# Part 2: Free-electron model — metal
# ============================================================================
print("\n" + "=" * 65)
print("  Part 2: Free-electron model DOS (metal)")
print("=" * 65)

grid_fe = (4, 4, 4)
nk_fe = 4 ** 3
from dft_solver.dos import _kpoint_grid_indices
kpts_fe = _kpoint_grid_indices(grid_fe)
kpts_bz = kpts_fe - 0.5

eigenvalues_fe = {}
for ik in range(nk_fe):
    k = kpts_bz[ik]
    e_k = 0.5 * np.dot(k, k)
    eigenvalues_fe[ik] = np.array([e_k])

dos_fe = compute_dos(
    eigenvalues_fe,
    n_electrons=2.0,
    grid=grid_fe,
    n_energy=1000,
    method="bloechl",
    plot=False,
)

print(f"  Free-electron DOS:")
print(f"  Fermi level  : {dos_fe.fermi_level:.6f} Ha ({dos_fe.fermi_level * 27.2114:.4f} eV)")
print(f"  Material type: {dos_fe.material_type.upper()}")
print(f"  Tetrahedra   : {dos_fe.n_tetrahedra}")

de = dos_fe.energies[1] - dos_fe.energies[0]
integrated = np.sum(dos_fe.dos_total) * de
print(f"  Integrated DOS = {integrated:.4f} (expected 2.0 for 1 band × 2 spin)")

plot_file_fe = os.path.join(os.path.dirname(__file__), "dos_free_electron.png")
plot_dos(dos_fe, filename=plot_file_fe, show=False,
         title="Free Electron DOS (Tetrahedron Method)")
print(f"  Plot saved: {plot_file_fe}")

# ============================================================================
# Part 3: Artificial semiconductor — gap detection
# ============================================================================
print("\n" + "=" * 65)
print("  Part 3: Artificial semiconductor (gap = 0.5 Ha ≈ 13.6 eV)")
print("=" * 65)

eigenvalues_sc = {}
for ik in range(nk_fe):
    k = kpts_bz[ik]
    e_k = 0.5 * np.dot(k, k)
    eigenvalues_sc[ik] = np.array([-1.0 - 0.2 * e_k, -0.5 - 0.1 * e_k, 0.5 + 0.2 * e_k])

dos_sc = compute_dos(
    eigenvalues_sc,
    n_electrons=4.0,
    grid=grid_fe,
    n_energy=1000,
    method="bloechl",
    plot=False,
)

print(f"  Gap = {dos_sc.band_gap:.4f} Ha ({dos_sc.band_gap * 27.2114:.2f} eV)")
print(f"  Material type: {dos_sc.material_type.upper()}")

plot_file_sc = os.path.join(os.path.dirname(__file__), "dos_semiconductor.png")
plot_dos(dos_sc, filename=plot_file_sc, show=False,
         title="Artificial Semiconductor DOS (Tetrahedron Method)",
         n_bands_show=3)
print(f"  Plot saved: {plot_file_sc}")

print("\n" + "=" * 65)
print("  All DOS computations complete ✓")
print("=" * 65)
