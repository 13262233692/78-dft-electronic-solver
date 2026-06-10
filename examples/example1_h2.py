"""
Example 1: Simple hydrogen molecule DFT calculation

Demonstrates basic usage of the DFT solver:
- Setting up a crystal cell
- Adding atoms
- Running SCF calculation
- Extracting eigenvalues and total energy
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from dft_solver import core, SCFSolver, SCFParams, KPoints, _USING_CPP_EXT


def main():
    print(f"Using C++ extension: {_USING_CPP_EXT}")

    a = 10.0
    cell = np.eye(3) * a

    ecut = 15.0
    basis = core.PlaneWaveBasis(cell, ecut)
    print(f"Plane wave basis: {basis.npw} plane waves")
    print(f"FFT grid: {basis.ngx} x {basis.ngy} x {basis.ngz}")

    atoms = core.Atoms(cell)
    atoms.add_atom("H", np.array([a/2 - 0.7, a/2, a/2]))
    atoms.add_atom("H", np.array([a/2 + 0.7, a/2, a/2]))
    print(f"System: {atoms.natoms} H atoms, {atoms.nelectrons()} electrons")

    kpoints = KPoints.gamma()

    params = SCFParams(
        max_steps=30,
        energy_tol=1e-5,
        density_tol=1e-4,
        mixing=0.3,
        diagonalization_method="eigsh",
        smearing=0.01,
        verbose=True,
    )

    solver = SCFSolver(basis, atoms, kpoints, params)
    result = solver.solve()

    print("\n" + "="*60)
    if result.converged:
        print(f"SCF converged in {result.n_iter} iterations")
    else:
        print(f"SCF NOT converged after {result.n_iter} iterations")

    print(f"Total energy: {result.total_energy:.8f} Ha")

    for ik in range(kpoints.nk):
        print(f"\nEigenvalues at k-point {ik}:")
        for i, e in enumerate(result.eigenvalues[ik]):
            occ = result.occupations[ik][i]
            print(f"  Band {i+1:2d}: {e:12.6f} Ha  (occ={occ:.4f})")

    print("\nEnergies per iteration:")
    for i, e in enumerate(result.energies_per_iter):
        print(f"  Step {i+1:2d}: {e:.8f} Ha")


if __name__ == "__main__":
    main()
