"""
Example 2: Silicon crystal band structure calculation

Demonstrates:
- Setting up a primitive cell of silicon
- Using Monkhorst-Pack k-point sampling
- Using LOBPCG vs ARPACK for sparse diagonalization
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from dft_solver import core, SCFSolver, SCFParams, KPoints, HamiltonianWrapper, _USING_CPP_EXT


def main():
    print(f"Using C++ extension: {_USING_CPP_EXT}")
    a_lat = 5.43

    cell = np.array([
        [0.0, a_lat/2, a_lat/2],
        [a_lat/2, 0.0, a_lat/2],
        [a_lat/2, a_lat/2, 0.0],
    ])

    ecut = 20.0
    basis = core.PlaneWaveBasis(cell, ecut)
    print(f"Silicon FCC cell volume: {basis.cell_volume:.4f} Ang^3")
    print(f"Plane waves: {basis.npw}")

    atoms = core.Atoms(cell)
    atoms.add_atom("Si", np.array([0.0, 0.0, 0.0]))
    atoms.add_atom("Si", np.array([a_lat/4, a_lat/4, a_lat/4]))
    print(f"Atoms: {atoms.natoms} Si atoms, {atoms.nelectrons()} electrons")

    kpoints = KPoints.monkhorst_pack((2, 2, 2))
    print(f"k-points: {kpoints.nk}")

    params = SCFParams(
        max_steps=50,
        energy_tol=1e-5,
        mixing=0.2,
        diagonalization_method="lobpcg",
        smearing=0.02,
        verbose=True,
    )

    solver = SCFSolver(basis, atoms, kpoints, params)
    result = solver.solve()

    print("\n" + "="*60)
    if result.converged:
        print(f"SCF converged in {result.n_iter} iterations")
    else:
        print(f"SCF NOT converged after {result.n_iter} iterations")

    print(f"Total energy per cell: {result.total_energy:.8f} Ha")
    print(f"Cohesive energy (per atom): {result.total_energy/2:.8f} Ha")

    for ik in range(kpoints.nk):
        print(f"\nk-point {ik}: {kpoints.kpoint(ik)}")
        for i, e in enumerate(result.eigenvalues[ik][:8]):
            print(f"  Band {i+1:2d}: {e:.6f} Ha")

    print("\nBand structure along high-symmetry path...")
    high_sym_path = [
        ("Gamma", np.array([0.0, 0.0, 0.0])),
        ("X",     np.array([0.0, 0.5, 0.5])),
        ("L",     np.array([0.5, 0.5, 0.5])),
    ]

    ham_wrap = HamiltonianWrapper(solver.hamiltonian.hamiltonian)

    for name, kpt in high_sym_path:
        evals, _ = ham_wrap.diagonalize(kpt, n_bands=8, method="eigsh")
        print(f"  {name}: {evals[:4]}")


if __name__ == "__main__":
    main()
