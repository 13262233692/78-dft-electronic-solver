"""
Charge-Sloshing Benchmark: Linear vs Broyden vs DIIS

Compares the three density-mixing algorithms on a metallic system
to demonstrate that Broyden + Kerker completely eliminates the
charge-sloshing instability described in the bug report.

Physics:
    A free-electron gas (or hydrogen cluster in a large box with
    high smearing, mimicking metallic behavior) has large DOS at
    the Fermi surface.  For these systems the Jacobian
        J_{GG'} = dρ[V]_{out,G} / dV_{in,G'}
    has eigenvalues close to -1 at small G, leading to 2-step
    oscillatory instability under Picard (linear) iteration.
"""

from __future__ import annotations
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, ".")
from dft_solver import (
    core, SCFSolver, SCFParams, KPoints, _USING_CPP_EXT,
    LinearMixer, BroydenMixer, DIISMixer,
)


def build_metallic_like_system(ecut: float = 6.0, a_box: float = 12.0):
    """
    Build a system mimicking metallic behavior:
    3 H atoms spread in a large cubic cell with large smearing
    so that the DOS at E_F is artificially high.
    """
    cell = np.eye(3) * a_box
    basis = core.PlaneWaveBasis(cell, ecut)
    atoms = core.Atoms(cell)
    atoms.add_atom("H", np.array([a_box * 0.30, a_box / 2, a_box / 2]))
    atoms.add_atom("H", np.array([a_box * 0.50, a_box / 2, a_box / 2]))
    atoms.add_atom("H", np.array([a_box * 0.70, a_box / 2, a_box / 2]))
    return basis, atoms


def run_one(method: str, **kwargs):
    """Run a single SCF with the given mixing method."""
    ecut = kwargs.pop("ecut", 5.0)
    a_box = kwargs.pop("a_box", 10.0)
    max_steps = kwargs.pop("max_steps", 20)
    smearing = kwargs.pop("smearing", 0.05)

    basis, atoms = build_metallic_like_system(ecut=ecut, a_box=a_box)
    kpoints = KPoints.gamma()

    # Build explicit params so we can set *any* mixer config
    params = SCFParams(
        max_steps=max_steps,
        energy_tol=1e-4,
        density_tol=1e-4,
        smearing=smearing,
        verbose=False,
        **kwargs,
    )

    solver = SCFSolver(basis, atoms, kpoints, params)
    result = solver.solve()
    return result


def count_oscillations(energies):
    """Count consecutive sign changes in ΔE — a measure of charge sloshing."""
    if len(energies) < 3:
        return 0
    sign_changes = 0
    dEs = np.diff(energies)
    for i in range(1, len(dEs)):
        if dEs[i] * dEs[i - 1] < 0:
            sign_changes += 1
    return sign_changes


def main():
    print("=" * 70)
    print("  CHARGE-SLOSHING BENCHMARK")
    print("  Comparing mixing algorithms on a metal-like 3H chain")
    print(f"  Using {'C++ extension' if _USING_CPP_EXT else 'pure-Python fallback'}")
    print("=" * 70)

    configs = [
        ("Linear (α=0.3, the OLD default)",  "linear",  {"mixing_alpha": 0.30,
                                                         "use_kerker_preconditioner": False}),
        ("Linear (α=0.1, extra-damped)",     "linear",  {"mixing_alpha": 0.10,
                                                         "use_kerker_preconditioner": False}),
        ("Broyden + Kerker (NEW default)",   "broyden", {"mixing_alpha": 0.40,
                                                         "use_kerker_preconditioner": True,
                                                         "kerker_kappa": 0.8}),
        ("DIIS / CDIIS",                     "diis",    {"mixing_alpha": 0.80,
                                                         "use_kerker_preconditioner": False}),
    ]

    print(f"\n{'Method':<45s} {'Steps':>5s}  {'Conv':>4s}  "
          f"{'E_final (Ha)':>14s}  {'#Osc':>5s}  {'Final ||r||':>12s}")
    print("-" * 100)

    all_results = {}

    for label, method, extra in configs:
        try:
            res = run_one(method=method,
                          mixing_method=method,
                          ecut=5.0, a_box=10.0,
                          max_steps=15, smearing=0.08,
                          **extra)
            osc = count_oscillations(res.energies_per_iter)
            final_r = (res.residual_norms[-1]
                       if res.residual_norms else float("nan"))
            status = "✓" if res.converged else "✗"
            print(f"{label:<45s} {res.n_iter:>5d}  {status:>4s}  "
                  f"{res.total_energy:>14.6f}  {osc:>5d}  {final_r:>12.3e}")
            all_results[label] = res
        except Exception as exc:
            print(f"{label:<45s} FAILED: {exc}")
            all_results[label] = None

    # -------- detailed diagnostics for the sloshing comparison ----------
    print("\n" + "=" * 70)
    print("  Energy vs iteration (linear α=0.3 vs Broyden+Kerker):")
    print("=" * 70)

    labels_compare = [configs[0][0], configs[2][0]]
    max_len = 0
    for lab in labels_compare:
        r = all_results.get(lab)
        if r is not None:
            max_len = max(max_len, len(r.energies_per_iter))

    print(f"\n{'Step':>5s}", end="")
    for lab in labels_compare:
        short = lab.split()[0]
        print(f"  {short:>16s}", end="")
    print()
    print("-" * (5 + 18 * len(labels_compare)))

    for step in range(1, max_len + 1):
        print(f"{step:>5d}", end="")
        for lab in labels_compare:
            r = all_results.get(lab)
            if r is not None and step - 1 < len(r.energies_per_iter):
                print(f"  {r.energies_per_iter[step - 1]:>16.6f}", end="")
            else:
                print(f"  {'---':>16s}", end="")
        print()

    print("\n" + "=" * 70)
    print("  Interpretation")
    print("=" * 70)
    print("""
    CHARGE-SLOSHING SIGNATURE:
      Linear(α=0.3) should show oscillating + / - ΔE pattern with
      alternating step directions. This is the exact instability you
      reported for metals: the long-wavelength (small |G|) Hartree
      modes overcorrect back and forth.

    BROYDEN + KERKER SUCCESS CRITERIA:
      (1) Fewer or zero oscillation sign-changes in ΔE.
      (2) Steadily decreasing residual ||r||.
      (3) Convergence in fewer steps than Linear(α=0.1) even though
          the effective α is ~0.4 (i.e. *faster* AND *more stable*).

    PHYSICS EXPLANATION:
      The Kerker preconditioner P_G = G²/(G²+κ²) analytically
      cancels the 4π/G² divergence in the Hartree response at small
      |G|.  Combined with Broyden's inverse-Jacobian estimate, the
      unstable eigenvector of the SCF fixed-point map is rotated out
      of the iteration within ~4 history steps, so the 2-step
      oscillation dies out almost immediately.
""")


if __name__ == "__main__":
    main()
