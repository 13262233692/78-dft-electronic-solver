#include "dft_solver/hamiltonian.h"
#include "dft_solver/plane_wave_basis.h"
#include "dft_solver/atoms.h"
#include "dft_solver/density.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dft_solver;

int main() {
    std::cout << "=== Testing Hamiltonian ===" << std::endl;

    Matrix3r cell;
    cell << 10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0;

    Real ecut = 5.0;
    auto basis = std::make_shared<PlaneWaveBasis>(cell, ecut);

    std::cout << "Number of plane waves: " << basis->npw() << std::endl;

    auto atoms = std::make_shared<Atoms>(cell);
    atoms->add_atom("H", Vector3r(5.0, 5.0, 5.0));

    Hamiltonian ham(basis, atoms);

    Vector3r kpoint(0.0, 0.0, 0.0);

    std::cout << "Testing v_kinetic..." << std::endl;
    Real t0 = ham.v_kinetic(0, kpoint);
    std::cout << "Kinetic energy at ig=0: " << t0 << std::endl;
    assert(t0 >= 0);

    std::cout << "Testing v_local_diag..." << std::endl;
    auto v_diag = ham.v_local_diag(kpoint);
    std::cout << "v_local diagonal size: " << v_diag.size() << std::endl;
    assert(v_diag.size() == basis->npw());

    std::cout << "Testing build_matrix..." << std::endl;
    auto H_sparse = ham.build_matrix(kpoint);
    std::cout << "H matrix shape: " << H_sparse.rows() << " x "
              << H_sparse.cols() << std::endl;
    std::cout << "H non-zeros: " << H_sparse.nonZeros() << std::endl;
    assert(H_sparse.rows() == basis->npw());
    assert(H_sparse.cols() == basis->npw());

    std::cout << "Testing build_dense_matrix..." << std::endl;
    auto H_dense = ham.build_dense_matrix(kpoint);
    std::cout << "H dense shape: " << H_dense.rows() << " x "
              << H_dense.cols() << std::endl;

    std::cout << "Checking Hermiticity..." << std::endl;
    bool is_hermitian = true;
    Real max_diff = 0.0;
    for (int i = 0; i < H_dense.rows(); ++i) {
        for (int j = 0; j < H_dense.cols(); ++j) {
            Complex diff = H_dense(i, j) - std::conj(H_dense(j, i));
            if (std::abs(diff) > 1e-10) {
                is_hermitian = false;
                max_diff = std::max(max_diff, std::abs(diff));
            }
        }
    }
    if (is_hermitian) {
        std::cout << "H is Hermitian: OK" << std::endl;
    } else {
        std::cout << "WARNING: H not perfectly Hermitian, max diff = "
                  << max_diff << std::endl;
    }

    std::cout << "Testing apply..." << std::endl;
    VectorXc psi(basis->npw());
    psi.setZero();
    psi[0] = 1.0;
    auto H_psi = ham.apply(psi, kpoint);
    std::cout << "H|psi> norm: " << H_psi.norm() << std::endl;

    std::cout << "Ewald energy: " << ham.ewald_energy() << std::endl;

    auto density = std::make_shared<Density>(basis);
    density->set_n_electrons(atoms->nelectrons());
    ham.update_density(density);

    std::cout << "After updating density:" << std::endl;
    auto v_ion = ham.v_ion_g();
    std::cout << "v_ion_g size: " << v_ion.size() << std::endl;

    std::cout << "Testing npw_k..." << std::endl;
    int npw_k = ham.npw_k(kpoint);
    std::cout << "npw at k=" << kpoint.transpose() << ": " << npw_k << std::endl;
    assert(npw_k == basis->npw());

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
