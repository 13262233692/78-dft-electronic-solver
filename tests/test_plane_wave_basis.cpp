#include "dft_solver/plane_wave_basis.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dft_solver;

int main() {
    std::cout << "=== Testing PlaneWaveBasis ===" << std::endl;

    Matrix3r cell;
    cell << 10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0;

    Real ecut = 10.0;
    PlaneWaveBasis basis(cell, ecut);

    std::cout << "Cell volume: " << basis.cell_volume() << std::endl;
    std::cout << "Number of plane waves: " << basis.npw() << std::endl;
    std::cout << "FFT grid: " << basis.ngx() << " x " << basis.ngy()
              << " x " << basis.ngz() << std::endl;

    assert(basis.cell_volume() > 0);
    assert(basis.npw() > 0);

    auto rc = basis.reciprocal_cell();
    Real two_pi = 2.0 * M_PI;
    assert(std::abs(rc(0, 0) - two_pi / 10.0) < 1e-10);
    assert(std::abs(rc(1, 1) - two_pi / 10.0) < 1e-10);
    assert(std::abs(rc(2, 2) - two_pi / 10.0) < 1e-10);

    auto gs = basis.g_vectors();
    bool sorted = true;
    for (size_t i = 1; i < gs.size(); ++i) {
        if (gs[i].norm2 < gs[i-1].norm2) {
            sorted = false;
            break;
        }
    }
    assert(sorted);
    std::cout << "G-vectors are sorted by norm2: OK" << std::endl;

    Vector3i test_miller(0, 0, 0);
    int idx = basis.find_g_vector_index(test_miller);
    assert(idx >= 0);
    std::cout << "Gamma point found at index: " << idx << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
