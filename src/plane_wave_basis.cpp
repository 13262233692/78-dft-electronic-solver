#include "dft_solver/plane_wave_basis.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace dft_solver {

PlaneWaveBasis::PlaneWaveBasis(const Matrix3r& cell, Real ecut)
    : cell_(cell), ecut_(ecut) {
    build_reciprocal_cell();
    build_g_vectors();
}

void PlaneWaveBasis::build_reciprocal_cell() {
    cell_volume_ = std::abs(cell_.determinant());
    const Real two_pi = 2.0 * M_PI;

    Vector3r a1 = cell_.col(0);
    Vector3r a2 = cell_.col(1);
    Vector3r a3 = cell_.col(2);

    Vector3r b1 = two_pi * a2.cross(a3) / cell_volume_;
    Vector3r b2 = two_pi * a3.cross(a1) / cell_volume_;
    Vector3r b3 = two_pi * a1.cross(a2) / cell_volume_;

    reciprocal_cell_.col(0) = b1;
    reciprocal_cell_.col(1) = b2;
    reciprocal_cell_.col(2) = b3;
}

void PlaneWaveBasis::build_g_vectors() {
    const Real two_ecut = 2.0 * ecut_;
    const Real gmax = std::sqrt(two_ecut);

    Vector3r b1 = reciprocal_cell_.col(0);
    Vector3r b2 = reciprocal_cell_.col(1);
    Vector3r b3 = reciprocal_cell_.col(2);

    ngx_ = static_cast<int>(std::ceil(gmax / b1.norm())) + 1;
    ngy_ = static_cast<int>(std::ceil(gmax / b2.norm())) + 1;
    ngz_ = static_cast<int>(std::ceil(gmax / b3.norm())) + 1;

    g_vectors_.clear();

    for (int i = -ngx_; i <= ngx_; ++i) {
        for (int j = -ngy_; j <= ngy_; ++j) {
            for (int k = -ngz_; k <= ngz_; ++k) {
                Vector3i miller(i, j, k);
                Vector3r cart = i * b1 + j * b2 + k * b3;
                Real norm2 = cart.squaredNorm();
                if (norm2 <= two_ecut) {
                    g_vectors_.emplace_back(miller, cart);
                }
            }
        }
    }

    std::sort(g_vectors_.begin(), g_vectors_.end(),
              [](const GVector& a, const GVector& b) {
                  return a.norm2 < b.norm2;
              });
}

int PlaneWaveBasis::find_g_vector_index(const Vector3i& miller) const {
    for (size_t i = 0; i < g_vectors_.size(); ++i) {
        if (g_vectors_[i].miller == miller) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

VectorXc PlaneWaveBasis::plane_waves(const Vector3r& k,
                                      const std::vector<Vector3r>& r) const {
    int nr = static_cast<int>(r.size());
    int npw_local = npw();
    VectorXc result(nr * npw_local);

    for (int ir = 0; ir < nr; ++ir) {
        for (int ig = 0; ig < npw_local; ++ig) {
            Vector3r gk = g_vectors_[ig].cartesian + k;
            Real phase = gk.dot(r[ir]);
            result[ir * npw_local + ig] = Complex(std::cos(phase), std::sin(phase));
        }
    }
    return result;
}

std::vector<GVector> PlaneWaveBasis::g_vectors_shifted(const Vector3r& k) const {
    std::vector<GVector> result;
    result.reserve(g_vectors_.size());
    for (const auto& g : g_vectors_) {
        Vector3r cart = g.cartesian + k;
        GVector gk(g.miller, cart);
        gk.norm2 = cart.squaredNorm();
        result.push_back(gk);
    }
    return result;
}

}
