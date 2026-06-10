#include "dft_solver/density.h"
#include <cmath>
#include <stdexcept>

namespace dft_solver {

Density::Density(PlaneWaveBasisPtr basis)
    : basis_(basis), n_electrons_(0.0) {
    rho_g_.resize(basis->npw());
    rho_g_.setZero();
}

void Density::set_from_r_space(const VectorXr& rho_r) {
    fft_r_to_g(rho_r, rho_g_);
    normalize();
}

void Density::set_from_eigenstates(const std::vector<VectorXc>& psi,
                                    const std::vector<Real>& occupations,
                                    const Vector3r& kpoint) {
    int npw_local = basis_->npw();
    int nbands = static_cast<int>(psi.size());

    rho_g_.setZero();

    auto gk = basis_->g_vectors_shifted(kpoint);

    for (int ib = 0; ib < nbands; ++ib) {
        const auto& psi_b = psi[ib];
        Real f = occupations[ib];

        for (int ig1 = 0; ig1 < npw_local; ++ig1) {
            Complex c1 = std::conj(psi_b[ig1]);
            for (int ig2 = 0; ig2 < npw_local; ++ig2) {
                Complex c2 = psi_b[ig2];
                Vector3i diff_miller = gk[ig1].miller - gk[ig2].miller;
                int idx = basis_->find_g_vector_index(diff_miller);
                if (idx >= 0) {
                    rho_g_[idx] += f * c1 * c2;
                }
            }
        }
    }

    Real volume = basis_->cell_volume();
    rho_g_ /= volume;
    normalize();
}

void Density::normalize() {
    if (n_electrons_ > 0) {
        Real current_ne = rho_g_[0].real() * basis_->cell_volume();
        if (std::abs(current_ne) > 1e-15) {
            rho_g_ *= n_electrons_ / current_ne;
        }
    }
}

VectorXr Density::rho_r() const {
    int nr = basis_->ngx() * basis_->ngy() * basis_->ngz();
    VectorXr result(nr);
    fft_g_to_r(rho_g_, result);
    return result;
}

Real Density::total_electrons() const {
    return rho_g_[0].real() * basis_->cell_volume();
}

VectorXc Density::hartree_potential_g() const {
    int npw_local = basis_->npw();
    VectorXc v_h(npw_local);
    v_h.setZero();

    const auto& gs = basis_->g_vectors();
    for (int ig = 0; ig < npw_local; ++ig) {
        Real g2 = gs[ig].norm2;
        if (g2 > 1e-12) {
            v_h[ig] = 4.0 * M_PI * rho_g_[ig] / g2;
        }
    }
    return v_h;
}

VectorXr Density::hartree_potential_r() const {
    auto v_h_g = hartree_potential_g();
    int nr = basis_->ngx() * basis_->ngy() * basis_->ngz();
    VectorXr result(nr);
    fft_g_to_r(v_h_g, result);
    return result;
}

Real Density::exchange_correlation_energy(Real rho) {
    if (rho < 1e-20) return 0.0;
    Real rs = std::pow(3.0 / (4.0 * M_PI * rho), 1.0 / 3.0);
    Real ex = -3.0 / (4.0 * M_PI) * std::pow(9.0 * M_PI / 4.0, 1.0 / 3.0) / rs;
    Real A = 0.0311, B = -0.048, C = 0.0020, D = -0.0116;
    Real ec;
    if (rs >= 1.0) {
        Real gamma = -0.1423, beta1 = 1.0529, beta2 = 0.3334;
        ec = gamma / (1.0 + beta1 * std::sqrt(rs) + beta2 * rs);
    } else {
        ec = A * std::log(rs) + B + C * rs * std::log(rs) + D * rs;
    }
    return ex + ec;
}

Real Density::exchange_correlation_potential(Real rho) {
    if (rho < 1e-20) return 0.0;
    Real rs = std::pow(3.0 / (4.0 * M_PI * rho), 1.0 / 3.0);
    Real drho_drs = -3.0 * std::pow(3.0 / (4.0 * M_PI), 1.0 / 3.0) *
                    std::pow(rs, -4.0 / 3.0) / 3.0;

    Real ex = -3.0 / (4.0 * M_PI) * std::pow(9.0 * M_PI / 4.0, 1.0 / 3.0) / rs;
    Real dex_drs = 3.0 / (4.0 * M_PI) * std::pow(9.0 * M_PI / 4.0, 1.0 / 3.0) / (rs * rs);

    Real ec, dec_drs;
    if (rs >= 1.0) {
        Real gamma = -0.1423, beta1 = 1.0529, beta2 = 0.3334;
        Real denom = 1.0 + beta1 * std::sqrt(rs) + beta2 * rs;
        ec = gamma / denom;
        dec_drs = -gamma * (beta1 / (2.0 * std::sqrt(rs)) + beta2) / (denom * denom);
    } else {
        Real A = 0.0311, B = -0.048, C = 0.0020, D = -0.0116;
        ec = A * std::log(rs) + B + C * rs * std::log(rs) + D * rs;
        dec_drs = A / rs + C * (std::log(rs) + 1.0) + D;
    }

    Real dexc_drs = dex_drs + dec_drs;
    return ex + ec - rho * dexc_drs / drho_drs;
}

VectorXr Density::v_xc_r() const {
    auto rho = rho_r();
    VectorXr vxc(rho.size());
    for (int i = 0; i < rho.size(); ++i) {
        vxc[i] = exchange_correlation_potential(rho[i]);
    }
    return vxc;
}

VectorXc Density::v_xc_g() const {
    auto vxc_r = v_xc_r();
    int npw_local = basis_->npw();
    VectorXc vxc_g(npw_local);
    fft_r_to_g(vxc_r, vxc_g);
    return vxc_g;
}

Real Density::e_xc() const {
    auto rho = rho_r();
    Real volume = basis_->cell_volume();
    int nr = rho.size();
    Real dV = volume / nr;
    Real exc = 0.0;
    for (int i = 0; i < nr; ++i) {
        exc += rho[i] * exchange_correlation_energy(rho[i]) * dV;
    }
    return exc;
}

Real Density::e_hartree() const {
    int npw_local = basis_->npw();
    Real volume = basis_->cell_volume();
    Real eh = 0.0;
    const auto& gs = basis_->g_vectors();
    for (int ig = 1; ig < npw_local; ++ig) {
        Real g2 = gs[ig].norm2;
        Complex rho = rho_g_[ig];
        eh += 2.0 * M_PI * std::norm(rho) / g2;
    }
    return eh * volume;
}

void Density::fft_r_to_g(const VectorXr& r_space, VectorXc& g_space) const {
    int ngx = basis_->ngx();
    int ngy = basis_->ngy();
    int ngz = basis_->ngz();
    int nr = ngx * ngy * ngz;
    int npw_local = basis_->npw();

    auto b1 = basis_->reciprocal_cell().col(0);
    auto b2 = basis_->reciprocal_cell().col(1);
    auto b3 = basis_->reciprocal_cell().col(2);
    auto a1 = basis_->cell().col(0);
    auto a2 = basis_->cell().col(1);
    auto a3 = basis_->cell().col(2);

    const auto& gs = basis_->g_vectors();

    for (int ig = 0; ig < npw_local; ++ig) {
        Complex sum(0.0, 0.0);
        for (int ix = 0; ix < ngx; ++ix) {
            for (int iy = 0; iy < ngy; ++iy) {
                for (int iz = 0; iz < ngz; ++iz) {
                    int ir = ix * ngy * ngz + iy * ngz + iz;
                    Vector3r r = (static_cast<Real>(ix) / ngx) * a1 +
                                 (static_cast<Real>(iy) / ngy) * a2 +
                                 (static_cast<Real>(iz) / ngz) * a3;
                    Real phase = gs[ig].cartesian.dot(r);
                    sum += r_space[ir] * Complex(std::cos(phase), -std::sin(phase));
                }
            }
        }
        g_space[ig] = sum / nr;
    }
}

void Density::fft_g_to_r(const VectorXc& g_space, VectorXr& r_space) const {
    int ngx = basis_->ngx();
    int ngy = basis_->ngy();
    int ngz = basis_->ngz();
    int nr = ngx * ngy * ngz;
    int npw_local = basis_->npw();

    auto a1 = basis_->cell().col(0);
    auto a2 = basis_->cell().col(1);
    auto a3 = basis_->cell().col(2);

    const auto& gs = basis_->g_vectors();

    for (int ix = 0; ix < ngx; ++ix) {
        for (int iy = 0; iy < ngy; ++iy) {
            for (int iz = 0; iz < ngz; ++iz) {
                int ir = ix * ngy * ngz + iy * ngz + iz;
                Vector3r r = (static_cast<Real>(ix) / ngx) * a1 +
                             (static_cast<Real>(iy) / ngy) * a2 +
                             (static_cast<Real>(iz) / ngz) * a3;
                Complex sum(0.0, 0.0);
                for (int ig = 0; ig < npw_local; ++ig) {
                    Real phase = gs[ig].cartesian.dot(r);
                    sum += g_space[ig] * Complex(std::cos(phase), std::sin(phase));
                }
                r_space[ir] = sum.real();
            }
        }
    }
}

}
