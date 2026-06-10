#include "dft_solver/hamiltonian.h"
#include <cmath>
#include <stdexcept>

namespace dft_solver {

Hamiltonian::Hamiltonian(PlaneWaveBasisPtr basis, AtomsPtr atoms)
    : basis_(basis), atoms_(atoms), ewald_energy_(0.0) {
    int npw_local = basis_->npw();
    v_ion_g_.resize(npw_local);
    v_ion_g_.setZero();
    v_hartree_g_.resize(npw_local);
    v_hartree_g_.setZero();
    v_xc_g_.resize(npw_local);
    v_xc_g_.setZero();

    build_ionic_potential();
    compute_ewald_energy();
}

std::vector<GVector> Hamiltonian::get_gk_vectors(const Vector3r& k) const {
    return basis_->g_vectors_shifted(k);
}

int Hamiltonian::npw_k(const Vector3r& kpoint) const {
    return basis_->npw();
}

void Hamiltonian::update_density(DensityPtr density) {
    density_ = density;
    v_hartree_g_ = density->hartree_potential_g();
    v_xc_g_ = density->v_xc_g();
    compute_potential_fourier();
}

void Hamiltonian::compute_potential_fourier() {
}

void Hamiltonian::build_ionic_potential() {
    int npw_local = basis_->npw();
    const auto& gs = basis_->g_vectors();
    const auto& pspots = atoms_->pseudopotentials();

    for (int ig = 0; ig < npw_local; ++ig) {
        Complex struct_fac(0.0, 0.0);
        Real G_norm = std::sqrt(gs[ig].norm2);

        for (const auto& atom : atoms_->atoms()) {
            Real phase = gs[ig].cartesian.dot(atom.position);
            struct_fac += Complex(std::cos(phase), -std::sin(phase));
        }

        if (!pspots.empty()) {
            auto pspot = pspots.begin()->second;
            v_ion_g_[ig] = struct_fac * pspot->v_local_of_g(G_norm) /
                           basis_->cell_volume();
        }
    }
}

void Hamiltonian::compute_ewald_energy() {
    Real eta = 1.0;
    Real volume = basis_->cell_volume();
    int natoms = atoms_->natoms();
    Real ewald = 0.0;

    for (int i = 0; i < natoms; ++i) {
        for (int j = 0; j < natoms; ++j) {
            int Zi = atoms_->atom(i).atomic_number;
            int Zj = atoms_->atom(j).atomic_number;
            Vector3r rij = atoms_->atom(i).position - atoms_->atom(j).position;

            if (i == j) {
                ewald -= Zi * Zj * std::sqrt(eta / M_PI);
            } else {
                Real r = rij.norm();
                if (r > 1e-8) {
                    ewald += 0.5 * Zi * Zj * std::erfc(std::sqrt(eta) * r) / r;
                }
            }
        }
    }

    int ng_max = 20;
    auto b1 = basis_->reciprocal_cell().col(0);
    auto b2 = basis_->reciprocal_cell().col(1);
    auto b3 = basis_->reciprocal_cell().col(2);

    for (int m1 = -ng_max; m1 <= ng_max; ++m1) {
        for (int m2 = -ng_max; m2 <= ng_max; ++m2) {
            for (int m3 = -ng_max; m3 <= ng_max; ++m3) {
                if (m1 == 0 && m2 == 0 && m3 == 0) continue;
                Vector3r G = m1 * b1 + m2 * b2 + m3 * b3;
                Real G2 = G.squaredNorm();
                Real factor = 4.0 * M_PI / volume * std::exp(-G2 / (4.0 * eta)) / G2;

                Complex S(0.0, 0.0);
                for (int i = 0; i < natoms; ++i) {
                    int Zi = atoms_->atom(i).atomic_number;
                    Real phase = G.dot(atoms_->atom(i).position);
                    S += static_cast<Real>(Zi) * Complex(std::cos(phase), -std::sin(phase));
                }
                ewald += 0.5 * factor * std::norm(S);
            }
        }
    }

    ewald_energy_ = ewald;
}

Real Hamiltonian::ewald_energy() const {
    return ewald_energy_;
}

Real Hamiltonian::v_kinetic(int ig, const Vector3r& kpoint) const {
    const auto& gs = basis_->g_vectors();
    Vector3r gk = gs[ig].cartesian + kpoint;
    return 0.5 * gk.squaredNorm();
}

Complex Hamiltonian::v_local(int ig1, int ig2, const Vector3r& kpoint) const {
    const auto& gs = basis_->g_vectors();
    Vector3i diff_miller = gs[ig1].miller - gs[ig2].miller;
    int idx = basis_->find_g_vector_index(diff_miller);

    if (idx < 0) return Complex(0.0, 0.0);

    Complex result = v_ion_g_[idx];
    if (density_) {
        result += v_hartree_g_[idx] + v_xc_g_[idx];
    }
    return result;
}

VectorXc Hamiltonian::v_local_diag(const Vector3r& kpoint) const {
    int npw_local = basis_->npw();
    VectorXc diag(npw_local);
    diag.setZero();

    for (int ig = 0; ig < npw_local; ++ig) {
        diag[ig] = v_local(ig, ig, kpoint);
    }
    return diag;
}

SparseMatrixXc Hamiltonian::v_local_sparse(const Vector3r& kpoint) const {
    int npw_local = basis_->npw();
    SparseMatrixXc mat(npw_local, npw_local);
    std::vector<Tripletc> triplets;
    triplets.reserve(npw_local * 50);

    for (int ig1 = 0; ig1 < npw_local; ++ig1) {
        for (int ig2 = 0; ig2 < npw_local; ++ig2) {
            Complex v = v_local(ig1, ig2, kpoint);
            if (std::abs(v) > 1e-15) {
                triplets.emplace_back(ig1, ig2, v);
            }
        }
    }

    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

SparseMatrixXc Hamiltonian::build_matrix(const Vector3r& kpoint) const {
    int npw_local = basis_->npw();
    SparseMatrixXc mat(npw_local, npw_local);
    std::vector<Tripletc> triplets;
    triplets.reserve(npw_local * 100);

    const auto& gs = basis_->g_vectors();

    for (int ig = 0; ig < npw_local; ++ig) {
        Real t = v_kinetic(ig, kpoint);
        triplets.emplace_back(ig, ig, Complex(t, 0.0));
    }

    for (int ig1 = 0; ig1 < npw_local; ++ig1) {
        for (int ig2 = 0; ig2 < npw_local; ++ig2) {
            if (ig1 == ig2) {
                Complex v = v_local(ig1, ig2, kpoint);
                triplets.emplace_back(ig1, ig2, v);
            } else {
                Complex v = v_local(ig1, ig2, kpoint);
                if (std::abs(v) > 1e-15) {
                    triplets.emplace_back(ig1, ig2, v);
                }
            }
        }
    }

    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

MatrixXc Hamiltonian::build_dense_matrix(const Vector3r& kpoint) const {
    int npw_local = basis_->npw();
    MatrixXc mat(npw_local, npw_local);
    mat.setZero();

    const auto& gs = basis_->g_vectors();

    for (int ig = 0; ig < npw_local; ++ig) {
        mat(ig, ig) = Complex(v_kinetic(ig, kpoint), 0.0);
    }

    for (int ig1 = 0; ig1 < npw_local; ++ig1) {
        for (int ig2 = 0; ig2 < npw_local; ++ig2) {
            mat(ig1, ig2) += v_local(ig1, ig2, kpoint);
        }
    }

    return mat;
}

VectorXc Hamiltonian::apply(const VectorXc& psi, const Vector3r& kpoint) const {
    int npw_local = basis_->npw();
    VectorXc result(npw_local);
    apply_inplace(psi, result, kpoint);
    return result;
}

void Hamiltonian::apply_inplace(const VectorXc& psi_in, VectorXc& psi_out,
                                 const Vector3r& kpoint) const {
    int npw_local = basis_->npw();
    psi_out.resize(npw_local);
    psi_out.setZero();

    const auto& gs = basis_->g_vectors();

    for (int ig = 0; ig < npw_local; ++ig) {
        psi_out[ig] = v_kinetic(ig, kpoint) * psi_in[ig];
    }

    int ngx = basis_->ngx();
    int ngy = basis_->ngy();
    int ngz = basis_->ngz();
    int nr = ngx * ngy * ngz;

    VectorXc psi_r(nr);
    psi_r.setZero();

    auto a1 = basis_->cell().col(0);
    auto a2 = basis_->cell().col(1);
    auto a3 = basis_->cell().col(2);

    for (int ix = 0; ix < ngx; ++ix) {
        for (int iy = 0; iy < ngy; ++iy) {
            for (int iz = 0; iz < ngz; ++iz) {
                int ir = ix * ngy * ngz + iy * ngz + iz;
                Vector3r r = (static_cast<Real>(ix) / ngx) * a1 +
                             (static_cast<Real>(iy) / ngy) * a2 +
                             (static_cast<Real>(iz) / ngz) * a3;
                Complex sum(0.0, 0.0);
                for (int ig = 0; ig < npw_local; ++ig) {
                    Vector3r gk = gs[ig].cartesian + kpoint;
                    Real phase = gk.dot(r);
                    sum += psi_in[ig] * Complex(std::cos(phase), std::sin(phase));
                }
                psi_r[ir] = sum;
            }
        }
    }

    VectorXr v_total_r(nr);
    v_total_r.setZero();

    if (density_) {
        v_total_r = density_->hartree_potential_r() + density_->v_xc_r();
    }

    for (int ir = 0; ir < nr; ++ir) {
        psi_r[ir] *= v_total_r[ir];
    }

    for (int ig = 0; ig < npw_local; ++ig) {
        Vector3r gk = gs[ig].cartesian + kpoint;
        Complex sum(0.0, 0.0);
        for (int ix = 0; ix < ngx; ++ix) {
            for (int iy = 0; iy < ngy; ++iy) {
                for (int iz = 0; iz < ngz; ++iz) {
                    int ir = ix * ngy * ngz + iy * ngz + iz;
                    Vector3r r = (static_cast<Real>(ix) / ngx) * a1 +
                                 (static_cast<Real>(iy) / ngy) * a2 +
                                 (static_cast<Real>(iz) / ngz) * a3;
                    Real phase = -gk.dot(r);
                    sum += psi_r[ir] * Complex(std::cos(phase), std::sin(phase));
                }
            }
        }
        psi_out[ig] += sum / nr;
    }

    for (int ig1 = 0; ig1 < npw_local; ++ig1) {
        for (int ig2 = 0; ig2 < npw_local; ++ig2) {
            Vector3i diff_miller = gs[ig1].miller - gs[ig2].miller;
            int idx = basis_->find_g_vector_index(diff_miller);
            if (idx >= 0 && std::abs(v_ion_g_[idx]) > 1e-15) {
                psi_out[ig1] += v_ion_g_[idx] * psi_in[ig2];
            }
        }
    }
}

Real Hamiltonian::compute_kinetic_energy(const std::vector<VectorXc>& psi,
                                          const std::vector<Real>& occ,
                                          const Vector3r& kpoint) const {
    Real ekin = 0.0;
    int npw_local = basis_->npw();
    const auto& gs = basis_->g_vectors();

    for (size_t ib = 0; ib < psi.size(); ++ib) {
        const auto& psi_b = psi[ib];
        Real f = occ[ib];
        for (int ig = 0; ig < npw_local; ++ig) {
            Vector3r gk = gs[ig].cartesian + kpoint;
            ekin += f * 0.5 * gk.squaredNorm() * std::norm(psi_b[ig]);
        }
    }
    return ekin * basis_->cell_volume();
}

}
