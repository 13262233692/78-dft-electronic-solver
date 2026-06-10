#pragma once

#include "dft_solver/types.h"
#include "dft_solver/plane_wave_basis.h"
#include "dft_solver/atoms.h"
#include "dft_solver/density.h"
#include <vector>
#include <memory>

namespace dft_solver {

class Hamiltonian {
public:
    Hamiltonian(PlaneWaveBasisPtr basis, AtomsPtr atoms);

    PlaneWaveBasisPtr basis() const { return basis_; }
    AtomsPtr atoms() const { return atoms_; }

    void update_density(DensityPtr density);
    DensityPtr density() const { return density_; }

    SparseMatrixXc build_matrix(const Vector3r& kpoint) const;

    MatrixXc build_dense_matrix(const Vector3r& kpoint) const;

    VectorXc apply(const VectorXc& psi, const Vector3r& kpoint) const;

    void apply_inplace(const VectorXc& psi_in, VectorXc& psi_out,
                       const Vector3r& kpoint) const;

    int npw_k(const Vector3r& kpoint) const;

    Real ewald_energy() const;

    Real v_kinetic(int ig, const Vector3r& kpoint) const;

    Complex v_local(int ig1, int ig2, const Vector3r& kpoint) const;

    VectorXc v_local_diag(const Vector3r& kpoint) const;

    SparseMatrixXc v_local_sparse(const Vector3r& kpoint) const;

    void compute_potential_fourier();

    const VectorXc& v_ion_g() const { return v_ion_g_; }

    Real compute_kinetic_energy(const std::vector<VectorXc>& psi,
                                const std::vector<Real>& occ,
                                const Vector3r& kpoint) const;

private:
    PlaneWaveBasisPtr basis_;
    AtomsPtr atoms_;
    DensityPtr density_;
    VectorXc v_ion_g_;
    VectorXc v_hartree_g_;
    VectorXc v_xc_g_;
    Real ewald_energy_;

    void build_ionic_potential();
    void compute_ewald_energy();

    std::vector<GVector> get_gk_vectors(const Vector3r& k) const;
};

using HamiltonianPtr = std::shared_ptr<Hamiltonian>;

}
