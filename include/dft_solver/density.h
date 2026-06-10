#pragma once

#include "dft_solver/types.h"
#include "dft_solver/plane_wave_basis.h"
#include <vector>
#include <memory>

namespace dft_solver {

class Density {
public:
    Density(PlaneWaveBasisPtr basis);

    PlaneWaveBasisPtr basis() const { return basis_; }

    void set_from_eigenstates(const std::vector<VectorXc>& psi,
                              const std::vector<Real>& occupations,
                              const Vector3r& kpoint = Vector3r::Zero());

    void set_from_r_space(const VectorXr& rho_r);

    const VectorXc& rho_g() const { return rho_g_; }
    VectorXr rho_r() const;

    Real total_electrons() const;
    Real n_electrons() const { return n_electrons_; }
    void set_n_electrons(Real n) { n_electrons_ = n; }

    VectorXr hartree_potential_r() const;
    VectorXc hartree_potential_g() const;

    static Real exchange_correlation_energy(Real rho);
    static Real exchange_correlation_potential(Real rho);

    VectorXr v_xc_r() const;
    VectorXc v_xc_g() const;
    Real e_xc() const;

    Real e_hartree() const;

private:
    PlaneWaveBasisPtr basis_;
    VectorXc rho_g_;
    Real n_electrons_;

    void normalize();
    void fft_r_to_g(const VectorXr& r_space, VectorXc& g_space) const;
    void fft_g_to_r(const VectorXc& g_space, VectorXr& r_space) const;
};

using DensityPtr = std::shared_ptr<Density>;

}
