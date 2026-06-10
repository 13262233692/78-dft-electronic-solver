#pragma once

#include "dft_solver/types.h"
#include <vector>
#include <memory>

namespace dft_solver {

class PlaneWaveBasis {
public:
    PlaneWaveBasis(const Matrix3r& cell, Real ecut);

    const Matrix3r& cell() const { return cell_; }
    const Matrix3r& reciprocal_cell() const { return reciprocal_cell_; }
    Real ecut() const { return ecut_; }
    Real cell_volume() const { return cell_volume_; }
    int npw() const { return static_cast<int>(g_vectors_.size()); }

    const std::vector<GVector>& g_vectors() const { return g_vectors_; }
    const GVector& g_vector(int i) const { return g_vectors_[i]; }

    int find_g_vector_index(const Vector3i& miller) const;

    VectorXc plane_waves(const Vector3r& k, const std::vector<Vector3r>& r) const;

    std::vector<GVector> g_vectors_shifted(const Vector3r& k) const;

    int ngx() const { return ngx_; }
    int ngy() const { return ngy_; }
    int ngz() const { return ngz_; }

private:
    Matrix3r cell_;
    Matrix3r reciprocal_cell_;
    Real ecut_;
    Real cell_volume_;
    std::vector<GVector> g_vectors_;
    int ngx_, ngy_, ngz_;

    void build_reciprocal_cell();
    void build_g_vectors();
};

using PlaneWaveBasisPtr = std::shared_ptr<PlaneWaveBasis>;

}
