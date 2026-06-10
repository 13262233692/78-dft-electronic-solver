#pragma once

#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace dft_solver {

using Real = double;
using Complex = std::complex<Real>;
using Vector3r = Eigen::Matrix<Real, 3, 1>;
using Vector3i = Eigen::Matrix<int, 3, 1>;
using Matrix3r = Eigen::Matrix<Real, 3, 3>;
using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using SparseMatrixXc = Eigen::SparseMatrix<Complex, Eigen::RowMajor>;
using SparseMatrixXr = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Tripletc = Eigen::Triplet<Complex>;

struct GVector {
    Vector3i miller;
    Vector3r cartesian;
    Real norm2;

    GVector() : miller(Vector3i::Zero()), cartesian(Vector3r::Zero()), norm2(0.0) {}
    GVector(const Vector3i& m, const Vector3r& c)
        : miller(m), cartesian(c), norm2(c.squaredNorm()) {}
};

}
