#include "dft_solver/atoms.h"
#include <cmath>
#include <stdexcept>
#include <map>

namespace dft_solver {

static const std::map<std::string, int> ATOMIC_NUMBERS = {
    {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6},
    {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10}, {"Na", 11}, {"Mg", 12},
    {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18},
    {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24},
    {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
    {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36}
};

Pseudopotential::Pseudopotential(int Z, const std::vector<Real>& r_grid,
                                 const std::vector<Real>& v_local,
                                 Real r_cutoff)
    : Z_(Z), r_grid_(r_grid), v_local_(v_local), r_cutoff_(r_cutoff) {
    build_spline();
}

void Pseudopotential::build_spline() {
    int n = static_cast<int>(r_grid_.size());
    spline_coeffs_.resize(4 * (n - 1), 0.0);

    if (n < 2) return;

    for (int i = 0; i < n - 1; ++i) {
        Real h = r_grid_[i + 1] - r_grid_[i];
        Real y0 = v_local_[i];
        Real y1 = v_local_[i + 1];
        spline_coeffs_[4 * i] = y0;
        spline_coeffs_[4 * i + 1] = (y1 - y0) / h;
        spline_coeffs_[4 * i + 2] = 0.0;
        spline_coeffs_[4 * i + 3] = 0.0;
    }
}

Complex Pseudopotential::v_local_of_g(Real G_norm) const {
    if (G_norm < 1e-10) {
        Real integral = 0.0;
        for (size_t i = 0; i < r_grid_.size() - 1; ++i) {
            Real r0 = r_grid_[i];
            Real r1 = r_grid_[i + 1];
            Real v0 = v_local_[i];
            Real v1 = v_local_[i + 1];
            integral += 0.5 * (r1 - r0) * (v0 * r0 * r0 + v1 * r1 * r1);
        }
        return 4.0 * M_PI * integral;
    }

    Real integral = 0.0;
    for (size_t i = 0; i < r_grid_.size() - 1; ++i) {
        Real r0 = r_grid_[i];
        Real r1 = r_grid_[i + 1];
        Real dr = r1 - r0;
        Real g0 = G_norm * r0;
        Real g1 = G_norm * r1;

        Real f0 = v_local_[i] * r0 * std::sin(g0) / G_norm;
        Real f1 = v_local_[i + 1] * r1 * std::sin(g1) / G_norm;
        integral += 0.5 * dr * (f0 + f1);
    }

    return 4.0 * M_PI * integral;
}

std::shared_ptr<Pseudopotential> Pseudopotential::create(int Z) {
    std::vector<Real> r_grid;
    std::vector<Real> v_local;

    Real r_max = 5.0;
    int n_points = 200;
    Real dr = r_max / n_points;

    for (int i = 0; i <= n_points; ++i) {
        Real r = (i + 1) * dr;
        r_grid.push_back(r);
        Real r_c = 1.5;
        Real v = -static_cast<Real>(Z) / r * std::erf(r / (std::sqrt(2.0) * r_c));
        v_local.push_back(v);
    }

    return std::make_shared<Pseudopotential>(Z, r_grid, v_local, 2.0);
}

Atoms::Atoms(const Matrix3r& cell)
    : cell_(cell), nelectrons_(0) {}

int Atoms::atomic_number(const std::string& symbol) const {
    auto it = ATOMIC_NUMBERS.find(symbol);
    if (it == ATOMIC_NUMBERS.end()) {
        throw std::runtime_error("Unknown element: " + symbol);
    }
    return it->second;
}

void Atoms::add_atom(const std::string& symbol, const Vector3r& position) {
    int Z = atomic_number(symbol);
    atoms_.emplace_back(symbol, Z, position);
    nelectrons_ += Z;

    if (pspots_.find(symbol) == pspots_.end()) {
        pspots_[symbol] = Pseudopotential::create(Z);
    }
}

PseudopotentialPtr Atoms::pseudopotential(int i) const {
    return pseudopotential(atoms_[i].symbol);
}

PseudopotentialPtr Atoms::pseudopotential(const std::string& symbol) const {
    auto it = pspots_.find(symbol);
    if (it == pspots_.end()) {
        throw std::runtime_error("No pseudopotential for: " + symbol);
    }
    return it->second;
}

}
