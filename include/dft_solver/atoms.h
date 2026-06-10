#pragma once

#include "dft_solver/types.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace dft_solver {

struct Atom {
    std::string symbol;
    int atomic_number;
    Vector3r position;

    Atom() : atomic_number(0), position(Vector3r::Zero()) {}
    Atom(const std::string& sym, int Z, const Vector3r& pos)
        : symbol(sym), atomic_number(Z), position(pos) {}
};

class Pseudopotential {
public:
    Pseudopotential(int Z, const std::vector<Real>& r_grid,
                    const std::vector<Real>& v_local,
                    Real r_cutoff = 2.0);

    int Z() const { return Z_; }
    Real r_cutoff() const { return r_cutoff_; }

    Complex v_local_of_g(Real G_norm) const;

    static std::shared_ptr<Pseudopotential> create(int Z);

private:
    int Z_;
    std::vector<Real> r_grid_;
    std::vector<Real> v_local_;
    Real r_cutoff_;

    void build_spline();
    mutable std::vector<Real> spline_coeffs_;
};

using PseudopotentialPtr = std::shared_ptr<Pseudopotential>;

class Atoms {
public:
    Atoms(const Matrix3r& cell);

    const Matrix3r& cell() const { return cell_; }
    int natoms() const { return static_cast<int>(atoms_.size()); }
    int nelectrons() const { return nelectrons_; }

    void add_atom(const std::string& symbol, const Vector3r& position);

    const Atom& atom(int i) const { return atoms_[i]; }
    const std::vector<Atom>& atoms() const { return atoms_; }

    PseudopotentialPtr pseudopotential(int i) const;
    PseudopotentialPtr pseudopotential(const std::string& symbol) const;

    const std::unordered_map<std::string, PseudopotentialPtr>&
    pseudopotentials() const { return pspots_; }

    int atomic_number(const std::string& symbol) const;

private:
    Matrix3r cell_;
    std::vector<Atom> atoms_;
    std::unordered_map<std::string, PseudopotentialPtr> pspots_;
    int nelectrons_;
};

using AtomsPtr = std::shared_ptr<Atoms>;

}
