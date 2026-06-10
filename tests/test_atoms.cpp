#include "dft_solver/atoms.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dft_solver;

int main() {
    std::cout << "=== Testing Atoms and Pseudopotential ===" << std::endl;

    Matrix3r cell;
    cell << 10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0;

    Atoms atoms(cell);
    assert(atoms.natoms() == 0);
    assert(atoms.nelectrons() == 0);

    Vector3r pos1(5.0, 5.0, 5.0);
    atoms.add_atom("H", pos1);
    assert(atoms.natoms() == 1);
    assert(atoms.nelectrons() == 1);
    std::cout << "Added H atom, nelectrons: " << atoms.nelectrons() << std::endl;

    Vector3r pos2(3.0, 3.0, 3.0);
    atoms.add_atom("He", pos2);
    assert(atoms.natoms() == 2);
    assert(atoms.nelectrons() == 3);
    std::cout << "Added He atom, total electrons: " << atoms.nelectrons() << std::endl;

    const auto& atom0 = atoms.atom(0);
    assert(atom0.symbol == "H");
    assert(atom0.atomic_number == 1);
    std::cout << "Atom 0: " << atom0.symbol << " Z=" << atom0.atomic_number << std::endl;

    int Z_C = atoms.atomic_number("C");
    assert(Z_C == 6);
    std::cout << "Carbon atomic number: " << Z_C << std::endl;

    auto pspot_H = atoms.pseudopotential("H");
    assert(pspot_H->Z() == 1);
    std::cout << "H pseudopotential r_cutoff: " << pspot_H->r_cutoff() << std::endl;

    Complex v_g0 = pspot_H->v_local_of_g(0.0);
    std::cout << "H pseudopotential at G=0: " << v_g0 << std::endl;
    assert(std::abs(v_g0.real()) > 0);

    Complex v_g1 = pspot_H->v_local_of_g(1.0);
    std::cout << "H pseudopotential at |G|=1: " << v_g1 << std::endl;

    auto pspot_create = Pseudopotential::create(8);
    assert(pspot_create->Z() == 8);
    std::cout << "Created O pseudopotential, Z=" << pspot_create->Z() << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
