#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "dft_solver/types.h"
#include "dft_solver/plane_wave_basis.h"
#include "dft_solver/atoms.h"
#include "dft_solver/density.h"
#include "dft_solver/hamiltonian.h"

namespace py = pybind11;
using namespace dft_solver;

PYBIND11_MODULE(_dft_core, m) {
    m.doc() = "DFT Electronic Solver C++ Core";

    py::class_<GVector>(m, "GVector")
        .def(py::init<>())
        .def(py::init<const Vector3i&, const Vector3r&>())
        .def_readwrite("miller", &GVector::miller)
        .def_readwrite("cartesian", &GVector::cartesian)
        .def_readwrite("norm2", &GVector::norm2);

    py::class_<PlaneWaveBasis, PlaneWaveBasisPtr>(m, "PlaneWaveBasis")
        .def(py::init<const Matrix3r&, Real>())
        .def_property_readonly("cell", &PlaneWaveBasis::cell)
        .def_property_readonly("reciprocal_cell", &PlaneWaveBasis::reciprocal_cell)
        .def_property_readonly("ecut", &PlaneWaveBasis::ecut)
        .def_property_readonly("cell_volume", &PlaneWaveBasis::cell_volume)
        .def_property_readonly("npw", &PlaneWaveBasis::npw)
        .def_property_readonly("ngx", &PlaneWaveBasis::ngx)
        .def_property_readonly("ngy", &PlaneWaveBasis::ngy)
        .def_property_readonly("ngz", &PlaneWaveBasis::ngz)
        .def_property_readonly("g_vectors", &PlaneWaveBasis::g_vectors)
        .def("g_vector", &PlaneWaveBasis::g_vector)
        .def("find_g_vector_index", &PlaneWaveBasis::find_g_vector_index)
        .def("plane_waves", &PlaneWaveBasis::plane_waves)
        .def("g_vectors_shifted", &PlaneWaveBasis::g_vectors_shifted);

    py::class_<Atom>(m, "Atom")
        .def(py::init<>())
        .def(py::init<const std::string&, int, const Vector3r&>())
        .def_readwrite("symbol", &Atom::symbol)
        .def_readwrite("atomic_number", &Atom::atomic_number)
        .def_readwrite("position", &Atom::position);

    py::class_<Pseudopotential, PseudopotentialPtr>(m, "Pseudopotential")
        .def(py::init<int, const std::vector<Real>&, const std::vector<Real>&, Real>(),
             py::arg("Z"), py::arg("r_grid"), py::arg("v_local"),
             py::arg("r_cutoff") = 2.0)
        .def_property_readonly("Z", &Pseudopotential::Z)
        .def_property_readonly("r_cutoff", &Pseudopotential::r_cutoff)
        .def("v_local_of_g", &Pseudopotential::v_local_of_g)
        .def_static("create", &Pseudopotential::create);

    py::class_<Atoms, AtomsPtr>(m, "Atoms")
        .def(py::init<const Matrix3r&>())
        .def_property_readonly("cell", &Atoms::cell)
        .def_property_readonly("natoms", &Atoms::natoms)
        .def_property_readonly("nelectrons", &Atoms::nelectrons)
        .def_property_readonly("atoms", &Atoms::atoms)
        .def("add_atom", &Atoms::add_atom)
        .def("atom", &Atoms::atom)
        .def("pseudopotential",
             py::overload_cast<int>(&Atoms::pseudopotential, py::const_))
        .def("pseudopotential",
             py::overload_cast<const std::string&>(&Atoms::pseudopotential, py::const_))
        .def("atomic_number", &Atoms::atomic_number);

    py::class_<Density, DensityPtr>(m, "Density")
        .def(py::init<PlaneWaveBasisPtr>())
        .def_property_readonly("basis", &Density::basis)
        .def("set_from_eigenstates", &Density::set_from_eigenstates,
             py::arg("psi"), py::arg("occupations"),
             py::arg("kpoint") = Vector3r::Zero())
        .def("set_from_r_space", &Density::set_from_r_space)
        .def_property("rho_g", &Density::rho_g,
            [](Density& d, const VectorXc& rho) {
                auto basis = d.basis();
                Density new_d(basis);
                new_d.set_n_electrons(d.n_electrons());
            })
        .def("rho_r", &Density::rho_r)
        .def("total_electrons", &Density::total_electrons)
        .def_property("n_electrons", &Density::n_electrons, &Density::set_n_electrons)
        .def("hartree_potential_r", &Density::hartree_potential_r)
        .def("hartree_potential_g", &Density::hartree_potential_g)
        .def_static("exchange_correlation_energy", &Density::exchange_correlation_energy)
        .def_static("exchange_correlation_potential", &Density::exchange_correlation_potential)
        .def("v_xc_r", &Density::v_xc_r)
        .def("v_xc_g", &Density::v_xc_g)
        .def("e_xc", &Density::e_xc)
        .def("e_hartree", &Density::e_hartree);

    py::class_<Hamiltonian, HamiltonianPtr>(m, "Hamiltonian")
        .def(py::init<PlaneWaveBasisPtr, AtomsPtr>())
        .def_property_readonly("basis", &Hamiltonian::basis)
        .def_property_readonly("atoms", &Hamiltonian::atoms)
        .def_property_readonly("density", &Hamiltonian::density)
        .def("update_density", &Hamiltonian::update_density)
        .def("build_matrix", &Hamiltonian::build_matrix)
        .def("build_dense_matrix", &Hamiltonian::build_dense_matrix)
        .def("apply", &Hamiltonian::apply)
        .def("apply_inplace", &Hamiltonian::apply_inplace)
        .def("npw_k", &Hamiltonian::npw_k)
        .def("ewald_energy", &Hamiltonian::ewald_energy)
        .def("v_kinetic", &Hamiltonian::v_kinetic)
        .def("v_local", &Hamiltonian::v_local)
        .def("v_local_diag", &Hamiltonian::v_local_diag)
        .def("v_local_sparse", &Hamiltonian::v_local_sparse)
        .def("compute_potential_fourier", &Hamiltonian::compute_potential_fourier)
        .def_property_readonly("v_ion_g", &Hamiltonian::v_ion_g)
        .def("compute_kinetic_energy", &Hamiltonian::compute_kinetic_energy);

    m.def("eigen_to_numpy_sparse", [](const SparseMatrixXc& mat) {
        py::array_t<int> indptr(std::vector<int>(mat.outerIndexPtr(),
                                                   mat.outerIndexPtr() + mat.outerSize() + 1));
        py::array_t<int> indices(std::vector<int>(mat.innerIndexPtr(),
                                                   mat.innerIndexPtr() + mat.nonZeros()));
        py::array_t<std::complex<double>> data(
            std::vector<std::complex<double>>(mat.valuePtr(),
                                               mat.valuePtr() + mat.nonZeros()));
        return py::make_tuple(data, indices, indptr,
                              py::make_tuple(mat.rows(), mat.cols()));
    });
}
