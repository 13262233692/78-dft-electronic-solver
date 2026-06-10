"""
Pure Python fallback implementation of _dft_core.

This provides a working Python implementation of all C++ classes
for testing and development when the C++ extension is not compiled.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math


def _trapz(y, x):
    """NumPy-version-compatible trapezoidal integration."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)


class GVector:
    def __init__(self, miller=None, cartesian=None):
        self.miller = np.array(miller if miller is not None else [0, 0, 0], dtype=int)
        self.cartesian = np.array(cartesian if cartesian is not None else [0.0, 0.0, 0.0], dtype=float)
        self.norm2 = float(np.dot(self.cartesian, self.cartesian))


class PlaneWaveBasis:
    def __init__(self, cell: np.ndarray, ecut: float):
        self._cell = np.asarray(cell, dtype=float)
        self._ecut = float(ecut)
        self._cell_volume = abs(np.linalg.det(self._cell))

        two_pi = 2.0 * math.pi
        a1, a2, a3 = self._cell[:, 0], self._cell[:, 1], self._cell[:, 2]
        self._reciprocal_cell = np.zeros((3, 3))
        self._reciprocal_cell[:, 0] = two_pi * np.cross(a2, a3) / self._cell_volume
        self._reciprocal_cell[:, 1] = two_pi * np.cross(a3, a1) / self._cell_volume
        self._reciprocal_cell[:, 2] = two_pi * np.cross(a1, a2) / self._cell_volume

        self._build_g_vectors()

    def _build_g_vectors(self):
        two_ecut = 2.0 * self._ecut
        gmax = math.sqrt(two_ecut)

        b1, b2, b3 = self._reciprocal_cell[:, 0], self._reciprocal_cell[:, 1], self._reciprocal_cell[:, 2]

        self._ngx = int(math.ceil(gmax / np.linalg.norm(b1))) + 1
        self._ngy = int(math.ceil(gmax / np.linalg.norm(b2))) + 1
        self._ngz = int(math.ceil(gmax / np.linalg.norm(b3))) + 1

        g_vectors = []
        for i in range(-self._ngx, self._ngx + 1):
            for j in range(-self._ngy, self._ngy + 1):
                for k in range(-self._ngz, self._ngz + 1):
                    cart = i * b1 + j * b2 + k * b3
                    norm2 = float(np.dot(cart, cart))
                    if norm2 <= two_ecut:
                        gv = GVector([i, j, k], cart)
                        g_vectors.append(gv)

        g_vectors.sort(key=lambda g: g.norm2)
        self._g_vectors = g_vectors
        self._miller_to_idx = {tuple(g.miller): i for i, g in enumerate(g_vectors)}

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @property
    def reciprocal_cell(self) -> np.ndarray:
        return self._reciprocal_cell

    @property
    def ecut(self) -> float:
        return self._ecut

    @property
    def cell_volume(self) -> float:
        return self._cell_volume

    @property
    def npw(self) -> int:
        return len(self._g_vectors)

    @property
    def g_vectors(self) -> List[GVector]:
        return self._g_vectors

    @property
    def ngx(self) -> int:
        return self._ngx

    @property
    def ngy(self) -> int:
        return self._ngy

    @property
    def ngz(self) -> int:
        return self._ngz

    def g_vector(self, i: int) -> GVector:
        return self._g_vectors[i]

    def find_g_vector_index(self, miller) -> int:
        return self._miller_to_idx.get(tuple(miller), -1)

    def plane_waves(self, k: np.ndarray, r_list: List[np.ndarray]) -> np.ndarray:
        nr = len(r_list)
        npw = self.npw
        result = np.zeros((nr, npw), dtype=complex)
        k = np.asarray(k, dtype=float)
        for ir, r in enumerate(r_list):
            for ig, g in enumerate(self._g_vectors):
                gk = g.cartesian + k
                phase = float(np.dot(gk, r))
                result[ir, ig] = complex(math.cos(phase), math.sin(phase))
        return result.flatten()

    def g_vectors_shifted(self, k: np.ndarray) -> List[GVector]:
        k = np.asarray(k, dtype=float)
        result = []
        for g in self._g_vectors:
            cart = g.cartesian + k
            gk = GVector(g.miller.copy(), cart)
            gk.norm2 = float(np.dot(cart, cart))
            result.append(gk)
        return result


class Atom:
    def __init__(self, symbol: str = "", atomic_number: int = 0, position=None):
        self.symbol = symbol
        self.atomic_number = int(atomic_number)
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=float)


ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
    "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36
}


class Pseudopotential:
    def __init__(self, Z: int, r_grid: List[float], v_local: List[float], r_cutoff: float = 2.0):
        self._Z = int(Z)
        self._r_grid = np.asarray(r_grid, dtype=float)
        self._v_local = np.asarray(v_local, dtype=float)
        self._r_cutoff = float(r_cutoff)

    @property
    def Z(self) -> int:
        return self._Z

    @property
    def r_cutoff(self) -> float:
        return self._r_cutoff

    def v_local_of_g(self, G_norm: float) -> complex:
        if G_norm < 1e-10:
            integral = _trapz(self._v_local * self._r_grid**2, self._r_grid)
            return 4.0 * math.pi * integral + 0j

        r = self._r_grid
        v = self._v_local
        integrand = v * r * np.sin(G_norm * r) / G_norm
        integral = _trapz(integrand, r)
        return 4.0 * math.pi * integral + 0j

    @staticmethod
    def create(Z: int) -> "Pseudopotential":
        r_max = 5.0
        n_points = 200
        r_grid = np.linspace(1e-3, r_max, n_points)
        r_c = 1.5
        v_local = -float(Z) / r_grid * [math.erf(r / (math.sqrt(2.0) * r_c)) for r in r_grid]
        return Pseudopotential(Z, r_grid.tolist(), v_local.tolist(), 2.0)


class Atoms:
    def __init__(self, cell: np.ndarray):
        self._cell = np.asarray(cell, dtype=float)
        self._atoms: List[Atom] = []
        self._pspots: Dict[str, Pseudopotential] = {}
        self._nelectrons = 0

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @property
    def natoms(self) -> int:
        return len(self._atoms)

    def nelectrons(self) -> int:
        return self._nelectrons

    @property
    def atoms(self) -> List[Atom]:
        return self._atoms

    @property
    def pseudopotentials(self) -> Dict[str, Pseudopotential]:
        return self._pspots

    def atomic_number(self, symbol: str) -> int:
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element: {symbol}")
        return ATOMIC_NUMBERS[symbol]

    def add_atom(self, symbol: str, position: np.ndarray):
        Z = self.atomic_number(symbol)
        self._atoms.append(Atom(symbol, Z, position))
        self._nelectrons += Z
        if symbol not in self._pspots:
            self._pspots[symbol] = Pseudopotential.create(Z)

    def atom(self, i: int) -> Atom:
        return self._atoms[i]

    def pseudopotential(self, arg):
        if isinstance(arg, int):
            return self._pspots[self._atoms[arg].symbol]
        elif isinstance(arg, str):
            return self._pspots[arg]
        else:
            raise TypeError("Argument must be int or str")


class Density:
    def __init__(self, basis: PlaneWaveBasis):
        self._basis = basis
        self._n_electrons = 0.0
        self._rho_g = np.zeros(basis.npw, dtype=complex)

    @property
    def basis(self) -> PlaneWaveBasis:
        return self._basis

    @property
    def n_electrons(self) -> float:
        return self._n_electrons

    @n_electrons.setter
    def n_electrons(self, n: float):
        self._n_electrons = float(n)

    def set_n_electrons(self, n: float):
        self._n_electrons = float(n)

    @property
    def rho_g(self) -> np.ndarray:
        return self._rho_g

    def set_from_r_space(self, rho_r: np.ndarray):
        self._rho_g = self._fft_r_to_g(np.asarray(rho_r, dtype=float))
        self._normalize()

    def set_from_eigenstates(self, psi: List[np.ndarray], occupations: List[float],
                              kpoint=None):
        if kpoint is None:
            kpoint = np.zeros(3)
        kpoint = np.asarray(kpoint, dtype=float)

        npw = self._basis.npw
        self._rho_g = np.zeros(npw, dtype=complex)
        gk = self._basis.g_vectors_shifted(kpoint)

        for ib, psi_b in enumerate(psi):
            psi_b = np.asarray(psi_b, dtype=complex)
            f = occupations[ib]
            for ig1 in range(npw):
                c1 = np.conj(psi_b[ig1])
                for ig2 in range(npw):
                    c2 = psi_b[ig2]
                    diff_miller = tuple(gk[ig1].miller - gk[ig2].miller)
                    idx = self._basis.find_g_vector_index(diff_miller)
                    if idx >= 0:
                        self._rho_g[idx] += f * c1 * c2

        self._rho_g /= self._basis.cell_volume
        self._normalize()

    def _normalize(self):
        if self._n_electrons > 0:
            current_ne = self._rho_g[0].real * self._basis.cell_volume
            if abs(current_ne) > 1e-15:
                self._rho_g *= self._n_electrons / current_ne

    def rho_r(self) -> np.ndarray:
        return self._fft_g_to_r(self._rho_g)

    def total_electrons(self) -> float:
        return self._rho_g[0].real * self._basis.cell_volume

    def hartree_potential_g(self) -> np.ndarray:
        npw = self._basis.npw
        v_h = np.zeros(npw, dtype=complex)
        gs = self._basis.g_vectors
        for ig in range(npw):
            g2 = gs[ig].norm2
            if g2 > 1e-12:
                v_h[ig] = 4.0 * math.pi * self._rho_g[ig] / g2
        return v_h

    def hartree_potential_r(self) -> np.ndarray:
        return self._fft_g_to_r(self.hartree_potential_g())

    @staticmethod
    def exchange_correlation_energy(rho: float) -> float:
        if rho < 1e-20:
            return 0.0
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        ex = -3.0 / (4.0 * math.pi) * (9.0 * math.pi / 4.0) ** (1.0 / 3.0) / rs
        if rs >= 1.0:
            gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
            ec = gamma / (1.0 + beta1 * math.sqrt(rs) + beta2 * rs)
        else:
            A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
            ec = A * math.log(rs) + B + C * rs * math.log(rs) + D * rs
        return ex + ec

    @staticmethod
    def exchange_correlation_potential(rho: float) -> float:
        if rho < 1e-20:
            return 0.0
        rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        drho_drs = -(3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * rs ** (-4.0 / 3.0)

        ex = -3.0 / (4.0 * math.pi) * (9.0 * math.pi / 4.0) ** (1.0 / 3.0) / rs
        dex_drs = 3.0 / (4.0 * math.pi) * (9.0 * math.pi / 4.0) ** (1.0 / 3.0) / (rs * rs)

        if rs >= 1.0:
            gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
            denom = 1.0 + beta1 * math.sqrt(rs) + beta2 * rs
            ec = gamma / denom
            dec_drs = -gamma * (beta1 / (2.0 * math.sqrt(rs)) + beta2) / (denom * denom)
        else:
            A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
            ec = A * math.log(rs) + B + C * rs * math.log(rs) + D * rs
            dec_drs = A / rs + C * (math.log(rs) + 1.0) + D

        exc = ex + ec
        dexc_drs = dex_drs + dec_drs
        return exc - rho * dexc_drs / drho_drs

    def v_xc_r(self) -> np.ndarray:
        rho = self.rho_r()
        vxc = np.zeros_like(rho)
        for i in range(rho.size):
            vxc[i] = self.exchange_correlation_potential(rho[i])
        return vxc

    def v_xc_g(self) -> np.ndarray:
        return self._fft_r_to_g(self.v_xc_r())

    def e_xc(self) -> float:
        rho = self.rho_r()
        volume = self._basis.cell_volume
        dV = volume / rho.size
        exc = 0.0
        for i in range(rho.size):
            exc += rho[i] * self.exchange_correlation_energy(rho[i]) * dV
        return exc

    def e_hartree(self) -> float:
        npw = self._basis.npw
        volume = self._basis.cell_volume
        eh = 0.0
        gs = self._basis.g_vectors
        for ig in range(1, npw):
            g2 = gs[ig].norm2
            eh += 2.0 * math.pi * abs(self._rho_g[ig]) ** 2 / g2
        return eh * volume

    def _fft_r_to_g(self, r_space: np.ndarray) -> np.ndarray:
        ngx, ngy, ngz = self._basis.ngx, self._basis.ngy, self._basis.ngz
        nr = ngx * ngy * ngz
        npw = self._basis.npw

        a1 = self._basis.cell[:, 0]
        a2 = self._basis.cell[:, 1]
        a3 = self._basis.cell[:, 2]
        gs = self._basis.g_vectors

        g_space = np.zeros(npw, dtype=complex)
        for ig in range(npw):
            g = gs[ig].cartesian
            total = 0.0 + 0.0j
            for ix in range(ngx):
                for iy in range(ngy):
                    for iz in range(ngz):
                        ir = ix * ngy * ngz + iy * ngz + iz
                        r = (ix / ngx) * a1 + (iy / ngy) * a2 + (iz / ngz) * a3
                        phase = float(np.dot(g, r))
                        total += r_space[ir] * complex(math.cos(phase), -math.sin(phase))
            g_space[ig] = total / nr
        return g_space

    def _fft_g_to_r(self, g_space: np.ndarray) -> np.ndarray:
        ngx, ngy, ngz = self._basis.ngx, self._basis.ngy, self._basis.ngz
        nr = ngx * ngy * ngz
        npw = self._basis.npw

        a1 = self._basis.cell[:, 0]
        a2 = self._basis.cell[:, 1]
        a3 = self._basis.cell[:, 2]
        gs = self._basis.g_vectors

        r_space = np.zeros(nr, dtype=float)
        for ix in range(ngx):
            for iy in range(ngy):
                for iz in range(ngz):
                    ir = ix * ngy * ngz + iy * ngz + iz
                    r = (ix / ngx) * a1 + (iy / ngy) * a2 + (iz / ngz) * a3
                    total = 0.0 + 0.0j
                    for ig in range(npw):
                        g = gs[ig].cartesian
                        phase = float(np.dot(g, r))
                        total += g_space[ig] * complex(math.cos(phase), math.sin(phase))
                    r_space[ir] = total.real
        return r_space


class Hamiltonian:
    def __init__(self, basis: PlaneWaveBasis, atoms: Atoms):
        self._basis = basis
        self._atoms = atoms
        self._density: Optional[Density] = None
        npw = basis.npw
        self._v_ion_g = np.zeros(npw, dtype=complex)
        self._v_hartree_g = np.zeros(npw, dtype=complex)
        self._v_xc_g = np.zeros(npw, dtype=complex)
        self._ewald_energy = 0.0
        self._build_ionic_potential()
        self._compute_ewald_energy()

    @property
    def basis(self) -> PlaneWaveBasis:
        return self._basis

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def density(self) -> Optional[Density]:
        return self._density

    @property
    def v_ion_g(self) -> np.ndarray:
        return self._v_ion_g

    def _build_ionic_potential(self):
        npw = self._basis.npw
        gs = self._basis.g_vectors
        pspots = self._atoms.pseudopotentials

        for ig in range(npw):
            struct_fac = 0.0 + 0.0j
            G_norm = math.sqrt(gs[ig].norm2)
            for atom in self._atoms.atoms:
                phase = float(np.dot(gs[ig].cartesian, atom.position))
                struct_fac += complex(math.cos(phase), -math.sin(phase))
            if pspots:
                pspot = next(iter(pspots.values()))
                self._v_ion_g[ig] = struct_fac * pspot.v_local_of_g(G_norm) / self._basis.cell_volume

    def _compute_ewald_energy(self):
        eta = 1.0
        volume = self._basis.cell_volume
        natoms = self._atoms.natoms
        ewald = 0.0

        for i in range(natoms):
            for j in range(natoms):
                Zi = self._atoms.atom(i).atomic_number
                Zj = self._atoms.atom(j).atomic_number
                rij = self._atoms.atom(i).position - self._atoms.atom(j).position
                if i == j:
                    ewald -= Zi * Zj * math.sqrt(eta / math.pi)
                else:
                    r = float(np.linalg.norm(rij))
                    if r > 1e-8:
                        ewald += 0.5 * Zi * Zj * math.erfc(math.sqrt(eta) * r) / r

        ng_max = 10
        b1, b2, b3 = self._basis.reciprocal_cell[:, 0], self._basis.reciprocal_cell[:, 1], self._basis.reciprocal_cell[:, 2]

        for m1 in range(-ng_max, ng_max + 1):
            for m2 in range(-ng_max, ng_max + 1):
                for m3 in range(-ng_max, ng_max + 1):
                    if m1 == 0 and m2 == 0 and m3 == 0:
                        continue
                    G = m1 * b1 + m2 * b2 + m3 * b3
                    G2 = float(np.dot(G, G))
                    factor = 4.0 * math.pi / volume * math.exp(-G2 / (4.0 * eta)) / G2
                    S = 0.0 + 0.0j
                    for i in range(natoms):
                        Zi = self._atoms.atom(i).atomic_number
                        phase = float(np.dot(G, self._atoms.atom(i).position))
                        S += Zi * complex(math.cos(phase), -math.sin(phase))
                    ewald += 0.5 * factor * abs(S) ** 2

        self._ewald_energy = ewald

    def update_density(self, density: Density):
        self._density = density
        self._v_hartree_g = density.hartree_potential_g()
        self._v_xc_g = density.v_xc_g()

    def npw_k(self, kpoint) -> int:
        return self._basis.npw

    def ewald_energy(self) -> float:
        return self._ewald_energy

    def v_kinetic(self, ig: int, kpoint) -> float:
        kpoint = np.asarray(kpoint, dtype=float)
        gk = self._basis.g_vector(ig).cartesian + kpoint
        return 0.5 * float(np.dot(gk, gk))

    def v_local(self, ig1: int, ig2: int, kpoint) -> complex:
        gs = self._basis.g_vectors
        diff_miller = tuple(gs[ig1].miller - gs[ig2].miller)
        idx = self._basis.find_g_vector_index(diff_miller)
        if idx < 0:
            return 0.0 + 0.0j
        result = self._v_ion_g[idx]
        if self._density is not None:
            result += self._v_hartree_g[idx] + self._v_xc_g[idx]
        return complex(result)

    def v_local_diag(self, kpoint) -> np.ndarray:
        npw = self._basis.npw
        diag = np.zeros(npw, dtype=complex)
        for ig in range(npw):
            diag[ig] = self.v_local(ig, ig, kpoint)
        return diag

    def v_local_sparse(self, kpoint):
        import scipy.sparse as sp
        npw = self._basis.npw
        rows, cols, data = [], [], []
        for ig1 in range(npw):
            for ig2 in range(npw):
                v = self.v_local(ig1, ig2, kpoint)
                if abs(v) > 1e-15:
                    rows.append(ig1)
                    cols.append(ig2)
                    data.append(v)
        return sp.csr_matrix((data, (rows, cols)), shape=(npw, npw))

    def build_matrix(self, kpoint):
        import scipy.sparse as sp
        npw = self._basis.npw
        rows, cols, data = [], [], []

        for ig in range(npw):
            t = self.v_kinetic(ig, kpoint)
            rows.append(ig)
            cols.append(ig)
            data.append(complex(t, 0.0))

        for ig1 in range(npw):
            for ig2 in range(npw):
                v = self.v_local(ig1, ig2, kpoint)
                if ig1 == ig2 or abs(v) > 1e-15:
                    rows.append(ig1)
                    cols.append(ig2)
                    data.append(v)

        return sp.csr_matrix((data, (rows, cols)), shape=(npw, npw))

    def build_dense_matrix(self, kpoint) -> np.ndarray:
        npw = self._basis.npw
        mat = np.zeros((npw, npw), dtype=complex)
        for ig in range(npw):
            mat[ig, ig] = self.v_kinetic(ig, kpoint)
        for ig1 in range(npw):
            for ig2 in range(npw):
                mat[ig1, ig2] += self.v_local(ig1, ig2, kpoint)
        return mat

    def apply(self, psi: np.ndarray, kpoint) -> np.ndarray:
        npw = self._basis.npw
        result = np.zeros(npw, dtype=complex)
        self.apply_inplace(psi, result, kpoint)
        return result

    def apply_inplace(self, psi_in: np.ndarray, psi_out: np.ndarray, kpoint):
        kpoint = np.asarray(kpoint, dtype=float)
        npw = self._basis.npw
        gs = self._basis.g_vectors

        psi_in = np.asarray(psi_in, dtype=complex)
        psi_out[:] = 0.0

        for ig in range(npw):
            psi_out[ig] = self.v_kinetic(ig, kpoint) * psi_in[ig]

        ngx, ngy, ngz = self._basis.ngx, self._basis.ngy, self._basis.ngz
        nr = ngx * ngy * ngz
        a1, a2, a3 = self._basis.cell[:, 0], self._basis.cell[:, 1], self._basis.cell[:, 2]

        psi_r = np.zeros(nr, dtype=complex)
        for ix in range(ngx):
            for iy in range(ngy):
                for iz in range(ngz):
                    ir = ix * ngy * ngz + iy * ngz + iz
                    r = (ix / ngx) * a1 + (iy / ngy) * a2 + (iz / ngz) * a3
                    total = 0.0 + 0.0j
                    for ig in range(npw):
                        gk = gs[ig].cartesian + kpoint
                        phase = float(np.dot(gk, r))
                        total += psi_in[ig] * complex(math.cos(phase), math.sin(phase))
                    psi_r[ir] = total

        v_total_r = np.zeros(nr, dtype=float)
        if self._density is not None:
            v_total_r = self._density.hartree_potential_r() + self._density.v_xc_r()

        psi_r *= v_total_r

        for ig in range(npw):
            gk = gs[ig].cartesian + kpoint
            total = 0.0 + 0.0j
            for ix in range(ngx):
                for iy in range(ngy):
                    for iz in range(ngz):
                        ir = ix * ngy * ngz + iy * ngz + iz
                        r = (ix / ngx) * a1 + (iy / ngy) * a2 + (iz / ngz) * a3
                        phase = float(-np.dot(gk, r))
                        total += psi_r[ir] * complex(math.cos(phase), math.sin(phase))
            psi_out[ig] += total / nr

        for ig1 in range(npw):
            for ig2 in range(npw):
                diff_miller = tuple(gs[ig1].miller - gs[ig2].miller)
                idx = self._basis.find_g_vector_index(diff_miller)
                if idx >= 0 and abs(self._v_ion_g[idx]) > 1e-15:
                    psi_out[ig1] += self._v_ion_g[idx] * psi_in[ig2]

    def compute_potential_fourier(self):
        pass

    def compute_kinetic_energy(self, psi: List[np.ndarray], occ: List[float], kpoint) -> float:
        kpoint = np.asarray(kpoint, dtype=float)
        npw = self._basis.npw
        gs = self._basis.g_vectors
        ekin = 0.0
        for ib, psi_b in enumerate(psi):
            psi_b = np.asarray(psi_b, dtype=complex)
            f = occ[ib]
            for ig in range(npw):
                gk = gs[ig].cartesian + kpoint
                ekin += f * 0.5 * float(np.dot(gk, gk)) * abs(psi_b[ig]) ** 2
        return ekin * self._basis.cell_volume


def eigen_to_numpy_sparse(mat):
    import scipy.sparse as sp
    mat_coo = mat.tocoo()
    return (mat_coo.data, mat_coo.row, mat_coo.col, (mat.shape[0], mat.shape[1]))
