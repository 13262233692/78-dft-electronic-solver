"""Hamiltonian wrapper that interfaces C++ core with SciPy."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import List, Tuple, Optional, Callable
from . import core


def eigen_to_scipy_sparse(mat_eigen):
    """Convert sparse matrix to scipy.sparse.csr_matrix.

    Works with both C++ Eigen sparse matrices and scipy sparse matrices.
    """
    import scipy.sparse as sp
    if sp.issparse(mat_eigen):
        return mat_eigen.tocsr()
    try:
        data, indices, indptr, shape = core.eigen_to_numpy_sparse(mat_eigen)
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    except Exception:
        return sp.csr_matrix(mat_eigen)


class HamiltonianLinearOperator(spla.LinearOperator):
    """Linear operator representing H|psi> for efficient matrix-free diagonalization."""

    def __init__(self, hamiltonian: core.Hamiltonian, kpoint: np.ndarray):
        n = hamiltonian.npw_k(kpoint)
        self._ham = hamiltonian
        self._kpoint = np.asarray(kpoint, dtype=float)
        super().__init__(dtype=np.complex128, shape=(n, n))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x_cpp = np.ascontiguousarray(x, dtype=np.complex128)
        result = np.zeros_like(x_cpp)
        self._ham.apply_inplace(x_cpp, result, self._kpoint)
        return result

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        nvecs = X.shape[1]
        result = np.zeros_like(X)
        for i in range(nvecs):
            result[:, i] = self._matvec(X[:, i])
        return result


class HamiltonianWrapper:
    """High-level wrapper providing SciPy-compatible Hamiltonian interface."""

    def __init__(self, hamiltonian: core.Hamiltonian):
        self._ham = hamiltonian

    @property
    def hamiltonian(self) -> core.Hamiltonian:
        return self._ham

    @property
    def basis(self):
        return self._ham.basis

    @property
    def atoms(self):
        return self._ham.atoms

    def get_matrix(self, kpoint: np.ndarray, sparse: bool = True):
        """Get the Hamiltonian matrix at a given k-point.

        Parameters
        ----------
        kpoint : array_like
            k-point in reciprocal lattice fractional coordinates
        sparse : bool
            If True, return scipy.sparse.csr_matrix; else dense numpy array

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
        """
        kpoint = np.asarray(kpoint, dtype=float)
        if sparse:
            mat_cpp = self._ham.build_matrix(kpoint)
            return eigen_to_scipy_sparse(mat_cpp)
        else:
            return np.asarray(self._ham.build_dense_matrix(kpoint))

    def get_linear_operator(self, kpoint: np.ndarray) -> spla.LinearOperator:
        """Get a matrix-free LinearOperator for H|psi>.

        This is more memory-efficient than building the full matrix for large systems.
        """
        return HamiltonianLinearOperator(self._ham, kpoint)

    def diagonalize(self, kpoint: np.ndarray, n_bands: int,
                    method: str = "eigsh",
                    preconditioner: Optional[spla.LinearOperator] = None,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize the Hamiltonian at a k-point to find lowest eigenvalues.

        Parameters
        ----------
        kpoint : array_like
            k-point in reciprocal lattice fractional coordinates
        n_bands : int
            Number of lowest eigenvalues (bands) to compute
        method : str
            Solver: "eigsh" (ARPACK) or "lobpcg"
        preconditioner : LinearOperator, optional
            Preconditioner for LOBPCG
        **kwargs : dict
            Extra arguments passed to the solver

        Returns
        -------
        eigenvalues : numpy.ndarray
            Shape (n_bands,), sorted ascending
        eigenvectors : numpy.ndarray
            Shape (npw, n_bands), each column is a wavefunction
        """
        kpoint = np.asarray(kpoint, dtype=float)
        n = self._ham.npw_k(kpoint)
        n_bands = min(n_bands, n - 1)

        if method == "eigsh":
            A = self.get_matrix(kpoint, sparse=True)
            if "which" not in kwargs:
                kwargs["which"] = "SA"
            if "tol" not in kwargs:
                kwargs["tol"] = 1e-8
            evals, evecs = spla.eigsh(A, k=n_bands, **kwargs)
            order = np.argsort(evals)
            return evals[order], evecs[:, order]

        elif method == "lobpcg":
            A_op = self.get_linear_operator(kpoint)
            X = np.random.randn(n, n_bands).astype(np.complex128)
            X += 1j * np.random.randn(n, n_bands)
            if "tol" not in kwargs:
                kwargs["tol"] = 1e-8
            if "maxiter" not in kwargs:
                kwargs["maxiter"] = 2000
            if preconditioner is not None:
                kwargs["M"] = preconditioner
            evals, evecs = spla.lobpcg(A_op, X, largest=False, **kwargs)
            order = np.argsort(evals)
            return evals[order], evecs[:, order]

        else:
            raise ValueError(f"Unknown method: {method}. Use 'eigsh' or 'lobpcg'.")

    def kinetic_preconditioner(self, kpoint: np.ndarray) -> spla.LinearOperator:
        """Build a kinetic-energy preconditioner for LOBPCG.

        T^{-1} diagonal preconditioner: M^{-1}_{ii} = 1/(|G+k|^2/2 + shift)
        """
        kpoint = np.asarray(kpoint, dtype=float)
        n = self._ham.npw_k(kpoint)
        t_diag = np.array([self._ham.v_kinetic(ig, kpoint) for ig in range(n)])
        shift = 1.0
        m_diag = 1.0 / (t_diag + shift)
        M = sp.diags(m_diag).tocsr()
        return spla.LinearOperator(shape=(n, n),
                                   matvec=lambda x: M @ x,
                                   dtype=np.complex128)

    def update_density(self, density):
        self._ham.update_density(density)
