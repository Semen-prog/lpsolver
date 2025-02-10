#include <lpsolver/cpptocu.hpp>
#include <lpsolver/ludec.hpp>

namespace LPSolver {
    mdata lu_csc_to_csr(const Matrix& A) {
        return csc_to_csr(A.outerSize(), A.nonZeros(), A.innerIndexPtr(), A.outerIndexPtr(), A.valuePtr());
    }

	Vector lu_solve(const Matrix& A, const Vector& b) {
		int n = A.outerSize();
		int nnz = A.nonZeros();
		double* xr = ax_equals_b_solver(n, nnz, lu_csc_to_csr(A), b.data());
		Vector x(n);
		for (int i = 0; i < n; ++i) {
			x(i) = xr[i];
		}
		free(static_cast<void*>(xr));
		return x;
	}
}; // namespace LPSolver
