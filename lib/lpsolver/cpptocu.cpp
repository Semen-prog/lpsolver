#include <lpsolver/cpptocu.hpp>
#include <lpsolver/ludec.hpp>
#include <stdexcept>
#include <vector>

namespace LPSolver {
    mdata lu_csc_to_csr(const Matrix& A) {
        return csc_to_csr(A.outerSize(), A.nonZeros(), A.innerIndexPtr(), A.outerIndexPtr(), A.valuePtr());
    }

    std::vector<Vector> lu_solve(const Matrix& A, const std::vector<Vector> &b) {
		int n = A.outerSize();
		int nnz = A.nonZeros();
		int bsz = b.size();
		std::vector<double> ball;
		for (int i = 0; i < bsz; ++i) {
			for (int j = 0; j < n; ++j) {
				ball.push_back(b[i](j));
			if (std::isnan(b[i](j))) {
				debug_print("!!!: {0} {1} ", i, j);
			}
			}
		}
		
        int status = 1;
		double* xr = ax_equals_b_solver(n, nnz, lu_csc_to_csr(A), bsz, ball.data(), &status);
        if (!status) {
            throw std::runtime_error("Factor is exactly singular.");
        }
	std::vector<Vector> x(bsz, Vector(n));
	for (int i = 0; i < bsz; ++i) {
		for (int j = 0; j < n; ++j) {
			x[i](j) = xr[i * n + j];
		}
	}
		free(static_cast<void*>(xr));
		return x;
	}
}; // namespace LPSolver
