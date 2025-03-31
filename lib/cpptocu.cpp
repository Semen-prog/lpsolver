#include <lpsolver/cpptocu.hpp>
#include <lpsolver/ludec.hpp>
#include <stdexcept>
#include <vector>

namespace LPSolver {
    std::vector<Vector> lu_solve(const Matrix& A, const std::vector<Vector> &bs) {
		int n = A.outerSize();
		int nnz = A.nonZeros();
		int rhs_size = bs.size();
		std::vector<double> b_all;
		for (int i = 0; i < rhs_size; ++i) {
			for (int j = 0; j < n; ++j) {
				b_all.push_back(bs[i](j));
			}
		}
		
        double* x_all = new double[rhs_size * n];
		int status = ax_equals_b_solver(n, nnz, A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), rhs_size, b_all.data(), x_all);
        if (!status) {
            throw std::runtime_error("Factor is exactly singular.");
        }

		std::vector<Vector> xs(rhs_size, Vector(n));
		for (int i = 0; i < rhs_size; ++i) {
			for (int j = 0; j < n; ++j) {
				xs[i](j) = x_all[i * n + j];
			}
		}

		free(x_all);
		return xs;
	}
}; // namespace LPSolver
