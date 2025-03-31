#ifndef CPPTOCU_HPP
#define CPPTOCU_HPP

#include <lpsolver/structs.hpp>
#include <vector>

namespace LPSolver {
	std::vector<Vector> lu_solve(const Matrix&, const std::vector<Vector>&);
}; // namespace LPSolver;

#endif // CPPTOCU_HPP
